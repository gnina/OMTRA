import os
import sys
import shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import redis
from rq import Queue
import uuid
from datetime import datetime, timedelta
import json
import aiofiles
import tempfile
import time

# Add shared module to path
sys.path.append(str(Path(__file__).parent.parent))

from shared.models import (
    JobSubmission, JobResponse, JobStatusResponse, JobResultResponse,
    UploadInitResponse, ArtifactInfo, JobStatus, generate_job_id, 
    generate_upload_token, SamplingParams
)
from shared.file_utils import (
    validate_file_safety, FileValidationError, create_job_directory,
    save_uploaded_file, list_job_outputs, create_zip_archive, get_job_directory
)
from shared.logging_utils import logger, log_api_request, log_file_upload, log_job_event

# Configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 8000))
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 26214400))
MAX_FILES_PER_JOB = int(os.getenv('MAX_FILES_PER_JOB', 3))
JOB_TTL_HOURS = int(os.getenv('JOB_TTL_HOURS', 48))

# Checkpoint configuration
CHECKPOINT_DIR = Path(os.getenv('CHECKPOINT_DIR', '/srv/app/checkpoints'))
CHECKPOINT_MAPPING = {
    "Unconditional": "uncond.ckpt",
    "Pharmacophore-conditioned": "phcond.ckpt", 
    "Protein-conditioned": "protcond.ckpt"
}

# Initialize Redis and RQ
redis_conn = redis.from_url(REDIS_URL)
task_queue = Queue('omtra_tasks', connection=redis_conn, default_timeout='180s')

# FastAPI app
app = FastAPI(
    title="OMTRA Molecule Sampler API",
    description="API for molecule sampling and generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Upload token storage (in production, use Redis or database)
upload_tokens = {}

class RequestLoggingMiddleware:
    """Middleware to log API requests"""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    duration_ms = (time.time() - start_time) * 1000
                    log_api_request(
                        logger,
                        method=scope["method"],
                        path=scope["path"],
                        status_code=message["status"],
                        duration_ms=duration_ms
                    )
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

app.add_middleware(RequestLoggingMiddleware)


def pydantic_to_dict(model_obj):
    """Return a dict from a Pydantic model for v1 or v2."""
    try:
        # Pydantic v2
        return model_obj.model_dump()
    except AttributeError:
        # Pydantic v1
        return model_obj.dict()


def get_checkpoint_path(sampling_mode: str) -> Optional[Path]:
    """Get the checkpoint path for a given sampling mode"""
    checkpoint_name = CHECKPOINT_MAPPING.get(sampling_mode)
    if not checkpoint_name:
        return None
    
    checkpoint_path = CHECKPOINT_DIR / checkpoint_name
    return checkpoint_path if checkpoint_path.exists() else None


def validate_checkpoint_availability(sampling_mode: str) -> bool:
    """Check if checkpoint is available for the given sampling mode"""
    checkpoint_path = get_checkpoint_path(sampling_mode)
    return checkpoint_path is not None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        redis_conn.ping()
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.post("/upload/init", response_model=UploadInitResponse)
async def init_upload():
    """Initialize file upload"""
    upload_token = generate_upload_token()
    upload_tokens[upload_token] = {
        'created_at': datetime.utcnow(),
        'used': False
    }
    
    return UploadInitResponse(
        upload_token=upload_token,
        max_file_size=MAX_FILE_SIZE
    )


@app.post("/upload/{upload_token}")
async def upload_file(upload_token: str, file: UploadFile = File(...)):
    """Upload a file using upload token"""
    
    # Validate upload token
    if upload_token not in upload_tokens:
        raise HTTPException(status_code=400, detail="Invalid upload token")
    
    token_data = upload_tokens[upload_token]
    if token_data['used']:
        raise HTTPException(status_code=400, detail="Upload token already used")
    
    # Check token expiry (1 hour)
    if datetime.utcnow() - token_data['created_at'] > timedelta(hours=1):
        raise HTTPException(status_code=400, detail="Upload token expired")
    
    try:
        # Read file content
        content = await file.read()
        
        # Validate file
        validation_result = validate_file_safety(file.filename, content)
        
        # Store file temporarily with upload token
        temp_dir = Path(f"/tmp/uploads/{upload_token}")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_file_path = temp_dir / validation_result['safe_filename']
        async with aiofiles.open(temp_file_path, 'wb') as f:
            await f.write(content)
        
        # Mark token as used and store file info
        upload_tokens[upload_token].update({
            'used': True,
            'file_path': str(temp_file_path),
            'validation_result': validation_result
        })
        
        # Log upload
        log_file_upload(
            logger,
            filename=file.filename,
            size=validation_result['size'],
            sha256=validation_result['sha256']
        )
        
        return {
            "message": "File uploaded successfully",
            "filename": validation_result['safe_filename'],
            "size": validation_result['size'],
            "upload_token": upload_token
        }
        
    except FileValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")


@app.post("/sample", response_model=JobResponse)
async def submit_job(job_data: JobSubmission):
    """Submit a sampling job"""
    
    try:
        # Use custom job ID if provided, otherwise generate one
        job_id = job_data.job_id if job_data.job_id else generate_job_id()
        
        # Validate custom job ID if provided
        if job_data.job_id:
            if not job_data.job_id.strip():
                raise HTTPException(status_code=400, detail="Custom job ID cannot be empty")
            if len(job_data.job_id) > 100:
                raise HTTPException(status_code=400, detail="Custom job ID too long (max 100 characters)")
            # Check if job ID already exists
            if redis_conn.exists(f"job:{job_id}"):
                raise HTTPException(status_code=400, detail=f"Job ID '{job_id}' already exists")
    except HTTPException as he:
        logger.error(f"HTTPException in job submission: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Error processing job submission: {e}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    
    try:
        # Validate sampling mode and checkpoint availability
        sampling_mode = job_data.params.sampling_mode
        if not validate_checkpoint_availability(sampling_mode):
            checkpoint_name = CHECKPOINT_MAPPING.get(sampling_mode, "unknown")
            raise HTTPException(
                status_code=400, 
                detail=f"Checkpoint not available for {sampling_mode} mode. Expected: {checkpoint_name}"
            )
        
        # Validate upload tokens and move files
        input_files = []
        for upload_token in job_data.uploads:
            if upload_token not in upload_tokens:
                raise HTTPException(status_code=400, detail=f"Invalid upload token: {upload_token}")
            
            token_data = upload_tokens[upload_token]
            if not token_data['used'] or 'file_path' not in token_data:
                raise HTTPException(status_code=400, detail=f"Upload token not used: {upload_token}")
            
            # Move file to job directory
            temp_file_path = Path(token_data['file_path'])
            if temp_file_path.exists():
                job_dir = create_job_directory(job_id)
                job_file_path = job_dir / "inputs" / temp_file_path.name
                shutil.move(str(temp_file_path), str(job_file_path))
                
                input_files.append({
                    'filename': temp_file_path.name,
                    'path': str(job_file_path),
                    **token_data['validation_result']
                })
            
            # Clean up upload token
            del upload_tokens[upload_token]
        
        # Additional validation for conditional modes
        if sampling_mode in ["Pharmacophore-conditioned", "Protein-conditioned"] and len(input_files) == 0:
            raise HTTPException(status_code=400, detail=f"{sampling_mode} requires at least one input file (e.g., pharmacophore .xyz or protein files)")
        
        # Get checkpoint path and add to params
        checkpoint_path = get_checkpoint_path(sampling_mode)
        params_dict = pydantic_to_dict(job_data.params)
        params_dict['checkpoint_path'] = str(checkpoint_path)
        
        # Override n_samples with num_samples from the request if provided
        if hasattr(job_data, 'num_samples') and job_data.num_samples is not None:
            params_dict['n_samples'] = job_data.num_samples
        
        # Use GPU if available, otherwise CPU
        try:
            if os.getenv('OMTRA_MODEL_AVAILABLE', 'false').lower() == 'false' or os.getenv('ENVIRONMENT', 'local').lower() == 'local':
                # Check if CUDA is available by checking environment variables
                cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', '')
                if cuda_visible_devices and cuda_visible_devices != '':
                    params_dict['device'] = 'cuda'
                else:
                    params_dict['device'] = 'cpu'
        except Exception:
            pass

        # Submit job to queue
        job = task_queue.enqueue(
            'worker.sampling_task',
            kwargs={
                'job_id': job_id,
                'params': params_dict,
                'input_files': input_files,
            },
            job_timeout='180s'
        )
        
        # Store job metadata
        job_metadata = {
            'job_id': job_id,
            'rq_job_id': job.id,
            'params': params_dict,
            'input_files': input_files,
            'created_at': datetime.utcnow().isoformat(),
            'status': JobStatus.QUEUED
        }
        
        redis_conn.setex(
            f"job:{job_id}",
            timedelta(hours=JOB_TTL_HOURS),
            json.dumps(job_metadata)
        )
        
        log_job_event(logger, job_id, "job_submitted", params=pydantic_to_dict(job_data.params))
        
        return JobResponse(job_id=job_id)
        
    except Exception as e:
        logger.error(f"Job submission failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Job submission failed: {str(e)}")


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status"""
    
    try:
        # Get job metadata
        job_data = redis_conn.get(f"job:{job_id}")
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_metadata = json.loads(job_data)
        
        # Get RQ job status
        from rq.job import Job
        try:
            rq_job = Job.fetch(job_metadata['rq_job_id'], connection=redis_conn)
            
            # Map RQ status to our status
            status_mapping = {
                'queued': JobStatus.QUEUED,
                'started': JobStatus.RUNNING,
                'finished': JobStatus.SUCCEEDED,
                'failed': JobStatus.FAILED,
                'canceled': JobStatus.CANCELED,
                'deferred': JobStatus.QUEUED
            }
            
            rq_raw_status = None
            try:
                rq_raw_status = rq_job.get_status()
            except Exception:
                # Fallback for older rq
                rq_raw_status = getattr(rq_job, 'status', None)
            status = status_mapping.get(rq_raw_status, JobStatus.QUEUED)
            
            # Calculate progress and timing
            progress = 0
            if status == JobStatus.RUNNING:
                progress = 50  # Simple progress estimation
            elif status == JobStatus.SUCCEEDED:
                progress = 100
            
            started_at = getattr(rq_job, 'started_at', None)
            completed_at = getattr(rq_job, 'ended_at', None)
            elapsed_seconds = None
            
            if started_at:
                end_time = completed_at or datetime.utcnow()
                elapsed_seconds = (end_time - started_at).total_seconds()
            
            is_failed = False
            try:
                is_failed = rq_job.is_failed
            except Exception:
                # Older rq exposes get_status() == 'failed'
                is_failed = (rq_raw_status == 'failed')
            
            return JobStatusResponse(
                job_id=job_id,
                state=status,
                progress=progress,
                message=rq_job.exc_info if is_failed else None,
                started_at=started_at,
                completed_at=completed_at,
                elapsed_seconds=elapsed_seconds
            )
            
        except Exception as e:
            logger.warning(f"Failed to fetch RQ job {job_metadata['rq_job_id']}: {e}")
            return JobStatusResponse(
                job_id=job_id,
                state=JobStatus.FAILED,
                message="Job status unknown"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")


@app.get("/result/{job_id}", response_model=JobResultResponse)
async def get_job_result(job_id: str):
    """Get job results"""
    
    try:
        # Get job metadata
        job_data = redis_conn.get(f"job:{job_id}")
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_metadata = json.loads(job_data)
        
        # Get current status
        status_response = await get_job_status(job_id)
        
        if status_response.state != JobStatus.SUCCEEDED:
            return JobResultResponse(
                job_id=job_id,
                state=status_response.state,
                params=SamplingParams(**job_metadata['params']),
                error_message=status_response.message
            )
        
        # List output files
        output_files = list_job_outputs(job_id)
        artifacts = []
        
        for file_path in output_files:
            if file_path.is_file():
                artifacts.append(ArtifactInfo(
                    id=str(uuid.uuid4()),
                    filename=file_path.name,
                    format=file_path.suffix[1:] if file_path.suffix else "unknown",
                    size=file_path.stat().st_size,
                    path_or_url=f"/download/{job_id}/{file_path.name}"
                ))
        
        return JobResultResponse(
            job_id=job_id,
            state=status_response.state,
            artifacts=artifacts,
            logs_url=f"/logs/{job_id}",
            params=SamplingParams(**job_metadata['params']),
            elapsed_seconds=status_response.elapsed_seconds
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Result retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Result retrieval failed")


@app.get("/download/{job_id}/all")
async def download_all_outputs(job_id: str, background_tasks: BackgroundTasks):
    """Download all outputs as ZIP"""
    
    zip_path = create_zip_archive(job_id)
    if not zip_path or not zip_path.exists():
        raise HTTPException(status_code=404, detail="No outputs to download")
    
    # Schedule cleanup after download
    def cleanup():
        try:
            zip_path.unlink()
        except:
            pass
    
    background_tasks.add_task(cleanup)
    
    return FileResponse(
        zip_path,
        filename=f"{job_id}_outputs.zip",
        media_type='application/zip'
    )


@app.get("/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download a job output file"""
    
    job_dir = get_job_directory(job_id)
    file_path = job_dir / "outputs" / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        filename=filename,
        media_type='application/octet-stream'
    )


@app.get("/download/{job_id}/inputs/{filename}")
async def download_input_file(job_id: str, filename: str):
    """Download a job input file"""
    
    job_dir = get_job_directory(job_id)
    file_path = job_dir / "inputs" / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Input file not found")
    
    return FileResponse(
        file_path,
        filename=filename,
        media_type='application/octet-stream'
    )


@app.get("/inputs/{job_id}")
async def list_input_files(job_id: str):
    """List all input files for a job"""
    
    job_dir = get_job_directory(job_id)
    inputs_dir = job_dir / "inputs"
    
    if not inputs_dir.exists():
        return {"files": []}
    
    files = []
    for file_path in inputs_dir.iterdir():
        if file_path.is_file():
            files.append({
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "extension": file_path.suffix.lower()
            })
    
    return {"files": files}


@app.get("/jobs")
async def list_all_jobs():
    """List all jobs"""
    try:
        # Get all job keys from Redis
        job_keys = redis_conn.keys("job:*")
        jobs = []
        
        for key in job_keys:
            job_data = redis_conn.get(key)
            if job_data:
                job_metadata = json.loads(job_data)
                # Get current status
                status_response = await get_job_status(job_metadata['job_id'])
                job_metadata.update({
                    'state': status_response.state,
                    'progress': status_response.progress,
                    'elapsed_seconds': status_response.elapsed_seconds,
                    'message': status_response.message
                })
                jobs.append(job_metadata)
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return {"jobs": jobs}
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail="Failed to list jobs")


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job: cancel if running, remove metadata and files"""
    try:
        # Load job metadata
        job_data = redis_conn.get(f"job:{job_id}")
        rq_job_id = None
        if job_data:
            metadata = json.loads(job_data)
            rq_job_id = metadata.get('rq_job_id')
        
        # Try to cancel the RQ job if it exists
        try:
            if rq_job_id:
                from rq.job import Job
                rq_job = Job.fetch(rq_job_id, connection=redis_conn)
                if rq_job and rq_job.get_status() in ["queued", "started", "deferred"]:
                    rq_job.cancel()
        except Exception:
            pass
        
        # Delete Redis key
        try:
            redis_conn.delete(f"job:{job_id}")
        except Exception:
            pass
        
        # Remove job directory
        try:
            job_dir = get_job_directory(job_id)
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
        except Exception:
            pass
        
        return {"message": "Job deleted", "job_id": job_id}
    except Exception as e:
        logger.error(f"Failed to delete job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete job")


@app.get("/logs/{job_id}")
async def get_job_logs(job_id: str):
    """Get job logs"""
    
    job_dir = get_job_directory(job_id)
    log_file = job_dir / "logs" / "job.log"
    
    if not log_file.exists():
        return {"logs": "No logs available"}
    
    try:
        async with aiofiles.open(log_file, 'r') as f:
            logs = await f.read()
        return {"logs": logs}
    except Exception as e:
        logger.error(f"Failed to read logs for job {job_id}: {e}")
        return {"logs": "Failed to read logs"}


@app.delete("/job/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a job"""
    
    try:
        # Get job metadata
        job_data = redis_conn.get(f"job:{job_id}")
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_metadata = json.loads(job_data)
        
        # Cancel RQ job
        from rq.job import Job
        try:
            rq_job = Job.fetch(job_metadata['rq_job_id'], connection=redis_conn)
            rq_job.cancel()
        except:
            pass  # Job might already be finished
        
        log_job_event(logger, job_id, "job_canceled")
        
        return {"message": "Job canceled"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job cancellation failed: {e}")
        raise HTTPException(status_code=500, detail="Job cancellation failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
