import os
import sys
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
    
    job_id = generate_job_id()
    
    try:
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
                temp_file_path.rename(job_file_path)
                
                input_files.append({
                    'filename': temp_file_path.name,
                    'path': str(job_file_path),
                    **token_data['validation_result']
                })
            
            # Clean up upload token
            del upload_tokens[upload_token]
        
        # Submit job to queue
        job = task_queue.enqueue(
            'worker.sampling_task',
            job_id=job_id,
            params=job_data.params.model_dump(),
            input_files=input_files,
            job_timeout='180s'
        )
        
        # Store job metadata
        job_metadata = {
            'job_id': job_id,
            'rq_job_id': job.id,
            'params': job_data.params.model_dump(),
            'input_files': input_files,
            'created_at': datetime.utcnow().isoformat(),
            'status': JobStatus.QUEUED
        }
        
        redis_conn.setex(
            f"job:{job_id}",
            timedelta(hours=JOB_TTL_HOURS),
            json.dumps(job_metadata)
        )
        
        log_job_event(logger, job_id, "job_submitted", params=job_data.params.model_dump())
        
        return JobResponse(job_id=job_id)
        
    except Exception as e:
        logger.error(f"Job submission failed: {e}")
        raise HTTPException(status_code=500, detail="Job submission failed")


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
            
            status = status_mapping.get(rq_job.status, JobStatus.QUEUED)
            
            # Calculate progress and timing
            progress = 0
            if status == JobStatus.RUNNING:
                progress = 50  # Simple progress estimation
            elif status == JobStatus.SUCCEEDED:
                progress = 100
            
            started_at = rq_job.started_at
            completed_at = rq_job.ended_at
            elapsed_seconds = None
            
            if started_at:
                end_time = completed_at or datetime.utcnow()
                elapsed_seconds = (end_time - started_at).total_seconds()
            
            return JobStatusResponse(
                job_id=job_id,
                state=status,
                progress=progress,
                message=rq_job.exc_info if rq_job.is_failed else None,
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
