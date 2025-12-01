import os
import sys
import shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import redis
from rq import Queue
import uuid
from datetime import datetime, timedelta
import json
import aiofiles
import time
import tempfile
import io

# Add shared module to path
sys.path.append(str(Path(__file__).parent.parent))

from shared.models import (
    JobSubmission, JobResponse, JobStatusResponse, JobResultResponse,
    UploadInitResponse, ArtifactInfo, JobStatus, generate_job_id, 
    generate_upload_token, SamplingParams
)
from shared.file_utils import (
    validate_file_safety, FileValidationError, create_job_directory,
    save_uploaded_file, list_job_outputs, create_zip_archive, get_job_directory,
    extract_pharmacophore_from_sdf, pharmacophore_list_to_xyz
)
from shared.logging_utils import logger, log_api_request, log_file_upload, log_job_event

try:
    from rdkit import Chem
    from rdkit.Chem import SDMolSupplier
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
from omtra.utils.checkpoints import (
    WEBAPP_TO_CHECKPOINT,
    get_checkpoint_path_for_webapp,
)

# Configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 8000))
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 26214400))
MAX_FILES_PER_JOB = int(os.getenv('MAX_FILES_PER_JOB', 3))
JOB_TTL_HOURS = int(os.getenv('JOB_TTL_HOURS', 48))

# Checkpoint configuration
CHECKPOINT_DIR = Path(os.getenv('CHECKPOINT_DIR', '/srv/app/checkpoints'))
CHECKPOINT_MAPPING = WEBAPP_TO_CHECKPOINT

# Check that required checkpoints exist locally
REQUIRED_CHECKPOINT_FILES = sorted(set(CHECKPOINT_MAPPING.values()))
missing_checkpoints = [
    filename for filename in REQUIRED_CHECKPOINT_FILES
    if not (CHECKPOINT_DIR / filename).exists()
]
if missing_checkpoints:
    logger.warning(
        f"Missing checkpoints: {', '.join(missing_checkpoints)}. "
        "API will still start, but sampling requests may fail until checkpoints are available. "
    )

# Initialize Redis and RQ
redis_conn = redis.from_url(REDIS_URL)
task_queue = Queue('omtra_tasks', connection=redis_conn, default_timeout='600s')

# FastAPI app
app = FastAPI(
    title="OMTRA Sampler API",
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
    return get_checkpoint_path_for_webapp(sampling_mode, CHECKPOINT_DIR)




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


@app.post("/extract-pharmacophore")
async def extract_pharmacophore_endpoint(file: UploadFile = File(...)):
    """Extract pharmacophore features from an uploaded SDF file"""
    
    try:
        # Validate filename exists and is SDF
        if not file.filename:
            logger.warning("Extract pharmacophore: No filename provided")
            raise HTTPException(status_code=400, detail="No filename provided")
        
        filename_lower = file.filename.lower()
        if not filename_lower.endswith('.sdf'):
            logger.warning(f"Extract pharmacophore: Invalid file type - {file.filename}")
            raise HTTPException(
                status_code=400, 
                detail=f"File must be an SDF file (got: {file.filename})"
            )
        
        # Read file content
        content = await file.read()
        
        if len(content) == 0:
            logger.warning("Extract pharmacophore: Empty file provided")
            raise HTTPException(status_code=400, detail="Empty file provided")
        
        # Extract pharmacophore features with timeout protection
        import asyncio
        
        # Run extraction in executor to prevent blocking and allow timeout
        loop = asyncio.get_event_loop()
        try:
            # Run extraction with timeout protection (120 seconds max)
            # Use run_in_executor for Python 3.10 compatibility
            result = await asyncio.wait_for(
                loop.run_in_executor(None, extract_pharmacophore_from_sdf, content),
                timeout=120.0
            )
            logger.info(f"Extracted {result.get('n_pharmacophores', 0)} pharmacophore features from {file.filename}")
            return result
        except asyncio.TimeoutError:
            logger.error("Extract pharmacophore: Extraction timed out after 120 seconds")
            raise HTTPException(
                status_code=504,
                detail="Pharmacophore extraction timed out. The molecule might be too large or complex."
            )
        except ValueError as e:
            # Re-raise ValueError as HTTP 400
            logger.error(f"Extract pharmacophore: ValueError: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Extract pharmacophore: Error during extraction: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Pharmacophore extraction failed: {str(e)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in pharmacophore extraction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pharmacophore extraction failed: {str(e)}")


@app.post("/pharmacophore-to-xyz")
async def pharmacophore_to_xyz_endpoint(data: dict = Body(...)):
    """Convert selected pharmacophore features to XYZ format"""
    
    try:
        pharmacophores = data.get('pharmacophores', [])
        selected_indices_list = data.get('selected_indices', [])
        center = data.get('center', True)
        
        if not pharmacophores:
            raise HTTPException(status_code=400, detail="No pharmacophores provided")
        
        # Convert list to set for filtering
        selected_indices = set(selected_indices_list) if selected_indices_list else None
        
        total_pharms = len(pharmacophores)
        n_selected = len(selected_indices) if selected_indices else total_pharms
        
        if selected_indices and n_selected == 0:
            raise HTTPException(status_code=400, detail="No pharmacophore features selected")
        
        xyz_content = pharmacophore_list_to_xyz(pharmacophores, selected_indices, center=center)
        
        # Verify the XYZ content
        xyz_lines = xyz_content.strip().split('\n')
        xyz_count = int(xyz_lines[0]) if len(xyz_lines) > 0 and xyz_lines[0].strip().isdigit() else 0
        
        if selected_indices and xyz_count != n_selected:
            logger.warning(f"XYZ file contains {xyz_count} features but {n_selected} were selected")
        
        return {
            'xyz_content': xyz_content,
            'n_selected': n_selected,
            'n_total': total_pharms
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"XYZ conversion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"XYZ conversion failed: {str(e)}")


@app.post("/sample", response_model=JobResponse)
async def submit_job(job_data: JobSubmission):
    """Submit a sampling job"""
    
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
    
    try:
        # Validate sampling mode and checkpoint availability
        sampling_mode = job_data.params.sampling_mode
        checkpoint_path = get_checkpoint_path(sampling_mode)
        if checkpoint_path is None:
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
        if sampling_mode in ["Pharmacophore-conditioned", "Protein-conditioned", "Protein+Pharmacophore-conditioned"] and len(input_files) == 0:
            raise HTTPException(status_code=400, detail=f"{sampling_mode} requires at least one input file (e.g., pharmacophore .xyz or protein files)")
        
        # Validate Protein-conditioned mode requires both protein and ligand files
        if sampling_mode == "Protein-conditioned":
            protein_files = [f for f in input_files if f['filename'].lower().endswith(('.pdb', '.cif'))]
            ligand_files = [f for f in input_files if f['filename'].lower().endswith('.sdf')]
            
            if not protein_files:
                raise HTTPException(
                    status_code=400, 
                    detail="Protein-conditioned mode requires a protein file (PDB/CIF format)"
                )
            if not ligand_files:
                raise HTTPException(
                    status_code=400, 
                    detail="Protein-conditioned mode requires a reference ligand file (SDF format) to identify the binding pocket"
                )
        
        # Get checkpoint path and add to params
        params_dict = pydantic_to_dict(job_data.params)
        params_dict['checkpoint_path'] = str(checkpoint_path)
        
        # Override n_samples with num_samples from the request if provided
        if hasattr(job_data, 'num_samples') and job_data.num_samples is not None:
            params_dict['n_samples'] = job_data.num_samples
        
        # Use GPU if available, otherwise CPU
        cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible_devices and cuda_visible_devices != '':
            params_dict['device'] = 'cuda'
        else:
            params_dict['device'] = 'cpu'

        # Submit job to queue
        # result_ttl controls how long the RQ Job object stays in Redis (default 500s)
        # Set to match JOB_TTL_HOURS so RQ jobs remain accessible for same duration as metadata
        result_ttl_seconds = JOB_TTL_HOURS * 3600
        # Increased timeout to 600s (10 minutes) for protein-conditioned sampling which can take longer
        # Use module path importable inside the worker container
        job = task_queue.enqueue(
            'worker.sampling_task',
            kwargs={
                'job_id': job_id,
                'params': params_dict,
                'input_files': input_files,
            },
            job_timeout='600s',
            result_ttl=result_ttl_seconds  # Keep RQ job object for 48 hours (same as metadata)
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
            # Progress is not accurately trackable during sampling, so we only show completion
            progress = 100 if status == JobStatus.SUCCEEDED else 0
            
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
            # RQ job expired or missing - use stored metadata as fallback
            stored_status = job_metadata.get('status')
            if stored_status:
                # Parse timestamps from metadata
                started_at = job_metadata.get('started_at')
                completed_at = job_metadata.get('completed_at')
                elapsed_seconds = None
                
                # Calculate elapsed time if both timestamps exist
                if started_at and completed_at:
                    try:
                        if isinstance(started_at, str):
                            started_dt = datetime.fromisoformat(started_at.replace('Z', '+00:00') if started_at.endswith('Z') else started_at)
                        else:
                            started_dt = started_at
                        
                        if isinstance(completed_at, str):
                            completed_dt = datetime.fromisoformat(completed_at.replace('Z', '+00:00') if completed_at.endswith('Z') else completed_at)
                        else:
                            completed_dt = completed_at
                        
                        elapsed_seconds = (completed_dt - started_dt).total_seconds()
                    except Exception:
                        pass
                
                # Parse datetime objects
                if isinstance(started_at, str):
                    started_at = datetime.fromisoformat(started_at.replace('Z', '+00:00') if started_at.endswith('Z') else started_at)
                elif not isinstance(started_at, datetime):
                    started_at = None
                
                if isinstance(completed_at, str):
                    completed_at = datetime.fromisoformat(completed_at.replace('Z', '+00:00') if completed_at.endswith('Z') else completed_at)
                elif not isinstance(completed_at, datetime):
                    completed_at = None
                
                # Map stored status
                status_mapping = {
                    'QUEUED': JobStatus.QUEUED,
                    'RUNNING': JobStatus.RUNNING,
                    'SUCCEEDED': JobStatus.SUCCEEDED,
                    'FAILED': JobStatus.FAILED,
                    'CANCELED': JobStatus.CANCELED
                }
                status = status_mapping.get(stored_status, JobStatus.FAILED)
                
                # Calculate progress - only show completion, not intermediate progress
                progress = 100 if status == JobStatus.SUCCEEDED else 0
                
                return JobStatusResponse(
                    job_id=job_id,
                    state=status,
                    progress=progress,
                    message=job_metadata.get('error') if status == JobStatus.FAILED else None,
                    started_at=started_at,
                    completed_at=completed_at,
                    elapsed_seconds=elapsed_seconds
                )
            
            # No stored status - job expired
            return JobStatusResponse(
                job_id=job_id,
                state=JobStatus.FAILED,
                message="Job status unknown - RQ job expired"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check failed: {e}", exc_info=True)
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
        logger.error(f"Result retrieval failed: {e}", exc_info=True)
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
        except Exception:
            pass  # Job might already be finished
        
        log_job_event(logger, job_id, "job_canceled")
        
        return {"message": "Job canceled"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job cancellation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Job cancellation failed")


# PoseView helper functions
def _embed_ligand_in_pdb(sdf_str: str, pdb_str: str) -> Optional[str]:
    """Embed ligand coordinates from SDF into PDB file as HETATM records"""
    try:
        sdf_lines = sdf_str.strip().split('\n')
        if len(sdf_lines) < 4:
            return None
        
        # Find counts line
        num_atoms, atom_start_idx = None, None
        if len(sdf_lines) > 3:
            try:
                parts = sdf_lines[3].split()
                if len(parts) > 0:
                    num_atoms = int(parts[0])
                    atom_start_idx = 4
            except (ValueError, IndexError):
                for i, line in enumerate(sdf_lines):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            if 0 < int(parts[0]) < 10000 and 0 <= int(parts[1]) < 10000:
                                num_atoms = int(parts[0])
                                atom_start_idx = i + 1
                                break
                        except ValueError:
                            continue
        
        if num_atoms is None or atom_start_idx is None:
            return None
        
        # Parse coordinates
        coords = []
        for i in range(atom_start_idx, atom_start_idx + num_atoms):
            if i < len(sdf_lines):
                parts = sdf_lines[i].split()
                if len(parts) >= 4:
                    try:
                        coords.append((float(parts[0]), float(parts[1]), float(parts[2]), parts[3]))
                    except (ValueError, IndexError):
                        continue
        
        if not coords:
            return None
        
        # Insert into PDB
        pdb_lines = pdb_str.split('\n')
        end_idx = next((i for i, line in enumerate(pdb_lines) if line.strip() == 'END'), None)
        max_atom_num = max((int(line[6:11].strip()) for line in pdb_lines if line.startswith(('ATOM', 'HETATM'))), default=0)
        
        hetatm_lines = [f"HETATM{max_atom_num + i + 1:5d}  {elem:2s}  LIG A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem:2s}  \n"
                       for i, (x, y, z, elem) in enumerate(coords)]
        
        if end_idx is not None:
            return '\n'.join(pdb_lines[:end_idx] + hetatm_lines + ['END\n'])
        return '\n'.join(pdb_lines + hetatm_lines + ['END\n'])
    except Exception:
        return None


def _poll_job(job_id: str, poll_url: str, poll_interval: int = 1, max_polls: int = 60):
    """Poll job status until complete"""
    import requests
    import time
    import logging
    logger = logging.getLogger(__name__)
    job = requests.get(poll_url + job_id + '/').json()
    status = job.get('status', '')
    current_poll = 0
    
    while status in ('pending', 'running'):
        if current_poll >= max_polls:
            return job
        time.sleep(poll_interval)
        job = requests.get(poll_url + job_id + '/').json()
        status = job.get('status', '')
        current_poll += 1
    return job


def _poll_poseview_job_v2(job_id: str, jobs_base_url: str, poll_interval: int = 1, max_attempts: int = 60) -> Optional[str]:
    """Poll ProteinsPlus API v2 for job completion and return SVG"""
    import requests
    import logging
    logger = logging.getLogger(__name__)
    job = _poll_job(job_id, jobs_base_url, poll_interval, max_attempts)
    if job.get('status', '').lower() == 'success':
        image_url = job.get('image')
        if image_url:
            try:
                svg_response = requests.get(image_url, timeout=30)
                svg_response.raise_for_status()
                return svg_response.text
            except Exception as e:
                logger.error(f"Failed to fetch SVG from {image_url}: {e}")
    return None


async def _cache_error(job_id: str, filename: str, status_code: int, detail: str):
    """Helper to cache error responses"""
    try:
        outputs_dir = get_job_directory(job_id) / "outputs"
        error_path = outputs_dir / f"{filename}_diagram_error.json"
        async with aiofiles.open(error_path, 'w') as f:
            await f.write(json.dumps({"statusCode": status_code, "message": detail.split('.')[0] if detail else "Failed", "detail": detail}))
    except Exception:
        pass


@app.get("/interaction-diagram/{job_id}/{filename}")
async def get_interaction_diagram(job_id: str, filename: str):
    """Generate a 2D interaction diagram using ProteinsPlus API v2"""
    
    try:
        import re
        import asyncio
        
        # ProteinsPlus API v2 base URL
        PROTEINS_PLUS_BASE = "https://proteins.plus"
        API_BASE = f"{PROTEINS_PLUS_BASE}/api/v2"
        
        # Get job directory
        job_dir = get_job_directory(job_id)
        logger.info(f"Generating interaction diagram for job_id={job_id}, filename={filename}, job_dir={job_dir}")
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get outputs directory first (to check for converted PDB)
        outputs_dir = job_dir / "outputs"
        if not outputs_dir.exists():
            raise HTTPException(status_code=404, detail="Output files not found")
        
        # Get inputs directory
        inputs_dir = job_dir / "inputs"
        if not inputs_dir.exists():
            raise HTTPException(status_code=404, detail="Input files not found")
        
        # List available files for debugging
        available_outputs = list(outputs_dir.glob("*"))
        logger.info(f"Available output files: {[f.name for f in available_outputs]}")

        pdb_files = list(inputs_dir.glob("*.pdb"))
        cif_files = list(inputs_dir.glob("*.cif"))
        protein_file = pdb_files[0] if pdb_files else (cif_files[0] if cif_files else None)
        
        if not protein_file:
            converted_pdb = outputs_dir / "protein_from_cif.pdb"
            if converted_pdb.exists():
                protein_file = converted_pdb
                logger.info(f"Using converted PDB file from outputs (no original found): {protein_file.name}")
            else:
                protein_file = None
        else:
            logger.info(f"Using original protein file from inputs: {protein_file.name} (for coordinate system consistency)")
        
        if not protein_file:
            raise HTTPException(status_code=400, detail="No protein file (PDB/CIF) found in job inputs or outputs")
        
        ligand_file = outputs_dir / filename
        logger.info(f"Looking for ligand file: {ligand_file} (exists: {ligand_file.exists()})")
        if not ligand_file.exists():
            raise HTTPException(status_code=404, detail=f"Ligand file not found: {filename}")
        
        # Check if cached diagram exists (store as {filename}_diagram.svg)
        diagram_filename = f"{filename}_diagram.svg"
        cached_diagram_path = outputs_dir / diagram_filename
        
        # Check if cached error exists (store as {filename}_diagram_error.json)
        error_filename = f"{filename}_diagram_error.json"
        cached_error_path = outputs_dir / error_filename
        
        # If cached error exists, return it immediately
        if cached_error_path.exists():
            try:
                async with aiofiles.open(cached_error_path, 'r') as f:
                    error_data = json.loads(await f.read())
                logger.info(f"Serving cached error for {filename}: {error_data.get('message', 'Unknown error')}")
                raise HTTPException(
                    status_code=error_data.get('statusCode', 500),
                    detail=error_data.get('detail', error_data.get('message', 'Failed to generate interaction diagram'))
                )
            except json.JSONDecodeError:
                logger.warning(f"Cached error file {cached_error_path} is invalid, deleting")
                try:
                    cached_error_path.unlink()
                except Exception:
                    pass
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"Failed to read cached error: {e}")
        
        # If cached diagram exists, serve it
        if cached_diagram_path.exists():
            async with aiofiles.open(cached_diagram_path, 'r') as f:
                svg_content = await f.read()
            logger.info(f"Serving cached diagram for job_id={job_id}, filename={filename}")
            from fastapi.responses import Response
            return Response(
                content=svg_content,
                media_type="image/svg+xml",
                headers={"Cache-Control": "public, max-age=3600"}
            )
        
        logger.info(f"No cached diagram found for {filename}, generating new one...")
        
        # Read protein file
        async with aiofiles.open(protein_file, 'r') as f:
            protein_str = await f.read()
        
        # Read ligand file
        async with aiofiles.open(ligand_file, 'r') as f:
            ligand_str = await f.read()
        
        logger.info(f"Read protein file: {len(protein_str)} chars, ligand file: {len(ligand_str)} chars")
        
        # Determine protein format
        protein_format = 'pdb' if protein_file.suffix.lower() == '.pdb' else 'cif'
        
        # Convert CIF to PDB if needed (ProteinsPlus API requires PDB format)
        if protein_format == 'cif':
            logger.info("Converting CIF to PDB for PoseView generation...")
            try:
                from biotite.structure.io import pdb
                from biotite.structure.io.pdbx import CIFFile, get_structure
                import biotite.structure as struc
                
                # Read CIF file using biotite
                cif_file = CIFFile.read(str(protein_file))
                st = get_structure(cif_file, model=1, include_bonds=False)
                
                # Remove waters and hydrogens
                st = st[st.res_name != "HOH"]
                st = st[st.element != "H"]
                st = st[st.element != "D"]
                
                total_atoms = len(st)
                logger.info(f"Biotite read CIF: {total_atoms} atoms")
                
                # Convert to PDB string
                pdb_file = pdb.PDBFile()
                pdb_file.set_structure(st)
                pdb_stringio = io.StringIO()
                pdb_file.write(pdb_stringio)
                protein_str = pdb_stringio.getvalue()
                protein_format = 'pdb'  # Now it's PDB format
                
            except ImportError:
                logger.error("biotite not available, cannot convert CIF to PDB")
                raise HTTPException(
                    status_code=503,
                    detail="CIF to PDB conversion not available. Please use a PDB file for interaction diagrams."
                )
            except Exception as conv_e:
                logger.error(f"CIF→PDB conversion failed: {conv_e}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to convert CIF to PDB: {str(conv_e)}"
                )
        
        # Ensure PDB has END record
        if protein_format == 'pdb':
            protein_lines = protein_str.strip().split('\n')
            has_end = any(line.strip() == 'END' or line.strip().startswith('END') for line in protein_lines[-5:])
            if not has_end:
                protein_str = protein_str.rstrip() + '\nEND\n'
                logger.info("Added END record to PDB")
        
        # Function to generate diagram using ProteinsPlus API v2 with embed→extract approach
        def generate_diagram(ligand_filename_param: str, protein_str_param: str, ligand_str_param: str) -> Optional[str]:
            """Generate diagram using ProteinsPlus API v2.
            
            Strategy: Use embed→extract approach which ensures proper coordinate alignment.
            This is the only reliable method that works consistently.
            """
            try:
                import requests
                # Log what we're about to process
                logger.info(f"generate_diagram called for ligand_filename={ligand_filename_param}, protein_len={len(protein_str_param)}, ligand_len={len(ligand_str_param)}")
                logger.info(f"Ligand file first 100 chars: {ligand_str_param[:100]}")
                PROTEINS_PLUS_BASE = "https://proteins.plus"
                API_BASE = f"{PROTEINS_PLUS_BASE}/api/v2"
                UPLOAD_URL = f"{API_BASE}/molecule_handler/upload/"
                UPLOAD_JOBS_URL = f"{API_BASE}/molecule_handler/upload/jobs/"
                PROTEINS_URL = f"{API_BASE}/molecule_handler/proteins/"
                LIGANDS_URL = f"{API_BASE}/molecule_handler/ligands/"
                POSEVIEW_URL = f"{API_BASE}/poseview/"
                POSEVIEW_JOBS_URL = f"{API_BASE}/poseview/jobs/"
                
                # Use embed→extract approach (ensures coordinate alignment)
                logger.info("Using embed→extract approach (embed → extract → generate)")
                
                # Step 1: Embed ligand in PDB
                logger.info("Step 1: Embedding ligand in PDB...")
                combined_pdb_str = _embed_ligand_in_pdb(ligand_str_param, protein_str_param)
                if not combined_pdb_str:
                    logger.error("✗ Failed to embed ligand")
                    return None
                
                logger.info(f"✓ Successfully embedded ligand, combined PDB: {len(combined_pdb_str)} chars")
                
                # Step 2: Extract ligand from combined PDB
                logger.info("Step 2: Uploading combined PDB to extract ligand...")
                combined_pdb_bytes = combined_pdb_str.encode('utf-8')
                combined_pdb_file_obj = io.BytesIO(combined_pdb_bytes)
                combined_pdb_file_obj.seek(0)
                files = {'protein_file': ('protein.pdb', combined_pdb_file_obj, 'chemical/x-pdb')}
                
                try:
                    preprocessing_job_submission = requests.post(UPLOAD_URL, files=files, timeout=120).json()
                    preprocessing_job = _poll_job(preprocessing_job_submission['job_id'], UPLOAD_JOBS_URL, poll_interval=1, max_polls=30)
                    
                    if preprocessing_job.get('status') == 'success':
                        protein_combined = requests.get(f"{PROTEINS_URL}{preprocessing_job['output_protein']}/", timeout=15).json()
                        
                        if protein_combined.get('ligand_set'):
                            ligand_id = protein_combined['ligand_set'][0]
                            ligand = requests.get(f"{LIGANDS_URL}{ligand_id}/", timeout=15).json()
                            logger.info(f"✓ Extracted ligand: {ligand.get('name', 'N/A')} (ID: {ligand_id})")
                            # Log ligand details for debugging
                            logger.info(f"Extracted ligand details: name={ligand.get('name')}, id={ligand_id}, original_filename={ligand_filename_param}")
                            # Log if multiple ligands were found
                            if len(protein_combined.get('ligand_set', [])) > 1:
                                logger.warning(f"Multiple ligands found in combined PDB ({len(protein_combined['ligand_set'])}), using first one: {ligand_id}")
                                logger.warning(f"All ligand IDs: {protein_combined['ligand_set']}")
                            
                            # Step 3: Upload original PDB (without embedded ligand)
                            logger.info("Step 3: Uploading original PDB...")
                            original_protein_bytes = protein_str_param.encode('utf-8')
                            original_protein_file_obj = io.BytesIO(original_protein_bytes)
                            original_protein_file_obj.seek(0)
                            files2 = {'protein_file': ('protein.pdb', original_protein_file_obj, 'chemical/x-pdb')}
                            preprocessing_job_submission2 = requests.post(UPLOAD_URL, files=files2, timeout=120).json()
                            preprocessing_job2 = _poll_job(preprocessing_job_submission2['job_id'], UPLOAD_JOBS_URL, poll_interval=1, max_polls=30)
                            
                            if preprocessing_job2.get('status') == 'success':
                                protein_original = requests.get(f"{PROTEINS_URL}{preprocessing_job2['output_protein']}/", timeout=15).json()
                                
                                # Step 4: Generate PoseView with extracted ligand
                                logger.info("Step 4: Generating PoseView diagram...")
                                query = {'protein_id': protein_original['id'], 'ligand_id': ligand_id}
                                poseview_job_submission = requests.post(POSEVIEW_URL, data=query, timeout=120).json()
                                poseview_job = _poll_job(poseview_job_submission['job_id'], POSEVIEW_JOBS_URL, poll_interval=1, max_polls=60)
                                
                                if poseview_job.get('status') == 'success':
                                    image_url = poseview_job.get('image')
                                    if image_url:
                                        logger.info(f"✓✓✓ SUCCESS! Image URL: {image_url}")
                                        logger.info(f"Using protein_id={protein_original['id']}, ligand_id={ligand_id} for PoseView")
                                        # Fetch the SVG from the URL
                                        try:
                                            svg_response = requests.get(image_url, timeout=30)
                                            svg_response.raise_for_status()
                                            svg_content = svg_response.text
                                            logger.info(f"Downloaded SVG, length: {len(svg_content)} bytes")
                                            return svg_content
                                        except Exception as e:
                                            logger.error(f"Failed to fetch SVG from {image_url}: {e}")
                                            return None
                                    else:
                                        logger.warning("PoseView job succeeded but no image URL")
                                        return None
                                else:
                                    error_msg = poseview_job.get('error', 'Unknown error')
                                    logger.warning(f"✗ PoseView failed for {ligand_filename_param}: {error_msg}")
                                    return None
                            else:
                                logger.warning(f"✗ Failed to upload original protein: {preprocessing_job2.get('status')}")
                                return None
                        else:
                            logger.warning("✗ No ligands found in combined PDB")
                            return None
                    else:
                        logger.warning(f"✗ Preprocessing failed: {preprocessing_job.get('error')}")
                        return None
                except Exception as e:
                    logger.warning(f"Error in embed→extract workflow: {e}")
                    return None
                
                logger.error("All PoseView generation methods failed")
                return None
                
            except Exception as e:
                logger.error(f"Error in generate_diagram: {e}", exc_info=True)
                return None
        
        # Run ProteinsPlus API call in executor (blocking operation)
        loop = asyncio.get_event_loop()
        svg = await asyncio.wait_for(
            loop.run_in_executor(None, generate_diagram, filename, protein_str, ligand_str),
            timeout=180.0  # 3 minutes
        )
        
        # Check if SVG is empty or contains only a border (no actual content)
        def is_empty_svg(svg_content: str) -> bool:
            """Check if SVG is empty or contains only border/background"""
            if not svg_content:
                return True
            # Remove whitespace and check for meaningful content
            svg_lower = svg_content.lower()
            # Empty SVG typically has only a border path like "M 0 0 L 600 0 L 600 600 L 0 600 Z"
            # Check if there are any text elements, circles, or paths with more than just the border
            has_text = '<text' in svg_lower or '<tspan' in svg_lower
            has_circles = svg_lower.count('<circle') > 0
            # Count paths - empty SVG usually has only 1-2 paths (border)
            path_count = svg_lower.count('<path')
            # Check if there are any interaction-related elements (lines, bonds, etc.)
            has_interactions = has_text or has_circles or path_count > 2
            # Also check if the SVG is suspiciously small (empty SVGs are usually < 500 bytes)
            is_too_small = len(svg_content) < 500
            return is_too_small and not has_interactions
        
        if not svg or is_empty_svg(svg):
            error_detail = f"Failed to generate interaction diagram for {filename}. PoseView could not detect any interactions. This can happen if the ligand is too far from the protein binding site or the coordinate systems don't align."
            logger.error(f"All interaction diagram generation methods failed for {filename} (MolModa proxy and PoseEdit API)")
            logger.error(f"  This usually means: 1) PoseView returned empty diagram (no interactions), 2) Ligand too far from protein, 3) Coordinate system mismatch")
            logger.error(f"  SVG length: {len(svg) if svg else 0} bytes")
            await _cache_error(job_id, filename, 503, error_detail)
            raise HTTPException(
                status_code=503,
                detail=error_detail
            )
        
        # Clean up SVG (remove width/height attributes for responsiveness)
        svg = re.sub(r'\s+width="\d+pt"\s+height="\d+pt"', '', svg)
        
        # Save SVG to cache (store in outputs directory like ligand files)
        diagram_filename = f"{filename}_diagram.svg"
        cached_diagram_path = outputs_dir / diagram_filename
        try:
            async with aiofiles.open(cached_diagram_path, 'w') as f:
                await f.write(svg)
            logger.info(f"Saved diagram to cache: {cached_diagram_path} ({len(svg)} bytes)")
        except Exception as e:
            logger.warning(f"Failed to save diagram to cache: {e} (continuing anyway)")
        
        # Return SVG as text
        from fastapi.responses import Response
        return Response(
            content=svg,
            media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=3600"}
        )
        
    except asyncio.TimeoutError:
        logger.error(f"PoseView generation timed out for job {job_id}, file {filename}")
        error_detail = "Interaction diagram generation timed out. Please try again later."
        await _cache_error(job_id, filename, 504, error_detail)
        raise HTTPException(
            status_code=504,
            detail=error_detail
        )
    except HTTPException as http_err:
        await _cache_error(job_id, filename, http_err.status_code, http_err.detail)
        raise
    except Exception as e:
        logger.error(f"Failed to generate interaction diagram for job {job_id}, file {filename}: {e}", exc_info=True)
        error_detail = f"Failed to generate interaction diagram: {str(e)}"
        await _cache_error(job_id, filename, 500, error_detail)
        raise HTTPException(
            status_code=500,
            detail=error_detail
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
