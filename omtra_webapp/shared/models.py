from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import hashlib
from datetime import datetime
import uuid


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class SamplingParams(BaseModel):
    """Parameters for molecule sampling"""
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    n_samples: int = Field(default=10, ge=1, le=100, description="Number of samples to generate")
    steps: int = Field(default=100, ge=10, le=1000, description="Number of sampling steps")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="Sampling temperature")
    guidance_scale: float = Field(default=1.0, ge=0.0, le=10.0, description="Guidance scale for conditioning")
    conditioning_strength: float = Field(default=1.0, ge=0.0, le=2.0, description="Conditioning strength")
    device: Optional[str] = Field(default="cuda", description="Device to run on")


class UploadInfo(BaseModel):
    """Information about an uploaded file"""
    filename: str
    size: int
    sha256: str
    mime_type: str
    upload_token: str


class JobSubmission(BaseModel):
    """Job submission request"""
    params: SamplingParams
    uploads: List[str] = Field(default=[], description="List of upload tokens")
    
    @validator('uploads')
    def validate_uploads(cls, v):
        if len(v) > 3:  # MAX_FILES_PER_JOB
            raise ValueError("Maximum 3 files per job")
        return v


class JobResponse(BaseModel):
    """Response when submitting a job"""
    job_id: str
    message: str = "Job submitted successfully"


class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str
    state: JobStatus
    progress: int = Field(ge=0, le=100, default=0)
    message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    elapsed_seconds: Optional[float] = None


class ArtifactInfo(BaseModel):
    """Information about a job artifact"""
    id: str
    filename: str
    format: str  # sdf, cif, mol2, pdb
    size: int
    path_or_url: str
    sha256: Optional[str] = None


class JobResultResponse(BaseModel):
    """Job result response"""
    job_id: str
    state: JobStatus
    artifacts: List[ArtifactInfo] = []
    logs_url: Optional[str] = None
    params: SamplingParams
    elapsed_seconds: Optional[float] = None
    error_message: Optional[str] = None


class UploadInitResponse(BaseModel):
    """Response for upload initialization"""
    upload_token: str
    upload_url: Optional[str] = None  # For presigned URLs
    max_file_size: int


def generate_job_id() -> str:
    """Generate a unique job ID"""
    return str(uuid.uuid4())


def generate_upload_token() -> str:
    """Generate a unique upload token"""
    return str(uuid.uuid4())


def calculate_file_hash(content: bytes) -> str:
    """Calculate SHA256 hash of file content"""
    return hashlib.sha256(content).hexdigest()


def validate_filename(filename: str) -> str:
    """Sanitize and validate filename"""
    import os
    import re
    
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace dangerous characters
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    
    # Ensure it doesn't start with a dot
    if filename.startswith('.'):
        filename = 'file_' + filename
        
    # Truncate if too long
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:96] + ext
        
    return filename


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """Validate file extension"""
    import os
    ext = os.path.splitext(filename)[1].lower()
    return ext in [e.lower() for e in allowed_extensions]


def get_mime_type(filename: str) -> str:
    """Get MIME type for a filename"""
    import mimetypes
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"
