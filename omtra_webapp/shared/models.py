from pydantic import BaseModel, Field, validator, model_validator
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
    sampling_mode: str = Field(default="Unconditional", description="Sampling mode: Unconditional, Pharmacophore-conditioned, Protein-conditioned, or Protein+Pharmacophore-conditioned")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    n_samples: int = Field(default=10, ge=1, le=100, description="Number of samples to generate")
    steps: int = Field(default=100, ge=10, le=1000, description="Number of sampling steps")
    device: Optional[str] = Field(default="cuda", description="Device to run on")
    n_lig_atoms_mean: Optional[float] = Field(default=None, ge=4, description="Mean number of atoms for ligand samples (if provided, uses normal distribution instead of dataset distribution)")
    n_lig_atoms_std: Optional[float] = Field(default=None, ge=0.1, description="Standard deviation for number of atoms (required if n_lig_atoms_mean is provided)")
    
    @validator('sampling_mode')
    def validate_sampling_mode(cls, v):
        valid_modes = ["Unconditional", "Pharmacophore-conditioned", "Protein-conditioned", "Protein+Pharmacophore-conditioned"]
        if v not in valid_modes:
            raise ValueError(f"sampling_mode must be one of {valid_modes}")
        return v
    
    @model_validator(mode='after')
    def validate_atoms_params(self):
        n_lig_atoms_mean = self.n_lig_atoms_mean
        n_lig_atoms_std = self.n_lig_atoms_std
        if n_lig_atoms_mean is not None and n_lig_atoms_std is None:
            raise ValueError("n_lig_atoms_std is required when n_lig_atoms_mean is provided")
        if n_lig_atoms_mean is None and n_lig_atoms_std is not None:
            raise ValueError("n_lig_atoms_mean is required when n_lig_atoms_std is provided")
        return self


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
    num_samples: Optional[int] = Field(default=None, description="Number of samples to generate (overrides params.n_samples)")
    job_id: Optional[str] = Field(default=None, description="Custom job ID (if not provided, will be auto-generated)")
    
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


