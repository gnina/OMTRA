import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional
import logging
import hashlib
import re

# Local helpers to avoid depending on Pydantic models in shared.models
def validate_filename(filename: str) -> str:
    """Sanitize and validate a filename for safe storage."""
    # Use only the basename to avoid directory traversal
    name = os.path.basename(filename)
    # Replace unsafe characters with underscores
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    # Prevent empty names
    if not name:
        raise ValueError("Invalid filename")
    # Limit length
    return name[:255]


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """Check if file has an allowed extension (case-insensitive)."""
    ext = Path(filename).suffix.lower()
    return ext in [e.lower() for e in allowed_extensions]


def calculate_file_hash(content: bytes) -> str:
    """Compute SHA-256 hash of file content."""
    return hashlib.sha256(content).hexdigest()


logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = ['.sdf', '.cif', '.mol2', '.pdb', '.xyz']
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 26214400))  # 25MB default
MAX_FILES_PER_JOB = int(os.getenv('MAX_FILES_PER_JOB', 3))


class FileValidationError(Exception):
    """Raised when file validation fails"""
    pass


def validate_file_safety(filename: str, content: bytes) -> dict:
    """
    Comprehensive file safety validation
    
    Returns:
        dict: Validation result with metadata
        
    Raises:
        FileValidationError: If validation fails
    """
    # Basic filename validation
    safe_filename = validate_filename(filename)
    
    # Extension validation
    if not validate_file_extension(safe_filename, ALLOWED_EXTENSIONS):
        raise FileValidationError(f"File extension not allowed. Allowed: {ALLOWED_EXTENSIONS}")
    
    # Size validation
    if len(content) > MAX_FILE_SIZE:
        raise FileValidationError(f"File too large. Max size: {MAX_FILE_SIZE} bytes")
    
    if len(content) == 0:
        raise FileValidationError("Empty file not allowed")
    
    # Calculate hash
    file_hash = calculate_file_hash(content)
    
    # Basic format validation based on extension
    try:
        validate_molecular_format(safe_filename, content)
    except Exception as e:
        raise FileValidationError(f"Invalid molecular file format: {str(e)}")
    
    return {
        'original_filename': filename,
        'safe_filename': safe_filename,
        'size': len(content),
        'sha256': file_hash,
        'mime_type': get_mime_type_from_content(content, safe_filename)
    }


def parse_xyz_atom_lines(content: str) -> tuple[list[str], bool]:
    """
    Parse XYZ file content and return atom lines and whether there's a comment.
    
    Returns:
        tuple: (atom_lines, has_comment)
    """
    lines = content.strip().split('\n')
    if len(lines) < 2:
        raise ValueError("XYZ file too short - need at least 2 lines")
    
    # First line should be number of atoms
    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        raise ValueError("First line of XYZ file must be number of atoms")
    
    # Determine if second line is a comment or an atom record
    has_comment = False
    if len(lines) > 2:
        second_line_fields = lines[1].strip().split()
        # Check if second line looks like an atom record (4+ fields, first field is element)
        if len(second_line_fields) >= 4:
            try:
                float(second_line_fields[0])
                has_comment = True
            except ValueError:
                has_comment = False
        else:
            has_comment = True
    
    if has_comment:
        atom_lines = lines[2:]  # Skip count + comment
    else:
        atom_lines = lines[1:]  # Skip count
    
    if len(atom_lines) < n_atoms:
        raise ValueError(f"XYZ file has {len(atom_lines)} atom lines but claims {n_atoms} atoms")
    
    return atom_lines, has_comment


def validate_molecular_format(filename: str, content: bytes) -> None:
    """
    Validate molecular file format using RDKit (if available) or basic checks
    """
    try:
        from rdkit import Chem
        from io import StringIO
        
        text_content = content.decode('utf-8', errors='ignore')
        ext = Path(filename).suffix.lower()
        
        if ext == '.sdf':
            # Try to parse as SDF
            supplier = Chem.SDMolSupplier()
            supplier.SetData(text_content)
            if not any(mol for mol in supplier if mol is not None):
                raise ValueError("No valid molecules found in SDF")
                
        elif ext == '.mol2':
            # Basic MOL2 validation - check for required sections
            required_sections = ['@<TRIPOS>MOLECULE', '@<TRIPOS>ATOM']
            for section in required_sections:
                if section not in text_content:
                    raise ValueError(f"Missing required MOL2 section: {section}")
                    
        elif ext == '.pdb':
            # Basic PDB validation
            lines = text_content.split('\n')
            has_atom_records = any(line.startswith(('ATOM', 'HETATM')) for line in lines)
            if not has_atom_records:
                raise ValueError("No ATOM/HETATM records found in PDB")
                
        elif ext == '.cif':
            # Basic CIF validation
            if not text_content.strip().startswith('data_'):
                raise ValueError("Invalid CIF format - missing data block")
                
        elif ext == '.xyz':
            # Use shared XYZ parsing logic
            atom_lines, _ = parse_xyz_atom_lines(text_content)
            
            # Validate atom lines
            for i, line in enumerate(atom_lines):
                fields = line.strip().split()
                if len(fields) < 4:
                    raise ValueError(f"Atom line {i+1} has insufficient fields: {line}")
    
    except ImportError:
        # RDKit not available, do basic text validation
        logger.warning("RDKit not available, using basic format validation")
        text_content = content.decode('utf-8', errors='ignore')
        
        # Very basic checks
        if len(text_content.strip()) < 10:
            raise ValueError("File content too short to be a valid molecular structure")
        
        # Check for obviously invalid content
        if any(char in text_content for char in ['\x00', '\xff']):
            raise ValueError("File contains binary data")


def get_mime_type_from_content(content: bytes, filename: str) -> str:
    """Determine MIME type from content and filename"""
    ext = Path(filename).suffix.lower()
    
    mime_types = {
        '.sdf': 'chemical/x-mdl-sdfile',
        '.mol2': 'chemical/x-mol2',
        '.pdb': 'chemical/x-pdb',
        '.cif': 'chemical/x-cif'
    }
    
    return mime_types.get(ext, 'application/octet-stream')


def create_job_directory(job_id: str) -> Path:
    """Create a secure job directory"""
    job_dir = Path(f"/srv/app/jobs/{job_id}")
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (job_dir / "inputs").mkdir(exist_ok=True)
    (job_dir / "outputs").mkdir(exist_ok=True)
    (job_dir / "logs").mkdir(exist_ok=True)
    
    return job_dir


def save_uploaded_file(job_id: str, filename: str, content: bytes) -> Path:
    """Safely save an uploaded file to job directory"""
    job_dir = create_job_directory(job_id)
    safe_filename = validate_filename(filename)
    
    file_path = job_dir / "inputs" / safe_filename
    
    # Write file securely
    with open(file_path, 'wb') as f:
        f.write(content)
    
    # Set restrictive permissions
    os.chmod(file_path, 0o644)
    
    return file_path


def cleanup_job_directory(job_id: str) -> None:
    """Clean up job directory"""
    job_dir = Path(f"/srv/app/jobs/{job_id}")
    if job_dir.exists():
        try:
            shutil.rmtree(job_dir)
            logger.info(f"Cleaned up job directory: {job_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup job directory {job_id}: {e}")


def get_job_directory(job_id: str) -> Path:
    """Get job directory path"""
    return Path(f"/srv/app/jobs/{job_id}")


def list_job_outputs(job_id: str) -> List[Path]:
    """List all output files for a job"""
    job_dir = get_job_directory(job_id)
    outputs_dir = job_dir / "outputs"
    
    if not outputs_dir.exists():
        return []
    
    return list(outputs_dir.glob("*"))


def create_zip_archive(job_id: str) -> Optional[Path]:
    """Create ZIP archive of job outputs"""
    import zipfile
    
    job_dir = get_job_directory(job_id)
    outputs_dir = job_dir / "outputs"
    
    if not outputs_dir.exists():
        logger.error(f"Outputs directory does not exist: {outputs_dir}")
        return None
    
    zip_path = job_dir / f"{job_id}_outputs.zip"
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in outputs_dir.glob("*"):
                if file_path.is_file():
                    zipf.write(file_path, file_path.name)
        
        logger.info(f"Successfully created ZIP archive: {zip_path}")
        return zip_path
    except Exception as e:
        logger.error(f"Failed to create ZIP archive for job {job_id}: {e}")
        return None
