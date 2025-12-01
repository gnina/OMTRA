import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional
import logging
import hashlib
import re
import sys

def validate_filename(filename: str) -> str:
    """Sanitize and validate a filename for safe storage."""
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

# Jobs base directory
def _resolve_jobs_base_dir() -> Path:
    preferred = Path(os.getenv('JOBS_DIR', '/srv/app/jobs'))
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except Exception:
        # Fallback to repository-local jobs dir
        repo_fallback = Path(__file__).resolve().parents[2] / '.omtra' / 'jobs'
        try:
            repo_fallback.mkdir(parents=True, exist_ok=True)
            return repo_fallback
        except Exception:
            # Fallback to home directory
            home_fallback = Path.home() / 'omtra_jobs'
            home_fallback.mkdir(parents=True, exist_ok=True)
            return home_fallback

JOBS_BASE_DIR = _resolve_jobs_base_dir()


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
    # Try to use the configured base dir, but fallback if it fails
    try:
        job_dir = JOBS_BASE_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        # If the configured base dir fails, try fallbacks
        logger.warning(f"Failed to create job directory in {JOBS_BASE_DIR}: {e}. Trying fallback paths...")
        # Re-evaluate fallback paths
        repo_fallback = Path(__file__).resolve().parents[2] / '.omtra' / 'jobs'
        try:
            repo_fallback.mkdir(parents=True, exist_ok=True)
            job_dir = repo_fallback / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Final fallback to home directory
            home_fallback = Path.home() / 'omtra_jobs'
            home_fallback.mkdir(parents=True, exist_ok=True)
            job_dir = home_fallback / job_id
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


def get_job_directory(job_id: str) -> Path:
    """Get job directory path"""
    return JOBS_BASE_DIR / job_id


def list_job_outputs(job_id: str) -> List[Path]:
    """List all output files for a job"""
    job_dir = get_job_directory(job_id)
    outputs_dir = job_dir / "outputs"
    
    if not outputs_dir.exists():
        return []
    
    return list(outputs_dir.glob("*"))


def create_zip_archive(job_id: str) -> Optional[Path]:
    """Create ZIP archive of job outputs and input files"""
    import zipfile
    
    job_dir = get_job_directory(job_id)
    outputs_dir = job_dir / "outputs"
    inputs_dir = job_dir / "inputs"
    
    if not outputs_dir.exists():
        logger.error(f"Outputs directory does not exist: {outputs_dir}")
        return None
    
    zip_path = job_dir / f"{job_id}_outputs.zip"
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add output files
            for file_path in outputs_dir.glob("*"):
                if file_path.is_file():
                    # TODO
                    if file_path.suffix.lower() == '.svg':
                        continue

                    if "per_molecule_metrics" in file_path.name and file_path.name != "per_molecule_metrics.json":
                        continue
                    zipf.write(file_path, file_path.name)
            
            # Add input files to a subdirectory in the zip
            if inputs_dir.exists():
                for file_path in inputs_dir.glob("*"):
                    if file_path.is_file():
                        # Store input files in an "inputs/" subdirectory in the zip
                        zipf.write(file_path, f"inputs/{file_path.name}")
        
        logger.info(f"Successfully created ZIP archive: {zip_path}")
        return zip_path
    except Exception as e:
        logger.error(f"Failed to create ZIP archive for job {job_id}: {e}")
        return None


def extract_pharmacophore_from_sdf(sdf_content: bytes) -> dict:
    """
    Extract pharmacophore features from an SDF ligand file.
    
    Returns:
        dict: {
            'pharmacophores': [
                {
                    'type': str,  # 'Aromatic', 'HydrogenDonor', etc.
                    'element': str,  # 'P', 'S', 'F', etc.
                    'position': [x, y, z],
                    'index': int  # Index for selection
                },
                ...
            ]
        }
    """
    try:
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from .pharmacophore_extraction import (
            get_pharmacophores,
            ph_idx_to_elem,
            ph_idx_to_type,
        )
        
        # Parse SDF content
        text_content = sdf_content.decode('utf-8', errors='ignore')
        
        # Validate SDF content is not empty
        if not text_content.strip():
            raise ValueError("SDF file is empty or contains no data")
        
        # Create supplier and parse molecule
        supplier = Chem.SDMolSupplier()
        supplier.SetData(text_content)
        
        mol = None
        mol_count = 0
        for m in supplier:
            mol_count += 1
            if m is not None:
                mol = m
                break
        
        if mol is None:
            raise ValueError(f"No valid molecule found in SDF (tried {mol_count} molecules)")

        try:
            # Check if molecule has conformer with 3D coordinates
            if mol.GetNumConformers() == 0:
                # No conformer exists - generate one
                try:
                    # Embed molecule to generate 3D coordinates
                    result = AllChem.EmbedMolecule(mol, randomSeed=42, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
                    if result != 0:
                        # Embedding failed, try without constraints
                        result = AllChem.EmbedMolecule(mol, randomSeed=42)
                        if result != 0:
                            raise ValueError("Failed to generate 3D coordinates for molecule. "
                                           "The molecule may be too complex or invalid.")
                    
                    if mol.GetNumConformers() == 0:
                        raise ValueError("Molecule has no 3D coordinates after embedding")
                        
                except Exception as embed_err:
                    raise ValueError(f"Failed to generate 3D coordinates for molecule: {str(embed_err)}")
            
            # Verify conformer has valid 3D positions
            conformer = mol.GetConformer(0)
            atom_positions = conformer.GetPositions()
            if atom_positions is None or len(atom_positions) == 0:
                raise ValueError("Conformer has no 3D atom positions")
            
            # Verify positions are actually 3D (not all zeros)

            if np.allclose(atom_positions, 0.0, atol=1e-6):
                raise ValueError("Conformer positions are all zero - invalid 3D coordinates")
            
            # for debugging
            original_n_atoms = mol.GetNumAtoms()
            original_positions = atom_positions.copy()
                
        except Exception as e:
            raise ValueError(f"Error validating molecule conformer: {str(e)}")
        

        try:
            P, X, V, I = get_pharmacophores(mol=mol, rec=None)

            
            if len(P) == 0:
                raise ValueError("No pharmacophore features extracted from molecule")
                
        except Exception as e:
            raise ValueError(f"Error extracting pharmacophore features: {str(e)}")
        
        pharmacophores = []
        
        for i in range(len(P)):
            # Get type index and convert to type name and element symbol
            type_idx = int(X[i])
            ptype = ph_idx_to_type[type_idx]
            element = ph_idx_to_elem[type_idx]

            pos = P[i]
            
            pharmacophores.append({
                'type': ptype,
                'element': element,
                'position': [float(pos[0]), float(pos[1]), float(pos[2])],  # 3D coordinates from get_pharmacophores
                'index': i,
                'selected': True  # All selected by default (user can deselect)
            })
        
        return {
            'pharmacophores': pharmacophores,
            'n_pharmacophores': len(pharmacophores)
        }
        
    except ImportError as e:
        raise ValueError(f"RDKit or omtra modules not available: {e}")
    except Exception as e:
        raise ValueError(f"Failed to extract pharmacophore: {str(e)}")


def pharmacophore_list_to_xyz(pharmacophores: list, selected_indices: set = None, center: bool = True) -> str:
    """
    Convert a list of pharmacophore features to XYZ format.
    
    Args:
        pharmacophores: List of pharmacophore dicts with 'element' and 'position'
        selected_indices: Set of indices to include (if None, include all)
        center: If True, center coordinates at origin (required for model input)
    
    Returns:
        str: XYZ file content
    """
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    total_pharms = len(pharmacophores)
    
    if selected_indices is None:
        selected_pharms = pharmacophores
        logger.info(f"pharmacophore_list_to_xyz: No selection - using all {total_pharms} features")
    else:
        selected_pharms = [p for i, p in enumerate(pharmacophores) if i in selected_indices]
        n_selected = len(selected_pharms)
        logger.info(f"pharmacophore_list_to_xyz: Selected {n_selected} of {total_pharms} features. Indices: {sorted(selected_indices)}")
        
        if n_selected == 0:
            raise ValueError("No pharmacophore features selected")
        
        if n_selected != len(selected_indices):
            logger.warning(f"pharmacophore_list_to_xyz: Expected {len(selected_indices)} features but got {n_selected} after filtering")
    
    # Extract positions and ensure they're numpy arrays
    positions_list = [p['position'] for p in selected_pharms]
    # Ensure each position is a list/array with 3 elements
    positions_list = [[float(coord) for coord in pos] if isinstance(pos, (list, tuple)) else pos for pos in positions_list]
    positions = np.array(positions_list, dtype=np.float32)
    
    # Verify positions shape
    if positions.shape[0] == 0:
        raise ValueError("No pharmacophore positions found")
    if positions.shape[1] != 3:
        raise ValueError(f"Expected 3D coordinates, got shape {positions.shape}")
    
    # Center coordinates at origin 
    if center:
        com = positions.mean(axis=0)
        positions = positions - com
        
        actual_com = positions.mean(axis=0)
        if np.abs(actual_com).max() > 1e-4:
            import warnings
            warnings.warn(f"Centering may have failed: COM = {actual_com}, expected near [0, 0, 0]")
    
    lines = [str(len(selected_pharms))]
    lines.append("Pharmacophore features")
    
    for i, pharm in enumerate(selected_pharms):
        elem = pharm['element']
        x, y, z = positions[i]
        lines.append(f"{elem} {x:.6f} {y:.6f} {z:.6f}")
    
    return '\n'.join(lines)
