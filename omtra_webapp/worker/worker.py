import os
import sys
import collections
import collections.abc

# Monkeypatch collections for Python 3.10+ compatibility (Mapping, Iterable, etc. moved to collections.abc)
if not hasattr(collections, 'Mapping'):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable
if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable
if not hasattr(collections, 'Sequence'):
    collections.Sequence = collections.abc.Sequence
if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, 'MutableSequence'):
    collections.MutableSequence = collections.abc.MutableSequence
if not hasattr(collections, 'Set'):
    collections.Set = collections.abc.Set
import time
import json
import traceback
import shutil
import logging
import subprocess
import tempfile
import io
import re
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import redis
from rq import Worker, Queue
import torch
import pandas as pd
import numpy as np

# Add shared module to path
sys.path.append(str(Path(__file__).parent.parent))

# Add /app to path so cli.py can import routines
app_dir = Path('/app')
if app_dir.exists():
    sys.path.insert(0, str(app_dir))

# Add OMTRA modules to path for metrics computation
omtra_root = Path('/omtra')
if omtra_root.exists():
    sys.path.insert(0, str(omtra_root))

from shared.logging_utils import setup_logging, log_job_event
from shared.file_utils import get_job_directory, create_job_directory
from omtra.utils.checkpoints import WEBAPP_TO_CHECKPOINT

# Configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
CHECKPOINT_DIR = Path(os.getenv('CHECKPOINT_DIR', '/srv/app/checkpoints'))

# Define modes that involve a protein (conditioned or docking)
# These modes should have protein-ligand metrics (vina score, clashes, etc.) computed
# and should have interaction diagrams generated if possible.
PROTEIN_INVOLVING_MODES = [
    'Protein-conditioned', 
    'Protein+Pharmacophore-conditioned', 
    'Rigid Docking', 
    'Rigid Docking + Pharmacophore'
]

DEFAULT_METRICS_OPTIONS = {
    'posebusters': True,
    'posecheck': True,
    'strain': True,
    'vina': True,
    'poseview': True,
}


def _resolve_metrics_options(params: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    """Merge job params metrics_options with defaults (None => all enabled)."""
    merged = dict(DEFAULT_METRICS_OPTIONS)
    if not params:
        return merged
    raw = params.get('metrics_options')
    if raw is None:
        return merged
    if hasattr(raw, 'model_dump'):
        raw = raw.model_dump()
    elif hasattr(raw, 'dict'):
        raw = raw.dict()
    if isinstance(raw, dict):
        for key in DEFAULT_METRICS_OPTIONS:
            if key in raw:
                merged[key] = bool(raw[key])
    return merged

# Setup logging
logger = setup_logging(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    structured=os.getenv('STRUCTURED_LOGGING', 'true').lower() == 'true'
)

# Check that required checkpoints exist locally before processing jobs
# Get required checkpoint filenames from the mapping
REQUIRED_CHECKPOINT_FILES = sorted(set(WEBAPP_TO_CHECKPOINT.values()))
missing_checkpoints = [
    filename for filename in REQUIRED_CHECKPOINT_FILES
    if not (CHECKPOINT_DIR / filename).exists()
]
if missing_checkpoints:
    logger.warning(
        f"Missing checkpoints: {', '.join(missing_checkpoints)}. "
        "Jobs referencing missing checkpoints will fail until they are available. "
        "Please download checkpoints manually to the checkpoint directory."
    )


# Interaction diagram generation (ProteinsPlus PoseView)
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


POSEVIEW_MAX_POLLS = int(os.environ.get("POSEVIEW_MAX_POLLS", "12"))
POSEVIEW_POLL_INTERVAL = float(os.environ.get("POSEVIEW_POLL_INTERVAL", "1"))
PROTEINSPLUS_UPLOAD_MAX_POLLS = int(os.environ.get("PROTEINSPLUS_UPLOAD_MAX_POLLS", "30"))


def _poll_job(
    job_id: str,
    poll_url: str,
    poll_interval: float = POSEVIEW_POLL_INTERVAL,
    max_polls: int = POSEVIEW_MAX_POLLS,
):
    """Poll ProteinsPlus job status until complete or max_polls."""
    import requests

    job = requests.get(poll_url + job_id + "/", timeout=30).json()
    status = job.get("status", "")
    current_poll = 0

    while status in ("pending", "running"):
        if current_poll >= max_polls:
            return job
        time.sleep(poll_interval)
        job = requests.get(poll_url + job_id + "/", timeout=30).json()
        status = job.get("status", "")
        current_poll += 1
    return job


DIAGRAMS_STATUS_FILE = "diagrams_status.json"


def _write_diagrams_status(outputs_dir: Path, **updates: Any) -> None:
    path = outputs_dir / DIAGRAMS_STATUS_FILE
    current: Dict[str, Any] = {
        "pending": [],
        "in_progress": None,
        "completed": [],
        "failed": [],
    }
    if path.exists():
        try:
            with open(path) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                current.update(loaded)
        except Exception:
            pass
    current.update(updates)
    current["updated_at"] = datetime.utcnow().isoformat() + "Z"
    with open(path, "w") as f:
        json.dump(current, f, indent=2)


def _enqueue_diagram_generation(
    job_id: str,
    outputs_dir: Path,
    diagram_queue: List[Path],
    protein_file: Path,
    job_logger: logging.Logger,
) -> None:
    """Enqueue serial PoseView diagram generation as a separate RQ job."""
    manifest_path = outputs_dir / "diagrams_manifest.json"
    protein_resolved = Path(protein_file).resolve()
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "protein_file": str(protein_resolved),
                "ligands": [p.name for p in diagram_queue],
            },
            f,
            indent=2,
        )
    _write_diagrams_status(
        outputs_dir,
        pending=[p.name for p in diagram_queue],
        in_progress=None,
        completed=[],
        failed=[],
    )
    try:
        redis_conn = redis.from_url(REDIS_URL)
        queue = Queue("omtra_tasks", connection=redis_conn)
        queue.enqueue(
            "worker.generate_diagrams_task",
            job_id,
            job_timeout=600,
            result_ttl=3600,
        )
        job_logger.info(
            f"Enqueued RQ diagram generation job for {len(diagram_queue)} ligands"
        )
    except Exception as e:
        job_logger.error(f"Failed to enqueue diagram generation: {e}", exc_info=True)


def _generate_diagrams_serial(
    outputs_dir: Path,
    ligand_paths: List[Path],
    protein_file: Path,
    job_logger: logging.Logger,
) -> Tuple[List[str], List[str]]:
    """Generate interaction diagrams one ligand at a time."""
    pending_names = [p.name for p in ligand_paths]
    _write_diagrams_status(
        outputs_dir,
        pending=pending_names,
        in_progress=None,
        completed=[],
        failed=[],
    )
    job_logger.info(f"Background diagram generation started for {len(ligand_paths)} ligands (serial)")

    completed: List[str] = []
    failed: List[str] = []
    protein_path = Path(protein_file) if not isinstance(protein_file, Path) else protein_file

    for sdf_path in ligand_paths:
        name = sdf_path.name
        diagram_path = outputs_dir / f"{name}_diagram.svg"
        error_path = outputs_dir / f"{name}_diagram_error.json"

        if diagram_path.exists() or error_path.exists():
            completed.append(name)
            pending_names = [n for n in pending_names if n != name]
            _write_diagrams_status(
                outputs_dir,
                pending=pending_names,
                in_progress=None,
                completed=completed,
                failed=failed,
            )
            continue

        _write_diagrams_status(
            outputs_dir,
            pending=pending_names,
            in_progress=name,
            completed=completed,
            failed=failed,
        )

        try:
            svg, error_msg = _generate_interaction_diagram(sdf_path, protein_path, job_logger)
            if svg and not error_msg:
                with open(diagram_path, "w") as f:
                    f.write(svg)
                completed.append(name)
            else:
                error_detail = error_msg or (
                    "Failed to generate interaction diagram. PoseView could not detect any interactions."
                )
                error_data = {
                    "statusCode": 503,
                    "message": error_detail.split(".")[0] if error_detail else "Failed",
                    "detail": error_detail,
                }
                with open(error_path, "w") as f:
                    json.dump(error_data, f)
                failed.append(name)
                job_logger.warning(f"Failed to generate diagram for {name}: {error_detail}")
        except Exception as e:
            error_data = {
                "statusCode": 500,
                "message": "Failed",
                "detail": f"Failed to generate interaction diagram: {str(e)}",
            }
            with open(error_path, "w") as f:
                json.dump(error_data, f)
            failed.append(name)
            job_logger.warning(f"Error generating diagram for {name}: {e}")

        pending_names = [n for n in pending_names if n != name]
        _write_diagrams_status(
            outputs_dir,
            pending=pending_names,
            in_progress=None,
            completed=completed,
            failed=failed,
        )

    job_logger.info(
        f"Diagram generation finished: {len(completed)} succeeded, {len(failed)} failed"
    )
    return completed, failed


def _load_posecheck_with_timeout(pdb_path: str, timeout: int = 120):
    """Load PoseCheck protein with a hard timeout.

    Returns the initialized PoseCheck object, or None if loading timed out
    or failed.  Uses a daemon thread so a hung hydride process won't block
    the worker forever.
    """
    result = [None]
    error = [None]

    def _load():
        try:
            from posecheck import PoseCheck
            pc = PoseCheck()
            pc.load_protein_from_pdb(pdb_path)
            result[0] = pc
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_load, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        logging.warning(
            f"PoseCheck protein loading timed out after {timeout}s for {pdb_path}"
        )
        return None
    if error[0]:
        logging.warning(f"PoseCheck protein loading failed: {error[0]}")
        return None
    return result[0]


def _is_blank_svg(svg: str) -> bool:
    """Check if SVG is blank or empty"""
    if not svg or not svg.strip():
        return True
    # Remove whitespace and check for minimal content
    trimmed = re.sub(r'\s+', ' ', svg.strip())
    
    # Check if it's just a border rectangle (single path with simple rectangle)
    border_patterns = [
        r'<path[^>]*d="[^"]*M\s+\d+\s+\d+\s+L\s+\d+\s+\d+\s+L\s+\d+\s+\d+\s+L\s+\d+\s+\d+\s+Z',  # Standard border
        r'<path[^>]*d="[^"]*M\s+0\s+0\s+L\s+600\s+0\s+L\s+600\s+600\s+L\s+0\s+600\s+Z',  # Exact 600x600 border
        r'<path[^>]*d="[^"]*M\s+0\s+0\s+L\s+\d+\s+0\s+L\s+\d+\s+\d+\s+L\s+0\s+\d+\s+Z\s+M\s+0\s+0',  # Border with extra M
    ]
    for pattern in border_patterns:
        if re.search(pattern, trimmed, re.IGNORECASE):
            # If it's a short SVG with only border, it's blank
            if len(trimmed) < 500:
                return True
            # Even if longer, if it only has one path element and it's a border, it's blank
            path_count = len(re.findall(r'<path[^>]*>', trimmed, re.IGNORECASE))
            if path_count == 1:
                return True
    
    # Check if it has meaningful content 
    has_meaningful_content = bool(
        re.search(r'<text[^>]*>', trimmed, re.IGNORECASE) or
        re.search(r'<circle[^>]*r="[^"]*"[^>]*>', trimmed, re.IGNORECASE) or
        (re.search(r'<path[^>]*d="[^"]*[ML][^"]*[ML]"', trimmed, re.IGNORECASE) and len(trimmed) > 500) or  # Multiple move/line commands (but not just a border)
        re.search(r'<path[^>]*d="[^"]*[CcQqSsTtAaZz]', trimmed, re.IGNORECASE)  # Curved paths (bezier, arc, etc.)
    )
    # If it's a very short SVG with no meaningful content, it's blank
    if not has_meaningful_content and len(trimmed) < 500:
        return True
    return False


def _generate_interaction_diagram(
    ligand_file: Path,
    protein_file: Path,
    job_logger: logging.Logger,
) -> Tuple[Optional[str], Optional[str]]:
    """Generate interaction diagram for a single ligand file using ProteinsPlus API v2.
    
    Returns:
        Tuple of (svg_content, error_message). If successful, returns (svg, None).
        If failed, returns (None, error_message).
    """
    try:
        import requests
        
        # Read files
        with open(ligand_file, 'r') as f:
            ligand_str = f.read()
        with open(protein_file, 'r') as f:
            protein_str = f.read()
        
        # Convert CIF to PDB if needed
        if protein_file.suffix.lower() == '.cif':
            try:
                from biotite.structure.io import pdb
                from biotite.structure.io.pdbx import CIFFile, get_structure

                cif_file = CIFFile.read(str(protein_file))
                st = get_structure(cif_file, model=1, include_bonds=False)
                st = st[st.res_name != "HOH"]
                st = st[st.element != "H"]
                st = st[st.element != "D"]
                
                pdb_file = pdb.PDBFile()
                pdb_file.set_structure(st)
                pdb_stringio = io.StringIO()
                pdb_file.write(pdb_stringio)
                protein_str = pdb_stringio.getvalue()
            except Exception as conv_e:
                error_msg = f"CIF→PDB conversion failed: {conv_e}"
                job_logger.warning(error_msg)
                return None, error_msg
        
        # Ensure PDB has END record
        protein_lines = protein_str.strip().split('\n')
        has_end = any(line.strip() == 'END' or line.strip().startswith('END') for line in protein_lines[-5:])
        if not has_end:
            protein_str = protein_str.rstrip() + '\nEND\n'
        
        PROTEINS_PLUS_BASE = "https://proteins.plus"
        API_BASE = f"{PROTEINS_PLUS_BASE}/api/v2"
        UPLOAD_URL = f"{API_BASE}/molecule_handler/upload/"
        UPLOAD_JOBS_URL = f"{API_BASE}/molecule_handler/upload/jobs/"
        PROTEINS_URL = f"{API_BASE}/molecule_handler/proteins/"
        POSEVIEW_URL = f"{API_BASE}/poseview/"
        POSEVIEW_JOBS_URL = f"{API_BASE}/poseview/jobs/"
        
        # Step 1: Embed ligand in PDB
        combined_pdb_str = _embed_ligand_in_pdb(ligand_str, protein_str)
        if not combined_pdb_str:
            error_msg = "Failed to embed ligand in PDB"
            job_logger.warning(f"{error_msg} for {ligand_file.name}")
            return None, error_msg
        
        # Step 2: Extract ligand from combined PDB
        combined_pdb_bytes = combined_pdb_str.encode('utf-8')
        combined_pdb_file_obj = io.BytesIO(combined_pdb_bytes)
        combined_pdb_file_obj.seek(0)
        files = {'protein_file': ('protein.pdb', combined_pdb_file_obj, 'chemical/x-pdb')}
        
        preprocessing_job_submission = requests.post(UPLOAD_URL, files=files, timeout=120).json()
        preprocessing_job = _poll_job(
            preprocessing_job_submission["job_id"],
            UPLOAD_JOBS_URL,
            max_polls=PROTEINSPLUS_UPLOAD_MAX_POLLS,
        )
        
        if preprocessing_job.get('status') != 'success':
            error_msg = f"Preprocessing failed: {preprocessing_job.get('error', 'Unknown error')}"
            job_logger.warning(f"{error_msg} for {ligand_file.name}")
            return None, error_msg
        
        protein_combined = requests.get(f"{PROTEINS_URL}{preprocessing_job['output_protein']}/", timeout=15).json()
        
        if not protein_combined.get('ligand_set'):
            error_msg = "No ligands found in combined PDB"
            job_logger.warning(f"{error_msg} for {ligand_file.name}")
            return None, error_msg
        
        ligand_id = protein_combined['ligand_set'][0]
        
        # Step 3: Upload original PDB
        original_protein_bytes = protein_str.encode('utf-8')
        original_protein_file_obj = io.BytesIO(original_protein_bytes)
        original_protein_file_obj.seek(0)
        files2 = {'protein_file': ('protein.pdb', original_protein_file_obj, 'chemical/x-pdb')}
        preprocessing_job_submission2 = requests.post(UPLOAD_URL, files=files2, timeout=120).json()
        preprocessing_job2 = _poll_job(
            preprocessing_job_submission2["job_id"],
            UPLOAD_JOBS_URL,
            max_polls=PROTEINSPLUS_UPLOAD_MAX_POLLS,
        )
        
        if preprocessing_job2.get('status') != 'success':
            error_msg = "Failed to upload original protein"
            job_logger.warning(f"{error_msg} for {ligand_file.name}")
            return None, error_msg
        
        protein_original = requests.get(f"{PROTEINS_URL}{preprocessing_job2['output_protein']}/", timeout=15).json()
        
        # Step 4: Generate PoseView
        query = {'protein_id': protein_original['id'], 'ligand_id': ligand_id}
        poseview_job_submission = requests.post(POSEVIEW_URL, data=query, timeout=120).json()
        poseview_job = _poll_job(poseview_job_submission["job_id"], POSEVIEW_JOBS_URL)

        if poseview_job.get("status") != "success":
            pv_status = poseview_job.get("status", "")
            pv_error = poseview_job.get("error")
            if pv_status in ("pending", "running"):
                error_msg = "PoseView timed out while generating the interaction diagram."
            else:
                error_msg = f"PoseView failed: {pv_error or pv_status or 'Unknown error'}"
            job_logger.warning(f"{error_msg} for {ligand_file.name}")
            return None, error_msg
        
        image_url = poseview_job.get('image')
        if not image_url:
            error_msg = "PoseView succeeded but no image URL"
            job_logger.warning(f"{error_msg} for {ligand_file.name}")
            return None, error_msg
        
        # Fetch SVG
        svg_response = requests.get(image_url, timeout=30)
        svg_response.raise_for_status()
        svg_content = svg_response.text
        
        # Check if SVG is blank
        if _is_blank_svg(svg_content):
            error_msg = "Generated diagram is empty or blank"
            job_logger.warning(f"{error_msg} for {ligand_file.name}")
            return None, error_msg
        
        # Clean up SVG (remove width/height attributes)
        svg_content = re.sub(r'\s+width="\d+pt"\s+height="\d+pt"', '', svg_content)
        
        return svg_content, None
        
    except Exception as e:
        error_msg = f"Failed to generate interaction diagram: {str(e)}"
        job_logger.warning(f"{error_msg} for {ligand_file.name}")
        return None, error_msg


def generate_diagrams_task(job_id: str) -> Dict[str, Any]:
    """RQ task: generate interaction diagrams serially after sampling completes."""
    job_dir = get_job_directory(job_id)
    log_file = job_dir / "logs" / "job.log"
    job_logger = setup_logging(level="INFO", log_file=log_file)
    outputs_dir = job_dir / "outputs"
    manifest_path = outputs_dir / "diagrams_manifest.json"

    if not manifest_path.exists():
        job_logger.warning(f"No diagrams_manifest.json for job {job_id}")
        return {"status": "skipped", "reason": "no manifest"}

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        protein_file = Path(manifest["protein_file"])
        ligand_paths = [outputs_dir / name for name in manifest.get("ligands", [])]
        ligand_paths = [p for p in ligand_paths if p.exists()]
        if not ligand_paths:
            job_logger.warning("Diagram manifest has no existing ligand files")
            return {"status": "skipped", "reason": "no ligands"}

        job_logger.info(f"Diagram RQ task started for {len(ligand_paths)} ligands")
        completed, failed = _generate_diagrams_serial(
            outputs_dir, ligand_paths, protein_file, job_logger
        )
        return {"status": "done", "completed": len(completed), "failed": len(failed)}
    except Exception as e:
        job_logger.error(f"Diagram generation task failed: {e}", exc_info=True)
        raise


def sampling_task(job_id: str, params: Dict[str, Any], input_files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main sampling task that runs the molecular generation model
    """
    # Ensure the full job directory structure exists (inputs/outputs/logs)
    job_dir = create_job_directory(job_id)
    log_file = job_dir / "logs" / "job.log"
    
    # Setup job-specific logging
    job_logger = setup_logging(
        level='INFO',
        log_file=log_file
    )
    
    try:
        log_job_event(job_logger, job_id, "sampling_started", params=params)

        # Get sampling mode and checkpoint path
        sampling_mode = params.get('sampling_mode')
        valid_sampling_modes = {
            'Unconditional',
            'Pharmacophore-conditioned',
            'Protein-conditioned',
            'Protein+Pharmacophore-conditioned',
        }
        if sampling_mode not in valid_sampling_modes:
            raise ValueError(
                f"Unknown sampling_mode {sampling_mode!r}. "
                f"Expected one of: {', '.join(sorted(valid_sampling_modes))}"
            )
        checkpoint_path = params.get('checkpoint_path')

        if not checkpoint_path:
            raise ValueError("No checkpoint path provided")

        # When fixed fragments are selected, run_omtra_sampler handles model
        # loading via cli.run_sample (which switches to partial task/checkpoint)
        has_fixed = bool(
            (params.get('fixed_atom_indices') and len(params.get('fixed_atom_indices', [])) > 0)
            or (params.get('fixed_brics_fragments') and len(params.get('fixed_brics_fragments', [])) > 0)
        )
        if has_fixed:
            model = None
        else:
            model = load_omtra_model(checkpoint_path, sampling_mode)
            if model is None:
                raise ValueError("No model found")
        
        result = run_omtra_sampler(job_id, params, input_files, model, job_logger)
        
        # Only log completion event if result is not None and is a dict
        if result is not None and isinstance(result, dict):
            log_job_event(job_logger, job_id, "sampling_completed", **result)
        else:
            # Log completion with a message if result is None or not a dict
            log_job_event(job_logger, job_id, "sampling_completed", 
                         message="Sampling completed but no result data returned" if result is None else f"Sampling completed with non-dict result: {type(result)}")
        
        # Update job status in Redis to SUCCEEDED (sampling done; diagrams may still be running in RQ)
        try:
            redis_conn = redis.from_url(REDIS_URL)
            job_data = redis_conn.get(f"job:{job_id}")
            if job_data:
                job_metadata = json.loads(job_data)
                job_metadata['status'] = 'SUCCEEDED'
                job_metadata['completed_at'] = datetime.utcnow().isoformat()
                redis_conn.setex(
                    f"job:{job_id}",
                    timedelta(hours=48),  # JOB_TTL_HOURS
                    json.dumps(job_metadata)
                )
        except Exception as e:
            job_logger.warning(f"Failed to update job status in Redis: {e}")
        
        return result
        
    except Exception as e:
        error_msg = f"Sampling failed: {str(e)}"
        job_logger.error(error_msg, exc_info=True)
        log_job_event(job_logger, job_id, "sampling_failed", error=error_msg)
        
        # Update job status in Redis to FAILED
        try:
            redis_conn = redis.from_url(REDIS_URL)
            job_data = redis_conn.get(f"job:{job_id}")
            if job_data:
                job_metadata = json.loads(job_data)
                job_metadata['status'] = 'FAILED'
                job_metadata['error'] = error_msg
                job_metadata['failed_at'] = datetime.utcnow().isoformat()
                redis_conn.setex(
                    f"job:{job_id}",
                    timedelta(hours=48),  # JOB_TTL_HOURS
                    json.dumps(job_metadata)
                )
        except Exception as redis_e:
            job_logger.warning(f"Failed to update job status in Redis: {redis_e}")
        
        # Write error to log file
        with open(log_file, 'a') as f:
            f.write(f"\nERROR: {error_msg}\n")
            f.write(traceback.format_exc())
        
        raise


def _format_pocket_residue(residue: Dict[str, Any]) -> str:
    return f"{residue['chain']}:{residue['res_id']}"


def _save_pocket_reference_coords(coords: Any, output_dir: Path) -> Path:
    coord_array = np.asarray(coords, dtype=np.float32)
    if coord_array.ndim != 2 or coord_array.shape[1] != 3 or coord_array.shape[0] == 0:
        raise ValueError("Detected pocket coordinate selection must be a non-empty Nx3 array")
    coord_path = output_dir / "pocket_reference_coords.npy"
    np.save(coord_path, coord_array)
    return coord_path


def _resolve_fixed_structure_for_task(
    params: Dict[str, Any],
    base_task_name: str,
    partial_task_mapping: Optional[Dict[str, str]] = None,
) -> tuple[str, Optional[str], Optional[str], bool]:
    """Return (task_name, fixed_atoms_str, fixed_brics_str, has_fixed)."""
    fixed_atom_indices = params.get('fixed_atom_indices') or []
    fixed_brics_fragments = params.get('fixed_brics_fragments') or []

    fixed_atoms_str = (
        ','.join(str(i) for i in fixed_atom_indices) if fixed_atom_indices else None
    )
    fixed_brics_str = (
        ','.join(str(i) for i in fixed_brics_fragments)
        if fixed_brics_fragments
        else None
    )
    has_fixed = bool(fixed_atoms_str or fixed_brics_str)

    task_name = base_task_name
    if has_fixed and partial_task_mapping:
        task_name = partial_task_mapping.get(base_task_name, base_task_name)
    elif has_fixed and base_task_name == 'rigid_docking_condensed':
        task_name = 'partial_rigid_docking_condensed'

    return task_name, fixed_atoms_str, fixed_brics_str, has_fixed


def docking_task(job_id: str, params: Dict[str, Any], input_files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Execute a docking job.
    Similar to sampling_task but for docking workflows.
    """
    job_logger = logging.getLogger(f"job.{job_id}")
    job_logger.info(f"Starting docking job {job_id}")
    
    try:
        # Get job directory
        job_dir = get_job_directory(job_id)
        if not job_dir or not job_dir.exists():
            raise ValueError(f"Job directory not found for job {job_id}")
        
        outputs_dir = job_dir / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        
        # Extract parameters
        docking_mode = params.get('docking_mode')
        n_samples = params.get('n_samples', 10)
        n_timesteps = params.get('steps', 100)
        batch_size = params.get('batch_size', 500)
        seed = params.get('seed')
        pocket_selection = params.get('pocket_selection')
        
        # Map docking mode to task name
        docking_task_mapping = {
            'Rigid Docking': 'rigid_docking_condensed',
            'Rigid Docking + Pharmacophore': 'rigid_docking_pharmacophore_condensed'
        }
        if docking_mode not in docking_task_mapping:
            raise ValueError(
                f"Unknown docking_mode {docking_mode!r}. "
                f"Expected one of: {', '.join(sorted(docking_task_mapping))}"
            )
        task_name, fixed_atoms_str, fixed_brics_str, has_fixed = _resolve_fixed_structure_for_task(
            params,
            docking_task_mapping[docking_mode],
        )
        if has_fixed:
            job_logger.info(
                f"Partial fixed-structure docking: task={task_name}, "
                f"fixed_atoms={fixed_atoms_str}, fixed_brics={fixed_brics_str}"
            )

        # Get checkpoint path (partial tasks use gnn_partial_prot_cond.ckpt directly)
        from omtra.utils.checkpoints import get_checkpoint_path_for_webapp
        if has_fixed and task_name.startswith('partial_'):
            checkpoint_path = CHECKPOINT_DIR / 'gnn_partial_prot_cond.ckpt'
        else:
            checkpoint_path = get_checkpoint_path_for_webapp(docking_mode, CHECKPOINT_DIR)
        if not checkpoint_path or not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found for {docking_mode}")
        
        # Use the same run_omtra_sampler infrastructure
        class MockArgs:
            def __init__(self):
                self.checkpoint = Path(checkpoint_path)
                self.task = task_name
                self.dataset = "pharmit"
                self.n_samples = n_samples
                self.n_replicates = 1
                self.dataset_start_idx = 0
                self.n_timesteps = n_timesteps
                self.batch_size = batch_size
                self.visualize = False
                self.output_dir = outputs_dir
                self.pharmit_path = None
                self.plinder_path = None
                self.split = 'val'
                self.stochastic_sampling = False
                self.noise_scaler = 1.0
                self.eps = 0.01
                self.use_gt_n_lig_atoms = False
                self.n_lig_atom_margin = 0.15
                self.n_lig_atoms_mean = None
                self.n_lig_atoms_mean = None
                self.n_lig_atoms_std = None
                self.pharmacophore_tolerance = params.get('pharmacophore_tolerance', 0.0)
                self.metrics = False
                self.protein_file = None
                self.ligand_file = None
                self.pharmacophore_file = None
                self.pocket_ligand = None
                self.pocket_center = None
                self.pocket_residues = None
                self.pocket_coords = None
                self.bbox_length = 23.0
                self.input_files_dir = None
                self.g_list_from_files = None
                self.fixed_atoms = fixed_atoms_str
                self.fixed_brics_fragments = fixed_brics_str
        
        # Handle input files
        if not input_files:
            raise ValueError("Docking requires input files (protein and ligand)")
        
        protein_file = None
        main_ligand_file = None
        pharmacophore_file = None
        
        # Track all SDF files to resolve pocket_ligand if needed
        all_sdfs = {} # filename -> path
        
        for file_info in input_files:
            file_path = Path(file_info['path'])
            filename = file_info['filename']
            filename_lower = filename.lower()
            
            if filename_lower.endswith(('.pdb', '.cif')):
                protein_file = file_path
            elif filename_lower.endswith('.sdf'):
                all_sdfs[filename] = file_path
                # Default main ligand is the first SDF found, 
                # but we might refine this after identifying pocket_ligand
                if main_ligand_file is None:
                    main_ligand_file = file_path
            elif filename_lower.endswith('.xyz'):
                pharmacophore_file = file_path
        
        if not protein_file:
            raise ValueError("Docking requires a protein file (PDB or CIF)")
        
        # Handle pocket selection first to identify pocket_ligand
        args = MockArgs()
        args.protein_file = protein_file
        
        if pocket_selection:
            pocket_type = pocket_selection.get('type')
            if pocket_type == 'center':
                center = pocket_selection.get('value')
                if isinstance(center, list) and len(center) == 3:
                    args.pocket_center = f"{center[0]},{center[1]},{center[2]}"
                    args.bbox_length = pocket_selection.get('bbox_length', 23.0)
            elif pocket_type == 'residues':
                residues = pocket_selection.get('value', [])
                if residues:
                    res_strs = [_format_pocket_residue(r) for r in residues]
                    args.pocket_residues = ','.join(res_strs)
            elif pocket_type == 'coords':
                args.pocket_coords = _save_pocket_reference_coords(
                    pocket_selection.get('value', []),
                    outputs_dir,
                )
            elif pocket_type == 'file':
                pocket_filename = pocket_selection.get('value')
                if pocket_filename in all_sdfs:
                    args.pocket_ligand = all_sdfs[pocket_filename]
                else:
                    args.pocket_ligand = main_ligand_file
                    job_logger.warning(f"Pocket ligand filename {pocket_filename} not in all_sdfs, falling back to {main_ligand_file}")
        
        # Now identify the main ligand (to be docked)
        # It should be the one that is NOT the pocket ligand, if possible.
        if len(all_sdfs) > 1 and args.pocket_ligand:
            for fname, fpath in all_sdfs.items():
                if fpath != args.pocket_ligand:
                    main_ligand_file = fpath
                    break
        
        if not main_ligand_file:
             raise ValueError("Docking requires a ligand file (SDF) to dock")
        
        args.ligand_file = main_ligand_file
        args.pharmacophore_file = pharmacophore_file
        
        if args.pocket_ligand:
             job_logger.info(f"Targeting pocket defined by reference ligand: {args.pocket_ligand.name}")
        job_logger.info(f"Main ligand to dock: {args.ligand_file.name}")
        
        # Run docking
        job_logger.info(f"Starting docking: {docking_mode}, {n_samples} samples, {n_timesteps} steps")
        
        summary = run_omtra_sampler(
            args=args,
            job_id=job_id,
            params=params,
            input_files=input_files,
            job_logger=job_logger,
            outputs_dir=outputs_dir,
            sampling_mode=docking_mode
        )
        
        job_logger.info("Docking completed successfully")
        
        # Update job status in Redis to SUCCEEDED
        try:
            redis_conn = redis.from_url(REDIS_URL)
            job_data = redis_conn.get(f"job:{job_id}")
            if job_data:
                job_metadata = json.loads(job_data)
                # Update job_metadata['params'] from summary['parameters_used']
                # But only if summary has data and we don't have better data already
                if summary and summary.get('parameters_used'):
                    job_params = job_metadata.get('params', {})
                    # Only update if the summary params have actual useful info (like docking_mode)
                    if summary['parameters_used'].get('docking_mode') or summary['parameters_used'].get('sampling_mode'):
                        job_params.update(summary['parameters_used'])
                        job_metadata['params'] = job_params
                
                # Ensure status is SUCCEEDED and completed_at is set
                job_metadata['status'] = 'SUCCEEDED'
                job_metadata['completed_at'] = datetime.utcnow().isoformat()

                redis_conn.setex(
                    f"job:{job_id}",
                    timedelta(hours=48),  # JOB_TTL_HOURS
                    json.dumps(job_metadata)
                )
        except Exception as e:
            job_logger.warning(f"Failed to update job status in Redis: {e}")
            
        return summary
        
    except Exception as e:
        error_msg = f"Docking failed: {str(e)}"
        job_logger.error(error_msg, exc_info=True)
        
        # Update job status in Redis to FAILED
        try:
            redis_conn = redis.from_url(REDIS_URL)
            job_data = redis_conn.get(f"job:{job_id}")
            if job_data:
                job_metadata = json.loads(job_data)
                job_metadata['status'] = 'FAILED'
                job_metadata['error'] = error_msg
                job_metadata['failed_at'] = datetime.utcnow().isoformat()
                redis_conn.setex(
                    f"job:{job_id}",
                    timedelta(hours=48),  # JOB_TTL_HOURS
                    json.dumps(job_metadata)
                )
        except Exception as redis_e:
            job_logger.warning(f"Failed to update job status in Redis: {redis_e}")
            
        raise


def _run_gnina_score_only(lig_file, prot_file, env=None):
    """
    Run GNINA with --score_only flag to compute VINA and CNN scores.
    Based on docking_eval.py _run_gnina function.
    
    Args:
        lig_file: Path to ligand SDF file
        prot_file: Path to protein PDB file
        env: Environment variables dict (optional)
        
    Returns:
        Dictionary with VINA score: {'minimizedAffinity': float}
        Returns None if GNINA fails
    """
    from rdkit import Chem
    
    # Find GNINA binary - check standard installation location
    possible_paths = [
        Path('/usr/local/bin/gnina'),
        Path('/usr/local/bin/gnina.1.3.2'),
    ]
    gnina_binary = next((p for p in possible_paths if p.exists()), None)
    
    if gnina_binary is None:
        logging.error(f"GNINA binary not found in {possible_paths}")
        return None
    
    # Create temporary output SDF file
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp:
        output_sdf = Path(tmp.name)
    
    try:
        scores = {
            'minimizedAffinity': None,  # vina score only
        }
        
        vina_cmd = [
            str(gnina_binary),
            '-r', str(prot_file),
            '-l', str(lig_file),
            '--score_only',
            '-o', str(output_sdf),
            '--seed', '42'
        ]
        
        # Set environment if provided
        cmd_env = os.environ.copy()
        if env:
            cmd_env.update(env)
        
        # Set LD_LIBRARY_PATH for CUDA libraries if needed
        # Include cuDNN paths
        cudnn_paths = [
            '/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib',
            '/usr/local/cuda/lib64',
            '/usr/local/cuda-12.1/lib64',
            '/usr/local/cuda-12.0/lib64',
        ]
        existing_paths = [p for p in cudnn_paths if Path(p).exists()]
        
        if 'LD_LIBRARY_PATH' in cmd_env and cmd_env['LD_LIBRARY_PATH']:
            # Append to existing LD_LIBRARY_PATH
            existing_ld_path = cmd_env['LD_LIBRARY_PATH'].split(':')
            all_paths = existing_ld_path + existing_paths
            # Remove duplicates while preserving order
            seen = set()
            unique_paths = []
            for p in all_paths:
                if p and p not in seen:
                    seen.add(p)
                    unique_paths.append(p)
            cmd_env['LD_LIBRARY_PATH'] = ':'.join(unique_paths)
        elif existing_paths:
            # Set new LD_LIBRARY_PATH
            cmd_env['LD_LIBRARY_PATH'] = ':'.join(existing_paths)
        
        try:
            import torch
            if torch.cuda.is_available():
                compute_cap = torch.cuda.get_device_capability(0)
                if compute_cap[0] < 7:
                    logging.info(f"GPU compute capability {compute_cap[0]}.{compute_cap[1]} detected. GNINA may fail on older GPUs, but will attempt anyway.")
        except Exception:
            pass
        
        cmd_result = subprocess.run(
            vina_cmd,
            capture_output=True,
            text=True,
            env=cmd_env,
            timeout=300  # 5 minute timeout
        )
        
        # If GPU mode fails due to compute capability, try CPU mode
        if cmd_result.returncode != 0:
            stderr_lower = cmd_result.stderr.lower() if cmd_result.stderr else ""
            if "no kernel image is available" in stderr_lower or "compute capability" in stderr_lower:
                logging.warning(f"GNINA GPU mode failed due to compute capability mismatch. Falling back to CPU mode...")
                logging.warning(f"GNINA stderr: {cmd_result.stderr[:300]}")
                
                # Retry with CPU mode (CUDA_VISIBLE_DEVICES='')
                cpu_env = cmd_env.copy()
                cpu_env['CUDA_VISIBLE_DEVICES'] = ''
                cmd_result = subprocess.run(
                    vina_cmd,
                    capture_output=True,
                    text=True,
                    env=cpu_env,
                    timeout=600
                )
                
                if cmd_result.returncode != 0:
                    logging.warning(f"GNINA CPU mode also failed (return code {cmd_result.returncode})")
                    logging.warning(f"GNINA stderr: {cmd_result.stderr[:500]}")
                    logging.warning(f"GNINA stdout: {cmd_result.stdout[:500]}")
                    return None
            else:
                logging.warning(f"GNINA scoring failed for {lig_file} (return code {cmd_result.returncode})")
                logging.warning(f"GNINA stdout: {cmd_result.stdout[:1000]}")
                logging.warning(f"GNINA stderr: {cmd_result.stderr[:1000]}")
                return None
        
        # Read scores from output SDF
        if not output_sdf.exists():
            logging.warning(f"GNINA output file does not exist: {output_sdf}")
            return None
        
        supplier = Chem.SDMolSupplier(str(output_sdf), sanitize=False, removeHs=False)
        
        mol_count = 0
        for lig in supplier:
            if lig is None:
                continue
            mol_count += 1
            
            # Extract scores from molecule properties
            for score_name in scores.keys():
                try:
                    if lig.HasProp(score_name):
                        scores[score_name] = float(lig.GetProp(score_name))
                except (ValueError, KeyError):
                    pass
        
        if mol_count == 0:
            logging.warning(f"GNINA output SDF had no molecules for {lig_file}")
            return None
        
        # Return None if no scores were extracted
        if all(v is None for v in scores.values()):
            logging.warning(f"GNINA output SDF had no score properties for {lig_file}")
            return None
        
        return scores
    
    except subprocess.TimeoutExpired:
        logging.warning(f"GNINA scoring timed out for {lig_file}")
        return None
    except Exception as e:
        logging.warning(f"GNINA scoring error for {lig_file}: {e}")
        return None
    finally:
        # Clean up temporary output file
        try:
            output_sdf.unlink(missing_ok=True)
        except Exception:
            pass


def compute_fast_molecule_metrics(
    mol,
    sample_name=None,
    sampling_mode=None,
    protein_file=None,
    posecheck_obj=None,
    metrics_options: Optional[Dict[str, bool]] = None,
):
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    opts = metrics_options if metrics_options is not None else dict(DEFAULT_METRICS_OPTIONS)
    compute_pb = opts.get('posebusters', True)
    compute_posecheck = opts.get('posecheck', True)
    compute_strain = opts.get('strain', True)
    compute_vina = opts.get('vina', True)

    # Store a copy of the molecule for PoseCheck before any sanitization attempts
    mol_to_preserve = Chem.Mol(mol) if mol is not None else None

    metrics = {
        'sample_name': sample_name,
        'n_atoms': 0,
        'is_connected': False,
        'molecular_weight': None,
        'logp': None,
        'tpsa': None,
    }
    
    if compute_strain:
        metrics['strain'] = None

    if compute_pb:
        metrics['pb_valid'] = False
        metrics['pb_failing_checks'] = None

    if sampling_mode in PROTEIN_INVOLVING_MODES:
        if compute_vina:
            metrics['vina_score'] = None
        if compute_posecheck:
            metrics.update({
                'clashes': None,
                'HBAcceptor': None,
                'HBDonor': None,
                'Hydrophobic': None,
            })
    
    if mol is None:
        return metrics
    
    sanitized_ok = False
    try:
        # Basic molecule properties
        metrics['n_atoms'] = mol.GetNumAtoms()
        
        # Check validity and connectedness
        try:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            metrics['n_connected_components'] = len(mol_frags)
            metrics['is_connected'] = (len(mol_frags) == 1)
        except Exception:
            metrics['n_connected_components'] = 0
            metrics['is_connected'] = False
        
        try:
            Chem.SanitizeMol(mol)
            sanitized_ok = True
            
            try:
                metrics['molecular_weight'] = round(Descriptors.MolWt(mol), 2)
            except Exception:
                pass
            try:
                metrics['logp'] = round(Descriptors.MolLogP(mol), 2)
            except Exception:
                pass
            try:
                metrics['tpsa'] = round(Descriptors.TPSA(mol), 2)
            except Exception:
                pass
                
        except Chem.rdchem.AtomValenceException:
            pass
        except Chem.rdchem.KekulizeException:
            pass
        except Exception:
            pass
        
        try:
            if any(atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms()):
                return metrics
        except Exception:
            pass
        
        if compute_pb and not metrics.get('is_connected', False):
            metrics['pb_valid'] = False
            if metrics.get('pb_failing_checks') is None:
                metrics['pb_failing_checks'] = ['connectivity']
            elif 'connectivity' not in metrics['pb_failing_checks']:
                metrics['pb_failing_checks'].append('connectivity')

        # If a protein file is available, compute pb_valid for any task
        if compute_pb and protein_file and Path(protein_file).exists():
            if metrics.get('is_connected', True):
                try:
                    # Compute pb_valid using PoseBusters
                    from omtra.eval.metrics.posebusters import pb_validate
                    df_pb = pb_validate(gen_ligs=[mol], mode="dock", true_lig=None, prot_file=str(protein_file))
                    if len(df_pb) > 0:

                        row_mask = df_pb['pb_sanitization'] == True
                        if row_mask.any():
                            # Get rows where sanitization is True, then check all pb_ columns
                            valid_rows = df_pb[row_mask]
                            # Filter out NA values and convert to boolean
                            # Only check columns that start with 'pb_' and are not 'pb_sanitization'
                            pb_cols = [col for col in valid_rows.columns if col.startswith('pb_') and col != 'pb_sanitization']
                            if pb_cols:
                                # For each row, check if all pb_ columns are True (excluding NA)
                                row_vals = []
                                for idx, row in valid_rows.iterrows():
                                    # Check each pb_ column, treating NA as False
                                    all_valid = True
                                    for col in pb_cols:
                                        val = row[col]
                                        # Handle pandas NA values and numpy types (numpy.bool_, numpy.int64, etc.)
                                        try:
                                            # Check if value is NaN, or if it's a boolean/numeric type that is falsy
                                            if pd.isna(val):
                                                all_valid = False
                                                break
                                            elif isinstance(val, (bool, np.bool_)) and not bool(val):
                                                all_valid = False
                                                break
                                            elif isinstance(val, (int, float, np.integer, np.floating)) and not bool(val):
                                                all_valid = False
                                                break
                                        except (TypeError, ValueError):
                                            all_valid = False
                                            break
                                    row_vals.append(all_valid)
                                metrics['pb_valid'] = bool(row_vals[0]) if len(row_vals) > 0 else False
                            else:
                                # No pb_ columns to check, just use sanitization
                                metrics['pb_valid'] = bool(sanitized_ok)
                        else:
                            metrics['pb_valid'] = False
                        try:
                            # Collect failing checks for diagnostics and metrics
                            failing = []
                            try:
                                first_row = df_pb.iloc[0]
                                # Check all pb_ columns (including sanitization)
                                for col, val in first_row.items():
                                    if col.startswith('pb_'):
                                        try:
                                            # Skip 'valid' column - don't include it in failing checks
                                            check_name = col[3:] if col.startswith('pb_') else col
                                            if check_name.lower() == 'valid':
                                                continue
                                            # Check if value is False, 0, or NaN (handle numpy types)
                                            if pd.isna(val):
                                                failing.append(check_name)
                                            elif isinstance(val, (bool, np.bool_)) and not bool(val):
                                                failing.append(check_name)
                                            elif isinstance(val, (int, float, np.integer, np.floating)) and not bool(val):
                                                failing.append(check_name)
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                            # Store failing checks in metrics
                            if metrics['pb_valid'] is False:
                                metrics['pb_failing_checks'] = failing if failing else ['sanitization']  # If no specific failures but invalid, sanitization likely failed
                            else:
                                metrics['pb_failing_checks'] = []  # Empty list when valid
                        except Exception:
                            pass
                    else:
                        metrics['pb_valid'] = False
                        metrics['pb_failing_checks'] = ['sanitization']  # Empty df_pb usually means sanitization failed
                except Exception as e:
                    # If PoseBusters fails, set pb_valid to False
                    metrics['pb_valid'] = False
                    # Try to extract sanitization error message from molecule property or exception
                    error_msg = None
                    try:
                        if mol.HasProp("_Invalid"):
                            error_msg = mol.GetProp("_Invalid")
                            if error_msg.startswith("Sanitization failed: "):
                                error_msg = error_msg[len("Sanitization failed: "):]
                    except Exception:
                        pass
                    if not error_msg:
                        error_msg = str(e) if "No module named" in str(e) else "pb_runtime_error"
                    metrics['pb_failing_checks'] = [error_msg] if error_msg and len(error_msg) > 0 and error_msg != "error" else ['error']
                    logging.warning(f"Failed to compute pb_valid for {sample_name}: {e}")
        elif compute_pb:
            # No protein file: compute pb_valid using 'dock' config for unconditional/pharmacophore jobs
            if sampling_mode in ['Unconditional', 'Pharmacophore-conditioned'] and metrics.get('is_connected', True):
                try:
                    # Compute pb_valid using PoseBusters with 'dock' config (no protein required)
                    from omtra.eval.metrics.posebusters import pb_validate
                    df_pb = pb_validate(gen_ligs=[mol], mode="dock", true_lig=None, prot_file=None)
                    if len(df_pb) > 0:
                        # Match docking_eval: pb_valid True only if sanitization True AND all checks pass
                        row_mask = df_pb['pb_sanitization'] == True
                        row_vals = df_pb[row_mask].values.astype(bool).all(axis=1)
                        metrics['pb_valid'] = bool(row_vals[0]) if len(row_vals) > 0 else False
                        
                        # Collect failing checks
                        failing = []
                        try:
                            first_row = df_pb.iloc[0]
                            # Check all pb_ columns (including sanitization)
                            for col, val in first_row.items():
                                if col.startswith('pb_'):
                                    try:
                                        # Skip 'valid' column - don't include it in failing checks
                                        check_name = col[3:] if col.startswith('pb_') else col
                                        if check_name.lower() == 'valid':
                                            continue
                                        # Check if value is False, 0, or NaN (handle numpy types)
                                        if pd.isna(val):
                                            failing.append(check_name)
                                        elif isinstance(val, (bool, np.bool_)) and not bool(val):
                                            failing.append(check_name)
                                        elif isinstance(val, (int, float, np.integer, np.floating)) and not bool(val):
                                            failing.append(check_name)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        
                        # Store failing checks in metrics
                        if metrics['pb_valid'] is False:
                            metrics['pb_failing_checks'] = failing if failing else ['sanitization']
                        else:
                            metrics['pb_failing_checks'] = []  # Empty list when valid
                    else:
                        metrics['pb_valid'] = False
                        metrics['pb_failing_checks'] = ['pb_sanitization']  # Empty df_pb usually means sanitization failed
                except Exception as e:
                    # If PoseBusters fails, fall back to sanitization check
                    metrics['pb_valid'] = bool(sanitized_ok)
                    if not sanitized_ok:
                        # Try to extract sanitization error message from molecule property or exception
                        error_msg = None
                        try:
                            if mol.HasProp("_Invalid"):
                                error_msg = mol.GetProp("_Invalid")
                                if error_msg.startswith("Sanitization failed: "):
                                    error_msg = error_msg[len("Sanitization failed: "):]
                        except Exception:
                            pass
                        if not error_msg:
                            error_msg = str(e) if "No module named" in str(e) else "pb_runtime_error"
                        metrics['pb_failing_checks'] = [error_msg] if error_msg and len(error_msg) > 0 and error_msg != "error" else ['error']
                    else:
                        metrics['pb_failing_checks'] = []  # No error if sanitization passed
                    logging.warning(f"Failed to compute pb_valid for {sample_name} (no protein): {e}, using sanitization check")
            elif compute_pb:
                metrics['pb_valid'] = bool(sanitized_ok)
                metrics['pb_failing_checks'] = [] if sanitized_ok else ['sanitization']
        
        # PoseCheck interactions and clashes for protein-involving jobs
        if (
            sampling_mode in PROTEIN_INVOLVING_MODES
            and compute_posecheck
        ):
            try:
                if not str(protein_file).lower().endswith('.pdb'):
                    pass
                elif posecheck_obj is None:
                    logging.info(f"Skipping PoseCheck metrics for {sample_name} (protein loading failed or timed out)")
                else:
                    pc = posecheck_obj
                    pc.load_ligands_from_mols([mol_to_preserve])

                    try:
                        clashes = pc.calculate_clashes()
                        try:
                            metrics['clashes'] = int(round(float(clashes.iloc[0]) if hasattr(clashes, 'iloc') else float(clashes[0])))
                        except Exception:
                            metrics['clashes'] = None
                    except Exception as clash_e:
                        logging.warning(f"Failed to compute clashes for {sample_name}: {clash_e}")
                        metrics['clashes'] = None

                    try:
                        interactions = pc.calculate_interactions()
                        label_map = {
                            'HBAcceptor': ['HBAcceptor', 'HBondAcceptor', 'HydrogenAcceptor', 'Acceptor'],
                            'HBDonor': ['HBDonor', 'HBondDonor', 'HydrogenDonor', 'Donor'],
                            'Hydrophobic': ['Hydrophobic'],
                        }
                        def column_matches(col, keywords):
                            try:
                                if isinstance(col, tuple) or isinstance(getattr(interactions.columns, 'levels', None), list):
                                    parts = [str(p) for p in (col if isinstance(col, tuple) else (col,))]
                                    return any(any(k.lower() in p.lower() for k in keywords) for p in parts)
                                return any(k.lower() in str(col).lower() for k in keywords)
                            except Exception:
                                return False
                        for i_type, keywords in label_map.items():
                            matched_cols = [col for col in interactions.columns if column_matches(col, keywords)]
                            if matched_cols:
                                try:
                                    i_sum = interactions[matched_cols].sum(axis=1)
                                    val = float(i_sum.iloc[0]) if len(i_sum) > 0 else 0.0
                                except Exception:
                                    try:
                                        val = float(interactions[matched_cols].sum(axis=1)[0])
                                    except Exception:
                                        val = 0.0
                                metrics[i_type] = int(round(val))
                            else:
                                metrics[i_type] = 0
                    except Exception as int_e:
                        logging.warning(f"Failed to compute interaction metrics for {sample_name}: {int_e}")
            except Exception as e:
                logging.warning(f"PoseCheck interactions failed for {sample_name}: {e}")

        # Strain
        if compute_strain:
            try:
                if posecheck_obj is not None:
                    pc = posecheck_obj
                    pc.load_ligands_from_mols([mol_to_preserve])
                else:
                    from posecheck import PoseCheck
                    pc = PoseCheck()
                    pc.load_ligands_from_mols([mol_to_preserve])
                
                strain = pc.calculate_strain_energy()
                if hasattr(strain, 'iloc'):
                    metrics['strain'] = round(float(strain.iloc[0]), 2)
                else:
                    metrics['strain'] = round(float(strain[0]), 2)
            except Exception as strain_e:
                logging.warning(f"Failed to compute strain for {sample_name}: {strain_e}")
                metrics['strain'] = None

        if compute_vina and protein_file and (str(protein_file).lower().endswith('.pdb') or str(protein_file).lower().endswith('.cif')):
            try:
                # Save molecule to temporary SDF file for GNINA
                with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as tmp_lig:
                    tmp_lig_path = Path(tmp_lig.name)
                    from omtra.eval.system import write_mols_to_sdf
                    write_mols_to_sdf([mol_to_preserve], str(tmp_lig_path))
                
                try:
                    gnina_scores = _run_gnina_score_only(
                        lig_file=str(tmp_lig_path),
                        prot_file=str(protein_file),
                        env=None
                    )
                    
                    if gnina_scores:
                        metrics['vina_score'] = gnina_scores.get('minimizedAffinity')
                    else:
                        _logging.warning(f"GNINA scoring returned no scores for {sample_name}")
                finally:
                    # Clean up temporary ligand file
                    try:
                        tmp_lig_path.unlink(missing_ok=True)
                    except Exception:
                        pass
            except Exception as gnina_e:
                logging.warning(f"Failed to compute GNINA scores for {sample_name}: {gnina_e}")
                # Leave gnina scores as None
    except Exception as e:
        logging.error(f"Error in compute_fast_molecule_metrics for {sample_name}: {e}")
        # Still return metrics even if there's an error
    
    return metrics


def load_omtra_model(checkpoint_path: str, sampling_mode: str):
    """
    Load the OMTRA model from checkpoint. Returns None if model is not available
    """

    try:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        # Add OMTRA modules to path
        # Add OMTRA modules to path
        # Try /omtra first (Docker), then relative path (Local)
        omtra_root = Path('/omtra')
        if not omtra_root.exists():
             omtra_root = Path(__file__).resolve().parent.parent.parent
        
        if str(omtra_root) not in sys.path:
            sys.path.insert(0, str(omtra_root))
        
        # Import OMTRA modules
        from omtra.load.quick import omtra_from_checkpoint
        import torch
        
        # Load model from checkpoint
        logger.info(f"Loading OMTRA model from {checkpoint_path}")
        model = omtra_from_checkpoint(str(checkpoint_path))
        
        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).eval()
        
        logger.info(f"Successfully loaded OMTRA model for {sampling_mode} mode on {device}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load OMTRA model: {e}")
        return None


def run_omtra_sampler(
    job_id: str, 
    params: Dict[str, Any] = None, 
    input_files: List[Dict[str, Any]] = None, 
    model = None, 
    job_logger: logging.Logger = None,
    args = None,
    outputs_dir = None,
    sampling_mode: str = None
) -> Dict[str, Any]:
    """
    Run OMTRA sampling using the CLI approach
    """
    if job_logger:
        job_logger.info(f"run_omtra_sampler called with job_id={job_id}, sampling_mode={sampling_mode}")
        job_logger.info(f"params keys: {list(params.keys()) if params else 'None'}")
        job_logger.info(f"input_files count: {len(input_files) if input_files else 0}")
    

    if args is not None:
        # Pre-configured args provided (e.g. from docking task)
        task_name = args.task
        chosen_prot_for_metrics = str(args.protein_file) if getattr(args, 'protein_file', None) else None
        n_samples = args.n_samples
    else:
        # Standard flow from params
        job_dir = get_job_directory(job_id)
        outputs_dir = job_dir / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        
        # Get parameters
        sampling_mode = params.get('sampling_mode')
        n_samples = params.get('n_samples', 10)
        n_timesteps = params.get('steps', 100)
        batch_size = params.get('batch_size', 500)
        checkpoint_path = params.get('checkpoint_path')
        seed = params.get('seed')
        n_lig_atoms_mean = params.get('n_lig_atoms_mean')
        n_lig_atoms_std = params.get('n_lig_atoms_std')
        pocket_selection = params.get('pocket_selection')
        
        # Set random seed for reproducibility
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        
        job_logger.info(f"Starting OMTRA sampling: {sampling_mode} mode, {n_samples} samples, {n_timesteps} timesteps")
        
        # Map sampling mode to OMTRA task
        task_mapping = {
            'Unconditional': 'denovo_ligand_condensed',
            'Pharmacophore-conditioned': 'denovo_ligand_from_pharmacophore_condensed',
            'Protein-conditioned': 'fixed_protein_ligand_denovo_condensed',
            'Protein+Pharmacophore-conditioned': 'fixed_protein_pharmacophore_ligand_denovo_condensed'
        }
        if sampling_mode not in task_mapping:
            raise ValueError(
                f"Unknown sampling_mode {sampling_mode!r}. "
                f"Expected one of: {', '.join(sorted(task_mapping))}"
            )
        
        partial_mapping = {
            'fixed_protein_ligand_denovo_condensed': 'partial_fixed_protein_ligand_denovo_condensed',
            'rigid_docking_condensed': 'partial_rigid_docking_condensed',
        }
        task_name, fixed_atoms_str, fixed_brics_str, has_fixed = _resolve_fixed_structure_for_task(
            params,
            task_mapping[sampling_mode],
            partial_task_mapping=partial_mapping,
        )
        if has_fixed and job_logger:
            job_logger.info(
                f"Partial fixed-structure sampling: task={task_name}, "
                f"fixed_atoms={fixed_atoms_str}, fixed_brics={fixed_brics_str}"
            )
            partial_ckpt = CHECKPOINT_DIR / 'gnn_partial_prot_cond.ckpt'
            if partial_ckpt.exists():
                checkpoint_path = str(partial_ckpt)
            elif job_logger:
                job_logger.warning(
                    f"Partial checkpoint missing at {partial_ckpt}; using {checkpoint_path}"
                )
        
        # Create a mock args object like the CLI does
        class MockArgs:
            def __init__(self):
                self.checkpoint = Path(checkpoint_path)
                self.task = task_name
                self.dataset = "pharmit"
                self.n_samples = n_samples
                self.n_replicates = 1
                self.dataset_start_idx = 0
                self.n_timesteps = n_timesteps
                self.batch_size = batch_size
                self.visualize = False
                self.output_dir = outputs_dir
                self.pharmit_path = None
                self.plinder_path = None
                self.split = 'val'
                self.stochastic_sampling = False
                self.noise_scaler = 1.0
                self.eps = 0.01
                self.use_gt_n_lig_atoms = False
                self.n_lig_atom_margin = 0.15
                self.n_lig_atoms_mean = n_lig_atoms_mean
                self.n_lig_atoms_mean = n_lig_atoms_mean
                self.n_lig_atoms_std = n_lig_atoms_std
                self.pharmacophore_tolerance = params.get('pharmacophore_tolerance', 0.0)
                self.metrics = False
                self.protein_file = None
                self.ligand_file = None
                self.pharmacophore_file = None
                self.pocket_ligand = None
                self.pocket_center = None
                self.pocket_residues = None
                self.pocket_coords = None
                self.bbox_length = 23.0
                self.input_files_dir = None
                self.g_list_from_files = None
                self.fixed_atoms = fixed_atoms_str
                self.fixed_brics_fragments = fixed_brics_str
        
        # Set up args based on sampling mode
        args = MockArgs()
        
        # Initialize variables that might be used later
        # These need to be None if we are in docking mode (where args is pre-filled and we skip this block)
        # But wait, this is the else block (for non-pre-filled args).
        # We need to initialize them OUTSIDE this block or handle them correctly.
        # Actually, let's just ensure they are initialized if not set.
        pass
        # Store pocket ligand filename for later resolution (after file mapping is created)
        pocket_ligand_filename = None
        if pocket_selection:
            pocket_type = pocket_selection.get('type')
            if pocket_type == 'center':
                center = pocket_selection.get('value')
                if isinstance(center, list) and len(center) == 3:
                    args.pocket_center = f"{center[0]},{center[1]},{center[2]}"
                    args.bbox_length = pocket_selection.get('bbox_length', 23.0)
            elif pocket_type == 'residues':
                residues = pocket_selection.get('value', [])
                if residues:
                    res_strs = [_format_pocket_residue(r) for r in residues]
                    args.pocket_residues = ','.join(res_strs)
            elif pocket_type == 'coords':
                args.pocket_coords = _save_pocket_reference_coords(
                    pocket_selection.get('value', []),
                    outputs_dir,
                )
            elif pocket_type == 'file':
                pocket_ligand_filename = pocket_selection.get('value') # Store for later resolution
        
        # Handle input files for conditional sampling
        if sampling_mode != 'Unconditional' and input_files:
            # Create temporary directory for input files
            temp_input_dir = outputs_dir / "temp_inputs"
            temp_input_dir.mkdir(exist_ok=True)
            
            # Copy input files to temp directory and handle pharmacophore extraction from SDF
            pharmacophore_file = None
            sdf_files = []
            xyz_files = []
            filename_to_tempfile = {}  # Map original filename to temp file path
            
            for i, file_info in enumerate(input_files):
                file_path = file_info['path']
                filename = file_info['filename']
                temp_file = temp_input_dir / f"input_{i:03d}{Path(file_path).suffix}"
                shutil.copy2(file_path, temp_file)
                
                # Store mapping from original filename to temp file path
                filename_to_tempfile[filename] = temp_file
                
                # Categorize files
                if filename.endswith('.xyz'):
                    xyz_files.append(temp_file)
                elif filename.endswith('.sdf'):
                    sdf_files.append(temp_file)
            
            # Resolve pocket ligand filename to temp file path
            if pocket_ligand_filename and pocket_ligand_filename in filename_to_tempfile:
                args.pocket_ligand = str(filename_to_tempfile[pocket_ligand_filename])
                job_logger.info(f"Resolved pocket ligand '{pocket_ligand_filename}' to '{args.pocket_ligand}'")
            elif pocket_ligand_filename:
                job_logger.warning(f"Pocket ligand file '{pocket_ligand_filename}' not found in input files")
            
            # For pharmacophore-conditioned sampling, prioritize XYZ files
            # If no XYZ file but SDF files exist, extract pharmacophores from SDF
            if sampling_mode in ['Pharmacophore-conditioned', 'Protein+Pharmacophore-conditioned', 'Rigid Docking + Pharmacophore']:
                if xyz_files:
                    # Use existing XYZ file
                    pharmacophore_file = xyz_files[0]
                    
                    # Verify XYZ file content (read first line to get feature count)
                    try:
                        with open(pharmacophore_file, 'r') as f:
                            first_line = f.readline().strip()
                            feature_count = int(first_line) if first_line.isdigit() else None
                            if feature_count is None:
                                job_logger.warning(f"Could not determine feature count from XYZ file")
                    except Exception as e:
                        job_logger.warning(f"Could not read XYZ file to verify feature count: {e}")
                elif sdf_files:
                    # Extract pharmacophores from first SDF file
                    try:
                        # Import from shared module
                        from shared.file_utils import extract_pharmacophore_from_sdf, pharmacophore_list_to_xyz
                        
                        # Read SDF file
                        with open(sdf_files[0], 'rb') as f:
                            sdf_content = f.read()
                        
                        # Extract pharmacophores
                        result = extract_pharmacophore_from_sdf(sdf_content)
                        pharmacophores = result.get('pharmacophores', [])
                        
                        if not pharmacophores:
                            raise ValueError("No pharmacophore features extracted from SDF file")
                        
                        center_coords = True
                        xyz_content = pharmacophore_list_to_xyz(pharmacophores, selected_indices=None, center=center_coords)
                        
                        # Save to temporary XYZ file
                        pharmacophore_file = temp_input_dir / "pharmacophore.xyz"
                        with open(pharmacophore_file, 'w') as f:
                            f.write(xyz_content)
                        
                    except Exception as e:
                        job_logger.error(f"Failed to extract pharmacophores from SDF: {e}", exc_info=True)
                        raise ValueError(f"Failed to extract pharmacophores from SDF file: {str(e)}")
                else:
                    raise ValueError("No pharmacophore files (XYZ) or ligand files (SDF) provided for pharmacophore-conditioned sampling")
            
            # Initialize protein file for metrics (used later when splitting SDF)
            chosen_prot_for_metrics = None
            if sampling_mode in ['Pharmacophore-conditioned', 'Protein+Pharmacophore-conditioned', 'Rigid Docking + Pharmacophore'] and pharmacophore_file:
                args.pharmacophore_file = pharmacophore_file
            
            if sampling_mode in PROTEIN_INVOLVING_MODES:
                # Find first protein file (PDB preferred) from temp_input_dir
                pdbs = list(temp_input_dir.glob('*.pdb'))
                cifs = list(temp_input_dir.glob('*.cif'))
                chosen_prot = pdbs[0] if pdbs else (cifs[0] if cifs else None)
                if chosen_prot is None:
                    raise ValueError(f"No protein file (.pdb/.cif) found for {sampling_mode}")
                args.protein_file = chosen_prot
                
            # Handle pocket ligand file path resolving for sampling
            if args.pocket_ligand and not os.path.isabs(str(args.pocket_ligand)):
                # Look in temp_input_dir for the filename
                pocket_ligand_path = temp_input_dir / str(args.pocket_ligand)
                if pocket_ligand_path.exists():
                    args.pocket_ligand = pocket_ligand_path
                    job_logger.info(f"Using pocket ligand from {args.pocket_ligand}")
                else:
                    job_logger.warning(f"Pocket ligand file {args.pocket_ligand} not found in temp_input_dir")

            # Final validation check before CLI call
            # Guard this block: only run if temp_input_dir is defined (i.e. we are NOT in pre-configured docking mode)
            if 'temp_input_dir' in locals() and sampling_mode in PROTEIN_INVOLVING_MODES:
                sdf_files = list(temp_input_dir.glob('*.sdf'))
                has_pocket = bool(getattr(args, 'pocket_center', None) or getattr(args, 'pocket_residues', None) or getattr(args, 'pocket_coords', None) or (getattr(args, 'pocket_ligand', None) and os.path.exists(str(args.pocket_ligand))))
                
                if sampling_mode in ['Protein-conditioned', 'Rigid Docking']:
                    if sdf_files:
                        args.ligand_file = sdf_files[0]
                    elif not has_pocket:
                        raise ValueError(f"No reference ligand file (.sdf) found for {sampling_mode}. A ligand file or pocket selection is required to identify the binding pocket.")
                
                elif sampling_mode in ['Protein+Pharmacophore-conditioned', 'Rigid Docking + Pharmacophore']:
                    if sdf_files:
                        args.ligand_file = sdf_files[0]
                    
                    if not has_pocket and not getattr(args, 'ligand_file', None):
                         raise ValueError(f"{sampling_mode} requires a pocket definition (center/residues) or a reference ligand file.")
                    
                    if not getattr(args, 'pharmacophore_file', None):
                         raise ValueError(f"{sampling_mode} requires a pharmacophore file (.xyz or .sdf).")

                args.n_samples = n_samples
                chosen_prot_for_metrics = str(chosen_prot)
            elif 'temp_input_dir' in locals():
                args.input_files_dir = temp_input_dir
            
            if sampling_mode not in ['Protein-conditioned', 'Rigid Docking']:
                args.n_samples = n_samples
                args.n_replicates = 1  # 1 replicate per sample
            
        else:
            args = MockArgs()
            # Initialize protein file for metrics (used later when splitting SDF)
            chosen_prot_for_metrics = None
        
    # Import and call the CLI's run_sample function
    from cli import run_sample
    run_sample(args)

    # Save template ligand coordinates for fixed-atom visualization in the web UI
    has_fixed_structure = bool(
        getattr(args, 'fixed_atoms', None) or getattr(args, 'fixed_brics_fragments', None)
    )
    lig_path = getattr(args, 'ligand_file', None)
    if has_fixed_structure and lig_path and outputs_dir and Path(str(lig_path)).exists():
        try:
            shutil.copy2(str(lig_path), outputs_dir / 'fixed_structure_reference.sdf')
            if job_logger:
                job_logger.info('Saved fixed_structure_reference.sdf for results viewer')
        except Exception as e:
            if job_logger:
                job_logger.warning(f'Could not save fixed_structure_reference.sdf: {e}')
    
    # Split the combined SDF file into individual sample files
    molecules = []
    all_molecule_metrics = []
    
    try:
        from rdkit import Chem
        from omtra.eval.system import write_mols_to_sdf
        
        # Find the generated SDF file - check sys_0_gt directory first
        sdf_files = []
        sys_gt_dir = outputs_dir / "sys_0_gt"
        if sys_gt_dir.exists():
            sdf_files = list(sys_gt_dir.glob("gen_ligands.sdf"))
        
        # Fallback to looking in outputs directory
        if not sdf_files:
            sdf_files = list(outputs_dir.glob("*_lig.sdf"))
        
        if not sdf_files:
            job_logger.error("No SDF file found after CLI sampling")
            raise Exception("No SDF file generated")
        
        combined_sdf = sdf_files[0]
        
        # Read all molecules from the combined file
        # Don't sanitize on read to see raw parsing issues, we'll sanitize manually
        supplier = Chem.SDMolSupplier(str(combined_sdf), sanitize=False)
        molecules = []
        failed_indices = []
        failed_reasons = []
        
        for idx, mol in enumerate(supplier):
            if mol is not None:
                # Try to sanitize to check if it's actually valid
                try:
                    Chem.SanitizeMol(mol)
                    molecules.append(mol)
                except Exception as sanitize_err:
                    # Molecule parsed but failed sanitization - still include it but mark as invalid
                    failed_indices.append(idx)
                    failed_reasons.append(f"Sanitization failed: {str(sanitize_err)}")
                    job_logger.warning(f"Molecule at index {idx} failed sanitization: {sanitize_err}")
                    # Mark molecule as invalid and include it anyway
                    try:
                        mol.SetProp("_Invalid", str(sanitize_err))
                    except Exception:
                        pass
                    molecules.append(mol)
            else:
                # Get the last error from the supplier if available
                error_msg = supplier.GetLastErrorText() if hasattr(supplier, 'GetLastErrorText') else "Unknown parsing error"
                failed_indices.append(idx)
                failed_reasons.append(f"Parse failed: {error_msg}")
                job_logger.warning(f"Molecule at index {idx} in SDF file failed to parse: {error_msg}")
        
        # Check if we got fewer molecules than expected
        expected_samples = n_samples
        if len(molecules) < expected_samples:
            job_logger.warning(
                f"Expected {expected_samples} molecules but only found {len(molecules)} molecules in SDF (some may be invalid). "
                f"Failed molecule indices: {failed_indices}"
            )
            for idx, reason in zip(failed_indices, failed_reasons):
                job_logger.warning(f"  Index {idx}: {reason}")
        
        job_logger.info(f"Found {len(molecules)} molecules (including {len(failed_indices)} invalid) out of {expected_samples} expected in combined SDF")
        
        # Find protein file from inputs for any task (pb_valid/metrics can use it)
        protein_file = None
        if chosen_prot_for_metrics:
            protein_file = chosen_prot_for_metrics
        elif input_files:
            # Prefer PDB over CIF for PoseBusters stability
            pdb_path = None
            cif_path = None
            for file_info in input_files:
                filename = file_info['filename']
                lower_name = filename.lower()
                if lower_name.endswith('.pdb') and pdb_path is None:
                    pdb_path = file_info['path']
                elif lower_name.endswith('.cif') and cif_path is None:
                    cif_path = file_info['path']
            protein_file = pdb_path or cif_path

        # If only CIF is available, try converting to temporary PDB for PoseBusters/PoseCheck
        if protein_file and str(protein_file).lower().endswith('.cif'):
            try:
                # Use biotite like the rest of OMTRA codebase (see omtra/utils/file_to_graph.py)
                from biotite.structure.io import pdb
                from biotite.structure.io.pdbx import CIFFile, get_structure
                
                # Read CIF file using biotite (same method as OMTRA uses elsewhere)
                cif_file = CIFFile.read(str(protein_file))
                st = get_structure(cif_file, model=1, include_bonds=False)
                
                # Remove waters and hydrogens 
                st = st[st.res_name != "HOH"]
                st = st[st.element != "H"]
                st = st[st.element != "D"]
                
                total_atoms = len(st)
                
                if total_atoms == 0:
                    job_logger.warning(f"CIF file has no atoms after filtering, cannot convert to PDB")
                    # Keep original CIF file
                else:
                    # Write temporary PDB using biotite
                    tmp_pdb = outputs_dir / "protein_from_cif.pdb"
                    pdb_file = pdb.PDBFile()
                    pdb_file.set_structure(st)
                    pdb_file.write(str(tmp_pdb))
                    
                    # Validate the converted PDB has ATOM records
                    with open(tmp_pdb, 'r') as fh:
                        pdb_content = fh.read()
                    
                    has_atoms = 'ATOM' in pdb_content or 'HETATM' in pdb_content
                    atom_count = pdb_content.count('\nATOM') + pdb_content.count('\nHETATM')
                    if has_atoms and atom_count > 0:
                        protein_file = str(tmp_pdb)
                    else:
                        job_logger.warning(f"CIF→PDB conversion produced empty PDB (no ATOM records), keeping original CIF file")
                        # Don't use the empty PDB, keep original CIF
            except ImportError:
                job_logger.warning("biotite not available, cannot convert CIF to PDB")
            except Exception as conv_e:
                job_logger.warning(f"CIF→PDB conversion failed: {conv_e}", exc_info=True)
        
        # Queue ligands for serial background diagram generation (after job completes)
        diagram_queue: List[Path] = []
        metrics_opts = _resolve_metrics_options(params)

        # Pre-load PoseCheck protein once for all samples
        posecheck_obj = None
        if (
            sampling_mode in PROTEIN_INVOLVING_MODES
            and protein_file
            and str(protein_file).lower().endswith('.pdb')
            and (metrics_opts.get('posecheck') or metrics_opts.get('strain'))
        ):
            try:
                posecheck_obj = _load_posecheck_with_timeout(str(protein_file), timeout=120)
                if posecheck_obj is None:
                    job_logger.warning("PoseCheck protein loading timed out or failed; interaction metrics will be skipped")
                else:
                    job_logger.info("PoseCheck protein loaded successfully")
            except Exception as pc_load_e:
                job_logger.warning(f"Failed to pre-load PoseCheck protein: {pc_load_e}; interaction metrics will be skipped")

        # Save individual sample files and compute metrics
        for i, mol in enumerate(molecules):
            individual_file = outputs_dir / f"sample_{i + 1:02d}.sdf"
            write_mols_to_sdf([mol], str(individual_file))
            
            # Check if molecule is invalid (marked with _Invalid property)
            is_invalid = False
            invalid_reason = None
            try:
                if mol.HasProp("_Invalid"):
                    is_invalid = True
                    invalid_reason = mol.GetProp("_Invalid")
            except Exception:
                pass
            
            if is_invalid:
                job_logger.warning(f"Saved sample {i+1} (INVALID - {invalid_reason}): {individual_file}")
            
            # Compute fast metrics for this molecule
            sample_name = f"sample_{i + 1:02d}"
            metrics = compute_fast_molecule_metrics(
                mol,
                sample_name=sample_name,
                sampling_mode=sampling_mode,
                protein_file=protein_file,
                posecheck_obj=posecheck_obj,
                metrics_options=metrics_opts,
            )
            
            # Mark invalid molecules in metrics with Warning
            if is_invalid:
                metrics['Warning'] = invalid_reason
            
            all_molecule_metrics.append(metrics)
            
            if metrics_opts.get('poseview') and sampling_mode in PROTEIN_INVOLVING_MODES and protein_file:
                diagram_queue.append(individual_file)
        
        # Include reference/pocket ligand as the first entry if available
        pocket_ligand_path = getattr(args, 'pocket_ligand', None)
        if pocket_ligand_path and Path(str(pocket_ligand_path)).exists() and sampling_mode in PROTEIN_INVOLVING_MODES:
            try:
                ref_mol = None
                supplier = Chem.SDMolSupplier(str(pocket_ligand_path), removeHs=True, sanitize=False)
                for m in supplier:
                    if m is not None:
                        m.UpdatePropertyCache(strict=False)
                        Chem.GetSymmSSSR(m)
                        ref_mol = m
                        break

                if ref_mol is not None:
                    ref_sdf = outputs_dir / "reference_ligand.sdf"
                    write_mols_to_sdf([ref_mol], str(ref_sdf))

                    ref_metrics = compute_fast_molecule_metrics(
                        ref_mol,
                        sample_name="reference_ligand",
                        sampling_mode=sampling_mode,
                        protein_file=protein_file,
                        posecheck_obj=posecheck_obj,
                        metrics_options=metrics_opts,
                    )
                    ref_metrics['is_reference'] = True
                    all_molecule_metrics.insert(0, ref_metrics)

                    if metrics_opts.get('poseview') and sampling_mode in PROTEIN_INVOLVING_MODES and protein_file:
                        diagram_queue.insert(0, ref_sdf)

                    job_logger.info(f"Added reference ligand from {pocket_ligand_path} with metrics")
                else:
                    job_logger.warning(f"Could not parse reference ligand from {pocket_ligand_path}")
            except Exception as ref_e:
                job_logger.warning(f"Failed to add reference ligand: {ref_e}")

        if metrics_opts.get('poseview') and diagram_queue and protein_file:
            _enqueue_diagram_generation(
                job_id,
                outputs_dir,
                diagram_queue,
                Path(protein_file),
                job_logger,
            )

        # Save per-molecule metrics to JSON file
        if all_molecule_metrics:
            metrics_file = outputs_dir / "per_molecule_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(all_molecule_metrics, f, indent=2)
        
        # Remove the combined file created by CLI to keep only individual files
        if combined_sdf.exists():
            combined_sdf.unlink()
        
    except Exception as e:
        job_logger.error(f"Failed to split SDF files: {e}")
        job_logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Generate summary
    num_molecules = len(molecules)
    summary = {
        'samples_generated': num_molecules,
        'input_files': len(input_files) if input_files else 0,
        'parameters_used': params if params else {},
        'task_name': task_name,
        'sampling_mode': sampling_mode,
        'processing_time_seconds': 0
    }
    
    # Save summary as JSON
    summary_file = outputs_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    job_logger.info("OMTRA sampling completed successfully")
    return summary
        



def main():
    """Main worker function"""
    logger.info("Starting OMTRA worker")
    
    # Connect to Redis
    redis_conn = redis.from_url(REDIS_URL)
    
    # Create queue
    queue = Queue('omtra_tasks', connection=redis_conn)
    
    # Start worker with unique name
    worker_name = f"omtra-worker-{os.getpid()}-{int(time.time())}"
    worker = Worker([queue], name=worker_name)
    logger.info(f"Worker {worker.name} started")
    worker.work()


if __name__ == '__main__':
    main()
