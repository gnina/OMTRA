import os
import sys
import time
import json
import traceback
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import redis
from rq import Worker, Queue
import logging
import torch

# Add shared module to path
sys.path.append(str(Path(__file__).parent.parent))

from shared.logging_utils import setup_logging, log_job_event
from shared.file_utils import get_job_directory, create_job_directory

# Configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
WORKER_TIMEOUT = int(os.getenv('WORKER_TIMEOUT', 180))

# Setup logging
logger = setup_logging(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    structured=os.getenv('STRUCTURED_LOGGING', 'true').lower() == 'true'
)


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
        sampling_mode = params.get('sampling_mode', 'Unconditional')
        checkpoint_path = params.get('checkpoint_path')
        
        if not checkpoint_path:
            job_logger.warning("No checkpoint path provided, using stub sampler")
            result = run_stub_sampler(job_id, params, input_files, job_logger)
        else:
            model = load_omtra_model(checkpoint_path, sampling_mode)
            
            if model is None:
                result = run_stub_sampler(job_id, params, input_files, job_logger)
            else:
                result = run_omtra_sampler(job_id, params, input_files, model, job_logger)
        
        log_job_event(job_logger, job_id, "sampling_completed", **result)
        
        # Update job status in Redis
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
                job_logger.info(f"Updated job status to SUCCEEDED in Redis")
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
                job_logger.info(f"Updated job status to FAILED in Redis")
        except Exception as redis_e:
            job_logger.warning(f"Failed to update job status in Redis: {redis_e}")
        
        # Write error to log file
        with open(log_file, 'a') as f:
            f.write(f"\nERROR: {error_msg}\n")
            f.write(traceback.format_exc())
        
        raise


def load_omtra_model(checkpoint_path: str, sampling_mode: str):
    """
    Load the OMTRA model from checkpoint. Returns None if model is not available (use stub instead)
    """
    try:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}, using stub sampler")
            return None
        
        # Add OMTRA modules to path
        omtra_root = Path('/omtra')
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
        logger.warning("Falling back to stub sampler")
        return None


def run_omtra_sampler(
    job_id: str, 
    params: Dict[str, Any], 
    input_files: List[Dict[str, Any]], 
    model, 
    job_logger: logging.Logger
) -> Dict[str, Any]:
    """
    Run OMTRA sampling using the CLI approach
    """
    job_dir = get_job_directory(job_id)
    outputs_dir = job_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    try:
        # Get parameters
        sampling_mode = params.get('sampling_mode', 'Unconditional')
        n_samples = params.get('n_samples', 10)
        n_timesteps = params.get('steps', 100)
        device = params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint_path = params.get('checkpoint_path')
        seed = params.get('seed')
        
        # Set random seed for reproducibility
        if seed is not None:
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            job_logger.info(f"Set random seed to {seed} for reproducibility")
        
        job_logger.info(f"Starting OMTRA sampling using CLI approach")
        job_logger.info(f"Device: {device}, Samples: {n_samples}, Timesteps: {n_timesteps}")
        
        # Map sampling mode to OMTRA task
        task_mapping = {
            'Unconditional': 'denovo_ligand_condensed',
            'Pharmacophore-conditioned': 'denovo_ligand_from_pharmacophore_condensed',
            'Protein-conditioned': 'denovo_ligand_condensed'
        }
        
        task_name = task_mapping.get(sampling_mode, 'denovo_ligand_condensed')
        
        # Create a mock args object like the CLI does
        class MockArgs:
            def __init__(self):
                self.checkpoint = Path(checkpoint_path)
                self.task = task_name
                self.dataset = "pharmit"  # Default dataset
                self.n_samples = n_samples
                self.n_replicates = 1
                self.dataset_start_idx = 0
                self.n_timesteps = n_timesteps
                self.visualize = False
                self.output_dir = outputs_dir
                self.pharmit_path = None
                self.plinder_path = None
                self.split = 'val'
                self.stochastic_sampling = False
                self.noise_scaler = 1.0
                self.eps = 0.01
                self.use_gt_n_lig_atoms = False
                self.n_lig_atom_margin = 15
                self.metrics = False
                self.protein_file = None
                self.ligand_file = None
                self.pharmacophore_file = None
                self.input_files_dir = None
                self.g_list_from_files = None
        
        # Handle input files for conditional sampling
        if sampling_mode != 'Unconditional' and input_files:
            job_logger.info(f"Creating conditional graphs from {len(input_files)} input files")
            
            # Create temporary directory for input files
            temp_input_dir = outputs_dir / "temp_inputs"
            temp_input_dir.mkdir(exist_ok=True)
            
            # Copy input files to temp directory
            pharmacophore_file = None
            for i, file_info in enumerate(input_files):
                file_path = file_info['path']
                temp_file = temp_input_dir / f"input_{i:03d}{Path(file_path).suffix}"
                shutil.copy2(file_path, temp_file)
                
                # Check if this is a pharmacophore file
                if file_info['filename'].endswith('.xyz'):
                    pharmacophore_file = temp_file
            
            # Set up args based on sampling mode
            args = MockArgs()
            if sampling_mode == 'Pharmacophore-conditioned' and pharmacophore_file:
                args.pharmacophore_file = pharmacophore_file
                job_logger.info(f"Using pharmacophore file: {pharmacophore_file}")
            else:
                args.input_files_dir = temp_input_dir
            
            # For pharmacophore-conditioned sampling, we want n_samples from the same pharmacophore
            args.n_samples = n_samples
            args.n_replicates = 1  # 1 replicate per sample
        else:
            args = MockArgs()
        
        # Import and call the CLI's run_sample function
        from cli import run_sample
        
        # Call the CLI function directly
        job_logger.info("Calling CLI run_sample function")
        run_sample(args)
        
        # Split the combined SDF file into individual sample files
        job_logger.info("Splitting combined SDF into individual sample files")
        try:
            from rdkit import Chem
            from omtra.eval.system import write_mols_to_sdf
            
            # Find the generated SDF file - check sys_0_gt directory first
            sdf_files = []
            sys_gt_dir = outputs_dir / "sys_0_gt"
            if sys_gt_dir.exists():
                sdf_files = list(sys_gt_dir.glob("gen_ligands.sdf"))
                if sdf_files:
                    job_logger.info(f"Found OMTRA generated SDF in sys_0_gt: {sdf_files[0]}")
            
            # Fallback to looking in outputs directory
            if not sdf_files:
                sdf_files = list(outputs_dir.glob("*_lig.sdf"))
                if sdf_files:
                    job_logger.info(f"Found SDF file in outputs: {sdf_files[0]}")
            
            if not sdf_files:
                job_logger.error("No SDF file found after CLI sampling")
                raise Exception("No SDF file generated")
            
            combined_sdf = sdf_files[0]
            job_logger.info(f"Using SDF file: {combined_sdf}")
            
            # Read all molecules from the combined file
            supplier = Chem.SDMolSupplier(str(combined_sdf))
            molecules = [mol for mol in supplier if mol is not None]
            
            job_logger.info(f"Found {len(molecules)} molecules in combined SDF")
            
            # Save individual sample files
            for i, mol in enumerate(molecules):
                individual_file = outputs_dir / f"sample_{i:03d}.sdf"
                write_mols_to_sdf([mol], str(individual_file))
                job_logger.info(f"Saved sample {i+1}: {individual_file}")
            
            # Remove the combined file created by CLI to keep only individual files
            if combined_sdf.exists():
                combined_sdf.unlink()
                job_logger.info(f"Removed combined file: {combined_sdf}")
            
        except Exception as e:
            job_logger.error(f"Failed to split SDF files: {e}")
            job_logger.info(f"Error details: {str(e)}")
            import traceback
            job_logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Generate summary
        summary = {
            'samples_generated': len(molecules),  # Use actual number of molecules found
            'input_files': len(input_files),
            'parameters_used': params,
            'task_name': task_name,
            'sampling_mode': sampling_mode,
            'processing_time_seconds': 0
        }
        
        # Save summary as JSON
        summary_file = outputs_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        job_logger.info("OMTRA sampling completed successfully using CLI approach")
        return summary
        
    except Exception as e:
        job_logger.error(f"OMTRA sampling failed: {e}")
        job_logger.info(f"Error details: {str(e)}")
        import traceback
        job_logger.error(f"Traceback: {traceback.format_exc()}")
        job_logger.warning("Falling back to stub sampler")
        return run_stub_sampler(job_id, params, input_files, job_logger)


def run_stub_sampler(
    job_id: str, 
    params: Dict[str, Any], 
    input_files: List[Dict[str, Any]], 
    job_logger: logging.Logger
) -> Dict[str, Any]:
    """
    Stub sampler for testing - generates dummy molecular structures
    """
    job_dir = get_job_directory(job_id)
    outputs_dir = job_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    job_logger.info("Running stub sampler for testing")
    
    # Simulate processing time
    n_samples = params.get('n_samples', 10)
    for i in range(n_samples):
        time.sleep(0.1)  # Simulate work
        
        # Generate dummy SDF content
        sdf_content = generate_dummy_sdf_molecule(i, params.get('seed', 42))
        
        # Save output file
        output_file = outputs_dir / f"sample_{i:03d}.sdf"
        with open(output_file, 'w') as f:
            f.write(sdf_content)
        
        job_logger.info(f"Generated sample {i+1}/{n_samples}")
    
    # Generate summary
    summary = {
        'samples_generated': n_samples,
        'input_files': len(input_files),
        'parameters_used': params,
        'processing_time_seconds': n_samples * 0.1
    }
    
    # Save summary as JSON
    summary_file = outputs_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def generate_dummy_sdf_molecule(index: int, seed: int = 42) -> str:
    """
    Generate a dummy SDF molecule for testing
    """
    import random
    if seed is None:
        seed = 42
    random.seed(seed + index)
    
    # Simple benzene-like structure with random coordinates
    atoms = []
    bonds = []
    
    # Generate 6 carbon atoms in a ring
    for i in range(6):
        angle = i * 60 * 3.14159 / 180
        x = 1.4 * (1 + 0.1 * random.random()) * (1 if random.random() > 0.5 else -1)
        y = 1.4 * (1 + 0.1 * random.random()) * (1 if random.random() > 0.5 else -1) 
        z = 0.1 * random.random()
        
        atoms.append(f"    {x:8.4f}    {y:8.4f}    {z:8.4f} C   0  0  0  0  0  0  0  0  0  0  0  0")
    
    # Add bonds (ring)
    for i in range(6):
        j = (i + 1) % 6
        bond_type = 1 if i % 2 == 0 else 2  # Alternating single/double
        bonds.append(f"  {i+1}  {j+1}  {bond_type}  0  0  0  0")
    
    sdf_content = f"""Sample Molecule {index}
  Generated by OMTRA stub sampler
  
  6  6  0  0  0  0  0  0  0  0999 V2000
""" + "\n".join(atoms) + "\n" + "\n".join(bonds) + """
M  END
$$$$
"""
    
    return sdf_content


def main():
    """Main worker function"""
    logger.info("Starting OMTRA worker")
    
    # Connect to Redis
    redis_conn = redis.from_url(REDIS_URL)
    
    # Create queue
    queue = Queue('omtra_tasks', connection=redis_conn)
    
    # Start worker with unique name
    import time
    worker_name = f"omtra-worker-{os.getpid()}-{int(time.time())}"
    worker = Worker([queue], name=worker_name)
    logger.info(f"Worker {worker.name} started")
    worker.work()


if __name__ == '__main__':
    main()
