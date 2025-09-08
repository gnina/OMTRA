import os
import sys
import time
import json
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
import redis
from rq import Worker, Queue, Connection
import logging

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
    job_dir = get_job_directory(job_id)
    log_file = job_dir / "logs" / "job.log"
    log_file.parent.mkdir(exist_ok=True)
    
    # Setup job-specific logging
    job_logger = setup_logging(
        level='INFO',
        log_file=log_file
    )
    
    try:
        log_job_event(job_logger, job_id, "sampling_started", params=params)
        
        # Load the molecular generation model
        model = load_omtra_model()
        
        if model is None:
            # Use stub sampler for testing
            result = run_stub_sampler(job_id, params, input_files, job_logger)
        else:
            # Use real model
            result = run_omtra_sampler(job_id, params, input_files, model, job_logger)
        
        log_job_event(job_logger, job_id, "sampling_completed", **result)
        return result
        
    except Exception as e:
        error_msg = f"Sampling failed: {str(e)}"
        job_logger.error(error_msg, exc_info=True)
        log_job_event(job_logger, job_id, "sampling_failed", error=error_msg)
        
        # Write error to log file
        with open(log_file, 'a') as f:
            f.write(f"\nERROR: {error_msg}\n")
            f.write(traceback.format_exc())
        
        raise


def load_omtra_model():
    """
    Load the OMTRA model. Returns None if model is not available (use stub instead)
    """
    try:
        model_path = os.getenv('MODEL_CHECKPOINT_PATH', '/srv/app/models/checkpoint.pth')
        config_path = os.getenv('MODEL_CONFIG_PATH', '/srv/app/models/config.yaml')
        
        if not Path(model_path).exists() or not Path(config_path).exists():
            logger.warning("Model files not found, using stub sampler")
            return None
        
        # Import OMTRA modules (adjust paths as needed)
        sys.path.append('/app/../..')  # Add parent OMTRA directory
        
        # This would load your actual OMTRA model
        # from omtra.models import load_model
        # from omtra.utils import load_config
        # 
        # config = load_config(config_path)
        # model = load_model(model_path, config)
        # return model
        
        logger.warning("Real model loading not implemented yet, using stub sampler")
        return None
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def run_omtra_sampler(
    job_id: str, 
    params: Dict[str, Any], 
    input_files: List[Dict[str, Any]], 
    model, 
    job_logger: logging.Logger
) -> Dict[str, Any]:
    """
    Run the actual OMTRA sampling
    """
    job_dir = get_job_directory(job_id)
    outputs_dir = job_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    job_logger.info("Starting OMTRA sampling with real model")
    
    # TODO: Implement actual OMTRA sampling logic
    # This would involve:
    # 1. Loading input structures
    # 2. Setting up conditioning/guidance
    # 3. Running the diffusion sampling
    # 4. Saving output structures
    
    # For now, fall back to stub
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
    
    # Start worker
    with Connection(redis_conn):
        worker = Worker([queue], name=f"omtra-worker-{os.getpid()}")
        logger.info(f"Worker {worker.name} started")
        worker.work()


if __name__ == '__main__':
    main()
