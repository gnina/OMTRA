from pathlib import Path
from typing import Optional, Dict
import os

# TODO: change file names before publishing docker images

TASK_TO_CHECKPOINT: Dict[str, str] = {
    # Unconditional tasks -> uncond.ckpt
    "denovo_ligand_condensed": "uncond.ckpt",  # Unconditional de novo ligand generation
    "ligand_conformer_condensed": "uncond.ckpt",  # Unconditional ligand conformer generation
    
    # Pharmacophore-conditioned (no protein) -> phcond.ckpt
    "denovo_ligand_from_pharmacophore_condensed": "phcond.ckpt",  # Pharmacophore-conditioned de novo ligand generation
    "ligand_conformer_from_pharmacophore_condensed": "phcond.ckpt",  # Pharmacophore-conditioned ligand conformer generation
    
    # Protein-conditioned -> protcond.ckpt
    "rigid_docking_condensed": "protcond.ckpt",  # Rigid docking
    "fixed_protein_ligand_denovo_condensed": "protcond.ckpt",  # Rigid protein, de novo ligand generation
    
    # Protein + pharmacophore -> protpharmcond.ckpt
    "rigid_docking_pharmacophore_condensed": "protpharmcond.ckpt",  # Pharmacophore-conditioned rigid docking
    "fixed_protein_pharmacophore_ligand_denovo_condensed": "protpharmcond.ckpt",  # Rigid protein + pharmacophore, de novo ligand generation
}

# Mapping from webapp sampling modes to checkpoint filenames
WEBAPP_TO_CHECKPOINT: Dict[str, str] = {
    "Unconditional": "uncond.ckpt",
    "Pharmacophore-conditioned": "phcond.ckpt",
    "Protein-conditioned": "protcond.ckpt",
    "Protein+Pharmacophore-conditioned": "protpharmcond.ckpt",
}

def get_checkpoint_path_for_task(
    task_name: str, 
    checkpoint_dir: Optional[Path] = None
) -> Optional[Path]:
    """
    Get checkpoint path for a given task name.
    
    Args:
        task_name: CLI task name (e.g., "denovo_ligand_condensed", "rigid_docking_condensed")
        checkpoint_dir: Directory containing checkpoints (defaults to OMTRA_CHECKPOINT_DIR env var or ./checkpoints)
    
    Returns:
        Path to checkpoint file, or None if not found
    """
    checkpoint_filename = TASK_TO_CHECKPOINT.get(task_name)
    if not checkpoint_filename:
        return None
    
    if checkpoint_dir is None:
        checkpoint_dir = Path(os.getenv("OMTRA_CHECKPOINT_DIR", "./checkpoints"))
    
    checkpoint_path = checkpoint_dir / checkpoint_filename
    
    if checkpoint_path.exists():
        return checkpoint_path
    
    return None


def get_checkpoint_path_for_webapp(
    sampling_mode: str, 
    checkpoint_dir: Optional[Path] = None
) -> Optional[Path]:
    """
    Get checkpoint path for webapp sampling mode.
    
    Args:
        sampling_mode: Webapp sampling mode (e.g., "Unconditional", "Protein-conditioned")
        checkpoint_dir: Directory containing checkpoints
    
    Returns:
        Path to checkpoint file, or None if not found
    """
    checkpoint_filename = WEBAPP_TO_CHECKPOINT.get(sampling_mode)
    if not checkpoint_filename:
        return None
    
    if checkpoint_dir is None:
        checkpoint_dir = Path(os.getenv("CHECKPOINT_DIR", "/srv/app/checkpoints"))
    
    checkpoint_path = checkpoint_dir / checkpoint_filename
    if checkpoint_path.exists():
        return checkpoint_path
    
    return None


