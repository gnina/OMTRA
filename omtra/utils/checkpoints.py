from pathlib import Path
from typing import Optional, Dict
import os
from omtra.utils import omtra_root

TASK_TO_CHECKPOINT: Dict[str, str] = {
    # Unconditional tasks -> uncond.ckpt
    "denovo_ligand_condensed": "uncond_r179_dbug_2025-11-03_21-19-252310/checkpoints/last.ckpt",  # Unconditional de novo ligand generation
    "ligand_conformer_condensed": "uncond_r179_dbug_2025-11-03_21-19-252310/checkpoints/last.ckpt",  # Unconditional ligand conformer generation
    
    # Pharmacophore-conditioned (no protein) -> phcond.ckpt
    "denovo_ligand_from_pharmacophore_condensed": "phcond_seeded_r192_debug_2025-11-20_16-15-701316/checkpoints/last.ckpt",  # Pharmacophore-conditioned de novo ligand generation
    "ligand_conformer_from_pharmacophore_condensed": "phcond_seeded_r192_debug_2025-11-20_16-15-701316/checkpoints/last.ckpt",  # Pharmacophore-conditioned ligand conformer generation
    
    # Protein-conditioned -> protcond.ckpt
    "rigid_docking_condensed": "mt_plcd_cr_2025-11-26_10-09-673392/checkpoints/last.ckpt",  # Rigid docking
    "fixed_protein_ligand_denovo_condensed": "mt_plcd_cr_2025-11-26_10-09-673392/checkpoints/last.ckpt",  # Rigid protein, de novo ligand generation
    
    # Protein + pharmacophore -> protpharmcond.ckpt
    "rigid_docking_pharmacophore_condensed": "protpharm_cond_cr_2025-11-24_13-00-028503/checkpoints/last.ckpt",  # Pharmacophore-conditioned rigid docking
    "fixed_protein_pharmacophore_ligand_denovo_condensed": "ph_noise_ckpt_rigid_prot_lig_pharm_3_debug3_2026-01-11_20-28-294287/checkpoints/last.ckpt",  # Rigid protein + pharmacophore, de novo ligand generation
}

# Mapping from webapp sampling modes to checkpoint filenames
WEBAPP_TO_CHECKPOINT: Dict[str, str] = {
    "Unconditional": "uncond_r179_dbug_2025-11-03_21-19-252310/checkpoints/last.ckpt",
    "Pharmacophore-conditioned": "phcond_seeded_r192_debug_2025-11-20_16-15-701316/checkpoints/last.ckpt",
    "Protein-conditioned": "mt_plcd_cr_2025-11-26_10-09-673392/checkpoints/last.ckpt",
    "Protein+Pharmacophore-conditioned": "ph_noise_ckpt_rigid_prot_lig_pharm_3_debug3_2026-01-11_20-28-294287/checkpoints/last.ckpt",
    # Docking modes (use same checkpoints as protein-conditioned modes)
    "Rigid Docking": "mt_plcd_cr_2025-11-26_10-09-673392/checkpoints/last.ckpt",
    "Rigid Docking + Pharmacophore": "protpharm_cond_cr_2025-11-24_13-00-028503/checkpoints/last.ckpt",
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
        default_ckpt_dir = Path(omtra_root()) / "omtra/trained_models/"
        checkpoint_dir = Path(os.getenv("OMTRA_CHECKPOINT_DIR", str(default_ckpt_dir)))
    
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
    import logging
    logger = logging.getLogger(__name__)
    
    checkpoint_filename = WEBAPP_TO_CHECKPOINT.get(sampling_mode)
    if not checkpoint_filename:
        logger.error(f"No checkpoint mapping found for mode: {sampling_mode}")
        return None
    
    if checkpoint_dir is None:
        checkpoint_dir = Path(os.getenv("CHECKPOINT_DIR", "/srv/app/checkpoints"))
    
    checkpoint_path = checkpoint_dir / checkpoint_filename
    logger.info(f"Checking checkpoint for '{sampling_mode}': {checkpoint_path}")
    
    if checkpoint_path.exists():
        logger.info(f"Checkpoint found: {checkpoint_path}")
        return checkpoint_path
    else:
        logger.error(f"Checkpoint file does not exist: {checkpoint_path}")
        logger.error(f"Checkpoint dir: {checkpoint_dir}, exists: {checkpoint_dir.exists()}")
        return None


