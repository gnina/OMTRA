"""
Pocket detection using pocketeer.
"""
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
import numpy as np
import warnings

logger = logging.getLogger(__name__)

# Suppress warnings from biotite/pocketeer
warnings.filterwarnings("ignore")

def sanitize_residues(atomarray):
    """
    Convert non-standard residue names to standard ones for Biotite/Pocketeer compatibility.
    """
    # Map common protonation states to standard names
    mapping = {
        "HIE": "HIS", "HID": "HIS", "HIP": "HIS",
        "CYX": "CYS", "CYM": "CYS",
        "ASH": "ASP", "GLH": "GLU",
        "LYN": "LYS"
    }
    
    count = 0
    if hasattr(atomarray, "res_name"):
        for i, res_name in enumerate(atomarray.res_name):
            if res_name in mapping:
                atomarray.res_name[i] = mapping[res_name]
                count += 1
    
    if count > 0:
        logger.info(f"Sanitized {count} residue names in protein structure")
    return atomarray

def detect_pockets(
    protein_content: bytes,
    protein_format: str = 'pdb',
    min_pocket_volume: float = 100.0,
    max_pockets: int = 10
) -> List[Dict[str, Any]]:
    """
    Detect pockets in a protein structure using pocketeer.
    
    Args:
        protein_content: Protein file content as bytes (PDB or CIF format)
        protein_format: Format of the protein file ('pdb' or 'cif')
        min_pocket_volume: Minimum pocket volume in Angstrom^3
        max_pockets: Maximum number of pockets to return
        
    Returns:
        List of pocket dictionaries with keys:
        - id: Unique pocket identifier
        - center: [x, y, z] coordinates of pocket center
        - bbox_length: Bounding box side length in Angstrom (cubic approximation)
        - score: Confidence score (if available)
        - volume: Pocket volume in Angstrom^3 (if available)
    """
    try:
        import pocketeer as pt
    except ImportError:
        logger.error("pocketeer is not installed. Please install it: pip install pocketeer")
        raise ImportError("pocketeer is required for pocket detection. Install with: pip install pocketeer")
    
    # Create temporary file for protein structure
    with tempfile.NamedTemporaryFile(mode='wb', suffix=f'.{protein_format}', delete=False) as tmp_file:
        tmp_file.write(protein_content)
        tmp_path = Path(tmp_file.name)
    
    try:
        # Load structure
        try:
            atomarray = pt.load_structure(str(tmp_path))
        except Exception as e:
            logger.error(f"Failed to load structure with pocketeer: {e}")
            raise ValueError(f"Failed to load protein structure: {e}")
            
        # Sanitize residues to prevent errors with non-standard names
        atomarray = sanitize_residues(atomarray)
        
        # Detect pockets
        try:
            pockets = pt.find_pockets(atomarray)
        except Exception as e:
            logger.error(f"Pocketeer failed to find pockets: {e}")
            # Try once more with relaxed settings if possible using fallback or just raise
            raise ValueError(f"Pocket detection failed: {e}")
            
        logger.info(f"Pocketeer found {len(pockets)} pockets")
        
        # Convert pocketeer results to our format
        result = []
        for i, pocket in enumerate(pockets):
            # Extract pocket volume
            volume = getattr(pocket, 'volume', 0.0)
            
            # Filter by minimum volume
            if volume < min_pocket_volume:
                continue
            
            # Stop if we have enough pockets
            if len(result) >= max_pockets:
                break
                
            pocket_id = getattr(pocket, 'pocket_id', i+1)
            
            # Get center coordinates
            center = getattr(pocket, 'centroid', None)
            if center is None:
                # Fallback to calculating from spheres
                if hasattr(pocket, 'spheres') and len(pocket.spheres) > 0:
                     centers = np.array([s.center for s in pocket.spheres])
                     center = np.mean(centers, axis=0)
                else:
                    center = np.array([0.0, 0.0, 0.0]) # Should not happen for valid pocket
            
            # Calculate bounding box from spheres
            bbox_length = 20.0 # Default fallback
            if hasattr(pocket, "spheres") and len(pocket.spheres) > 0:
                 centers = np.array([s.center for s in pocket.spheres])
                 min_coords = np.min(centers, axis=0)
                 max_coords = np.max(centers, axis=0)
                 # Add some padding (radius of alpha spheres is roughly 1-4A)
                 padding = 4.0 
                 bbox_size = np.max(max_coords - min_coords) + padding
                 bbox_length = float(bbox_size)
            
            # Get score
            score = getattr(pocket, 'score', 0.0)
            
            result.append({
                'id': f"pocket_{pocket_id}",
                'center': [float(center[0]), float(center[1]), float(center[2])],
                'bbox_length': float(bbox_length),
                'score': float(score),
                'volume': float(volume),
            })
        
        logger.info(f"Returning {len(result)} valid pockets after filtering")
        return result
        
    except Exception as e:
        logger.error(f"Error detecting pockets: {e}", exc_info=True)
        raise
    finally:
        # Clean up temporary file
        try:
            tmp_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {tmp_path}: {e}")
