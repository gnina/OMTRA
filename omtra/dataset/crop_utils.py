"""
Utilities for dynamic cropping of protein pocket structures at train time.

These functions operate on numpy arrays directly (no biotite conversion needed)
and use scipy.spatial.cKDTree for efficient O(n log n) radius queries.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from typing import Dict, Optional, Set, Tuple, List

from omtra.data.plinder import StructureData, BackboneData, LigandData


def compute_close_residues(
    protein_coords: np.ndarray,
    protein_res_ids: np.ndarray,
    protein_chain_ids: np.ndarray,
    reference_coords: np.ndarray,
    crop_distance: float,
) -> Set[Tuple]:
    """
    Find all (chain_id, res_id) pairs that have at least one atom within 
    crop_distance of any reference coordinate.
    
    Uses cKDTree for efficient spatial queries.
    
    Parameters
    ----------
    protein_coords : np.ndarray
        Shape (N, 3) - protein atom coordinates
    protein_res_ids : np.ndarray
        Shape (N,) - residue ID for each protein atom
    protein_chain_ids : np.ndarray
        Shape (N,) - chain ID for each protein atom
    reference_coords : np.ndarray
        Shape (M, 3) - reference coordinates (e.g., ligand atoms)
    crop_distance : float
        Distance cutoff for considering an atom "close"
        
    Returns
    -------
    Set[Tuple]
        Set of (chain_id, res_id) pairs that are within crop_distance
    """
    if len(protein_coords) == 0 or len(reference_coords) == 0:
        return set()
    
    # Build KD-tree on protein coordinates
    tree = cKDTree(protein_coords)
    
    # Query all reference points at once
    close_atom_indices = set()
    for ref_coord in reference_coords:
        indices = tree.query_ball_point(ref_coord, r=crop_distance)
        close_atom_indices.update(indices)
    
    # Collect unique (chain_id, res_id) pairs from close atoms
    close_residues = set()
    for idx in close_atom_indices:
        close_residues.add((protein_chain_ids[idx], protein_res_ids[idx]))
    
    return close_residues


def crop_structure_data(
    structure: StructureData,
    reference_coords: np.ndarray,
    crop_distance: float,
) -> Optional[StructureData]:
    """
    Crop a StructureData to only include residues within crop_distance of 
    reference coordinates.
    
    Parameters
    ----------
    structure : StructureData
        The protein structure to crop (e.g., pocket)
    reference_coords : np.ndarray
        Shape (M, 3) - reference coordinates (typically ligand atoms)
    crop_distance : float
        Distance cutoff for the crop
        
    Returns
    -------
    Optional[StructureData]
        Cropped structure, or None if no residues remain
    """
    # Find residues within crop distance
    close_residues = compute_close_residues(
        structure.coords,
        structure.res_ids,
        structure.chain_ids,
        reference_coords,
        crop_distance,
    )
    
    if not close_residues:
        return None
    
    # Build atom-level mask: include all atoms from close residues
    atom_mask = np.array([
        (structure.chain_ids[i], structure.res_ids[i]) in close_residues
        for i in range(len(structure.coords))
    ], dtype=bool)
    
    if not atom_mask.any():
        return None
    
    # Crop backbone data if present
    cropped_backbone = None
    if structure.backbone is not None and structure.backbone.res_ids is not None:
        bb = structure.backbone
        bb_mask = np.array([
            (bb.chain_ids[i], bb.res_ids[i]) in close_residues
            for i in range(len(bb.coords))
        ], dtype=bool)
        
        if bb_mask.any():
            cropped_backbone = BackboneData(
                coords=bb.coords[bb_mask],
                res_ids=bb.res_ids[bb_mask],
                res_names=bb.res_names[bb_mask],
                chain_ids=bb.chain_ids[bb_mask],
            )
    
    return StructureData(
        coords=structure.coords[atom_mask],
        atom_names=structure.atom_names[atom_mask] if structure.atom_names is not None else None,
        elements=structure.elements[atom_mask] if structure.elements is not None else None,
        res_ids=structure.res_ids[atom_mask],
        res_names=structure.res_names[atom_mask] if structure.res_names is not None else None,
        chain_ids=structure.chain_ids[atom_mask],
        backbone_mask=structure.backbone_mask[atom_mask] if structure.backbone_mask is not None else None,
        backbone=cropped_backbone,
        pocket_embedding=None,  # embeddings would need special handling if present
        cif=structure.cif,
    )

def crop_structure_data_multiple_distances(
    structure: StructureData,
    reference_coords_groups: List[np.ndarray],
    crop_distances: List[float],
) -> Optional[StructureData]:
    """
    Crop a StructureData to only include residues within specified crop distances 
    for different groups of reference coordinates.
    
    Parameters
    ----------
    structure : StructureData
        The protein structure to crop (e.g., pocket)
    reference_coords_groups : List[np.ndarray]
        List of reference coordinate groups, each of shape (M, 3)
        (typically groups of ligand atoms)
    crop_distances : List[float]
        List of distance cutoffs corresponding to each reference coordinate group
        
    Returns
    -------
    Optional[StructureData]
        Cropped structure, or None if no residues remain
    """
    # Find residues within crop distances for all coordinate groups
    close_residues = set()
    
    for ref_coords, crop_dist in zip(reference_coords_groups, crop_distances):
        # Compute close residues for this specific group and distance
        group_close_residues = compute_close_residues(
            structure.coords,
            structure.res_ids,
            structure.chain_ids,
            ref_coords,
            crop_dist,
        )
        
        # Union the close residues from all groups
        close_residues.update(group_close_residues)
    
    if not close_residues:
        return None
    
    # Build atom-level mask: include all atoms from close residues
    atom_mask = np.array([
        (structure.chain_ids[i], structure.res_ids[i]) in close_residues
        for i in range(len(structure.coords))
    ], dtype=bool)
    
    if not atom_mask.any():
        return None
    
    # Crop backbone data if present
    cropped_backbone = None
    if structure.backbone is not None and structure.backbone.res_ids is not None:
        bb = structure.backbone
        bb_mask = np.array([
            (bb.chain_ids[i], bb.res_ids[i]) in close_residues
            for i in range(len(bb.coords))
        ], dtype=bool)
        
        if bb_mask.any():
            cropped_backbone = BackboneData(
                coords=bb.coords[bb_mask],
                res_ids=bb.res_ids[bb_mask],
                res_names=bb.res_names[bb_mask],
                chain_ids=bb.chain_ids[bb_mask],
            )
    
    return StructureData(
        coords=structure.coords[atom_mask],
        atom_names=structure.atom_names[atom_mask] if structure.atom_names is not None else None,
        elements=structure.elements[atom_mask] if structure.elements is not None else None,
        res_ids=structure.res_ids[atom_mask],
        res_names=structure.res_names[atom_mask] if structure.res_names is not None else None,
        chain_ids=structure.chain_ids[atom_mask],
        backbone_mask=structure.backbone_mask[atom_mask] if structure.backbone_mask is not None else None,
        backbone=cropped_backbone,
        pocket_embedding=None,  # embeddings would need special handling if present
        cif=structure.cif,
    )

def compute_structure_crop_mask(
    structure: StructureData,
    reference_coords: np.ndarray,
    crop_distance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute atom-level and backbone-level boolean masks for cropping a structure.
    
    This is useful when you need to apply the same mask to aligned structures
    (e.g., holo pocket and apo/pred link structures).
    
    Parameters
    ----------
    structure : StructureData
        The protein structure to compute masks for
    reference_coords : np.ndarray
        Shape (M, 3) - reference coordinates (typically ligand atoms)
    crop_distance : float
        Distance cutoff for the crop
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (atom_mask, backbone_mask) - boolean arrays
    """
    close_residues = compute_close_residues(
        structure.coords,
        structure.res_ids,
        structure.chain_ids,
        reference_coords,
        crop_distance,
    )
    
    # Atom-level mask
    atom_mask = np.array([
        (structure.chain_ids[i], structure.res_ids[i]) in close_residues
        for i in range(len(structure.coords))
    ], dtype=bool)
    
    # Backbone-level mask
    bb_mask = np.zeros(0, dtype=bool)
    if structure.backbone is not None and structure.backbone.res_ids is not None:
        bb = structure.backbone
        bb_mask = np.array([
            (bb.chain_ids[i], bb.res_ids[i]) in close_residues
            for i in range(len(bb.coords))
        ], dtype=bool)
    
    return atom_mask, bb_mask


def filter_npndes_by_distance(
    npndes: Optional[Dict[str, LigandData]],
    reference_coords: np.ndarray,
    crop_distance: float,
) -> Optional[Dict[str, LigandData]]:
    """
    Filter NPNDEs to only include those with at least one atom within 
    crop_distance of reference coordinates.
    
    Parameters
    ----------
    npndes : Optional[Dict[str, LigandData]]
        Dictionary mapping NPNDE IDs to LigandData objects
    reference_coords : np.ndarray
        Shape (M, 3) - reference coordinates (typically ligand atoms)
    crop_distance : float
        Distance cutoff
        
    Returns
    -------
    Optional[Dict[str, LigandData]]
        Filtered dictionary, or None if empty
    """
    if npndes is None or len(npndes) == 0:
        return None
    
    if len(reference_coords) == 0:
        return None
    
    filtered = {}
    for npnde_id, npnde_data in npndes.items():
        if len(npnde_data.coords) == 0:
            continue
            
        # Compute minimum distance between NPNDE and reference coords
        dists = cdist(npnde_data.coords, reference_coords)
        min_dist = dists.min()
        
        if min_dist <= crop_distance:
            filtered[npnde_id] = npnde_data
    
    return filtered if filtered else None


def sample_crop_distance(min_distance: float, max_distance: float) -> float:
    """
    Sample a crop distance uniformly between min and max.
    
    Parameters
    ----------
    min_distance : float
        Minimum crop distance
    max_distance : float
        Maximum crop distance
        
    Returns
    -------
    float
        Sampled crop distance
    """
    return np.random.uniform(min_distance, max_distance)
