import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass

@dataclass
class BackboneData:
    coords: np.ndarray # (num_res, 3, 3)
    res_ids: np.ndarray
    res_names: np.ndarray
    chain_ids: np.ndarray
    

@dataclass
class StructureData: 
    coords: np.ndarray
    atom_names: np.ndarray
    elements: np.ndarray
    res_ids: np.ndarray
    res_names: np.ndarray
    chain_ids: np.ndarray
    backbone_mask: np.ndarray
    backbone: BackboneData
    cif: Optional[str] = None


@dataclass
class LigandData:
    coords: np.ndarray
    atom_types: np.ndarray
    atom_charges: np.ndarray
    bond_types: np.ndarray
    bond_indices: np.ndarray
    is_covalent: bool
    ccd: str
    sdf: str
    linkages: Optional[List[str]] = (
        None  # "{auth_resid}:{resname}{assym_id}{seq_resid}{atom_name}__{auth_resid}:{resname}{assym_id}{seq_resid}{atom_name}"
    )


@dataclass
class PharmacophoreData: 
    coords: np.ndarray
    types: np.ndarray
    vectors: np.ndarray
    interactions: np.ndarray


@dataclass
class SystemData:
    system_id: str
    ligand_id: str
    receptor: StructureData # 150 KB
    ligand: LigandData # 1.2 KB
    pharmacophore: PharmacophoreData # 1.6 KB
    pocket: StructureData # 10 KB
    npndes: Optional[Dict[str, LigandData]] = None # 1.2 KB
    link_id: Optional[str] = None
    link_type: Optional[str] = None
    link: Optional[StructureData] = None # 150 KB