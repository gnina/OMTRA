import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass

@dataclass
class StructureData:
    coords: np.ndarray
    atom_names: np.ndarray
    elements: np.ndarray
    res_ids: np.ndarray
    res_names: np.ndarray
    chain_ids: np.ndarray
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
    receptor: StructureData
    ligand: LigandData
    pharmacophore: PharmacophoreData
    pocket: StructureData
    npndes: Optional[Dict[str, LigandData]] = None
    link_id: Optional[str] = None
    link_type: Optional[str] = None
    link: Optional[StructureData] = None