import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
import biotite.structure as struc
from omtra.constants import charge_map

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
    pocket_embedding: Optional[np.ndarray] = None
    cif: Optional[str] = None
    
    def to_atom_array(self) -> struc.AtomArray:
        n_atoms = len(self.coords)
        atom_array = struc.AtomArray(n_atoms)
        
        atom_array.coord = self.coords
        
        atom_array.set_annotation("atom_name", self.atom_names)
        atom_array.set_annotation("element", self.elements)
        atom_array.set_annotation("res_id", self.res_ids)
        atom_array.set_annotation("res_name", self.res_names)
        atom_array.set_annotation("chain_id", self.chain_ids)
        
        return atom_array


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
        None  # "{auth_resid}:{resname}:{assym_id}:{seq_resid}:{atom_name}__{auth_resid}:{resname}:{assym_id}:{seq_resid}:{atom_name}"
    )
    
    def to_atom_array(self, atom_type_map) -> struc.AtomArray:
        n_atoms = len(self.coords)
        atom_array = struc.AtomArray(n_atoms)
        
        atom_array.coord = self.coords
        atom_array.set_annotation("atom_name", np.array([atom_type_map[atom] for atom in self.atom_types]))
        atom_array.set_annotation("element", np.array([atom_type_map[atom] for atom in self.atom_types])) 
        atom_array.set_annotation("res_id", np.full(n_atoms, 1, dtype=int))
        atom_array.set_annotation("res_name", np.full(n_atoms, self.ccd))
        atom_array.set_annotation("chain_id", np.full(n_atoms, self.ccd))
        atom_array.set_annotation("charge", np.array([charge_map[int(charge)] for charge in self.atom_charges]))
        
        return atom_array


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
    ligand: LigandData # 1.2 KB
    receptor: Optional[StructureData] = None # 150 KB
    pharmacophore: Optional[PharmacophoreData] = None # 1.6 KB
    pocket: Optional[StructureData] = None # 10 KB
    npndes: Optional[Dict[str, LigandData]] = None # 1.2 KB
    link_id: Optional[str] = None
    link_type: Optional[str] = None
    link: Optional[StructureData] = None # 150 KB