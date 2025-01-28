import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np
from biotite.structure.io.pdbx import CIFFile, get_structure
from omtra.data.xae_ligand import MoleculeTensorizer
from rdkit import Chem

from plinder.core import PlinderSystem


@dataclass
class StructureData:
    coords: np.ndarray
    atom_names: np.ndarray
    res_ids: np.ndarray
    res_names: np.ndarray
    chain_ids: np.ndarray
    res_idx: Optional[np.ndarray] = None


@dataclass
class LigandData:
    coords: np.ndarray
    atom_types: np.ndarray
    atom_charges: np.ndarray
    bond_types: Optional[np.ndarray]
    bond_indices: Optional[np.ndarray]


class PDBWriter:
    def __init__(self, chain_mapping: Optional[Dict[str, str]] = None):
        self.chain_mapping = chain_mapping

    def write(self, struct_data: StructureData, output_path: str):
        struct = struc.AtomArray(len(struct_data.coords))

        struct.coord = struct_data.coords
        struct.atom_name = struct_data.atom_names
        struct.res_id = struct_data.res_ids
        struct.chain_id = np.array(
            [self.chain_mapping[chain_id] for chain_id in struct_data.chain_ids]
        )
        struct.res_name = struct_data.res_names
        struct.hetero = np.full(len(struct_data.coords), False)

        pdb_file = pdb.PDBFile()
        pdb_file.set_structure(struct)
        pdb_file.write(output_path)


class StructureProcessor:
    def __init__(self, atom_map: List[str], pocket_cutoff: float = 5.0, n_cpus: int = 1):
        self.atom_map = atom_map
        self.pocket_cutoff = pocket_cutoff
        self.tensorizer = MoleculeTensorizer(atom_map=atom_map, n_cpus=n_cpus)

    def load_structure(self, path: str) -> struc.AtomArray:
        cif_file = CIFFile.read(path)
        structure = get_structure(
            cif_file, model=1, use_author_fields=False, include_bonds=True
        )
        return structure[structure.res_name != "HOH"]

    def process_structure(self, structure: struc.AtomArray, chain_mapping: Optional[Dict[str, str]] = None) -> StructureData:
        if chain_mapping is not None:
            chain_ids = [chain_mapping.get(chain, chain) for chain in structure.chain_id]
        else:
            chain_ids = structure.chain_id

        return StructureData(
            coords=structure.coord,
            atom_names=structure.atom_name,
            res_ids=structure.res_id,
            res_names=structure.res_name,
            chain_ids=chain_ids,
        )

    def process_ligands(self, ligand_paths: List[str]) -> Dict[str, LigandData]:
        ligand_mols = [Chem.SDMolSupplier(path)[0] for path in ligand_paths]
        (
            positions,
            atom_types,
            atom_charges,
            bond_types,
            bond_idxs,
            _,
        ) = self.tensorizer.featurize_molecules(ligand_mols)

        ligands_data = {}
        for i, path in enumerate(ligand_paths):
            ligand_key = os.path.splitext(os.path.basename(path))[0]

            ligands_data[ligand_key] = LigandData(
                coords=positions[i],
                atom_types=atom_types[i],
                atom_charges=atom_charges[i],
                bond_types=bond_types[i] if bond_types[i] is not None else None,
                bond_indices=bond_idxs[i] if bond_idxs[i] is not None else None,
            )

        return ligands_data

    def extract_pocket(
        self, receptor: struc.AtomArray, ligand_coords: np.ndarray
    ) -> StructureData:
        receptor_cell_list = struc.CellList(receptor, cell_size=self.pocket_cutoff)

        close_atom_indices = []
        for lig_coord in ligand_coords:
            indices = receptor_cell_list.get_atoms(lig_coord, radius=self.pocket_cutoff)
            close_atom_indices.extend(indices)

        close_res_ids = receptor.res_id[close_atom_indices]
        close_chain_ids = receptor.chain_id[close_atom_indices]
        unique_res_pairs = set(zip(close_res_ids, close_chain_ids))

        pocket_indices = []
        for res_id, chain_id in unique_res_pairs:
            res_mask = (receptor.res_id == res_id) & (receptor.chain_id == chain_id)
            res_indices = np.where(res_mask)[0]
            pocket_indices.extend(res_indices)

        unique_res_ids = sorted(set(receptor.res_id[pocket_indices]))
        res_id_to_idx = {res_id: idx for idx, res_id in enumerate(unique_res_ids)}
        pocket_res_idx = np.array(
            [res_id_to_idx[res_id] for res_id in receptor.res_id[pocket_indices]],
            dtype=np.int64,
        )

        return StructureData(
            coords=receptor.coord[pocket_indices],
            atom_names=receptor.atom_name[pocket_indices],
            res_ids=receptor.res_id[pocket_indices],  # original residue ids
            res_names=receptor.res_name[pocket_indices],
            chain_ids=receptor.chain_id[pocket_indices],
            res_idx=pocket_res_idx,  # 0-indexed pocket residue ids
        )


class SystemProcessor:
    def __init__(self, atom_map: List[str], pocket_cutoff: float = 5.0):
        self.structure_processor = StructureProcessor(
            atom_map, pocket_cutoff=pocket_cutoff
        )
        self.pdb_writer = None

    def process_system(self, system_id: str, save_pockets: bool = False) -> Dict:
        system = PlinderSystem(system_id=system_id)
        receptor_path = system.receptor_cif
        ligand_paths = list(system.ligand_sdfs.values())

        # Get apo paths
        apo_ids = system.linked_structures[system.linked_structures["kind"] == "apo"][
            "id"
        ].tolist()
        apo_paths = [
            system.get_linked_structure(link_kind="apo", link_id=id) for id in apo_ids
        ]

        result = self.process_structures(
            receptor_path=receptor_path,
            ligand_paths=ligand_paths,
            apo_paths=apo_paths,
            chain_mapping=system.chain_mapping,
            save_pockets=save_pockets,
        )

        result["entry_annotation"] = system.entry
        result["system_annotation"] = system.system

        return result

    def process_structures(
        self,
        receptor_path: str,
        ligand_paths: List[str],
        apo_paths: Optional[List[str]] = None,
        chain_mapping: Optional[Dict[str, str]] = None,
        save_pockets: bool = False,
    ) -> Dict:
        if save_pockets:
            self.pdb_writer = PDBWriter(chain_mapping)

        # Process receptor
        receptor = self.structure_processor.load_structure(receptor_path)
        receptor_data = self.structure_processor.process_structure(receptor, chain_mapping)

        # Process ligands
        ligands_data = self.structure_processor.process_ligands(ligand_paths)

        # Process pockets
        pockets_data = {}
        for ligand_key, ligand in ligands_data.items():
            pocket_data = self.structure_processor.extract_pocket(
                receptor, ligand.coords
            )

            if save_pockets:
                output_dir = os.path.dirname(receptor_path)
                pocket_path = os.path.join(output_dir, f"pocket_{ligand_key}.pdb")
                self.pdb_writer.write(pocket_data, pocket_path)

            pockets_data[ligand_key] = pocket_data

        # Process apo structures
        apo_structures = {}
        if apo_paths:
            for apo_path in apo_paths:
                filename = os.path.basename(apo_path)
                apo_key = os.path.splitext(filename)[0]
                apo_struct = self.structure_processor.load_structure(apo_path)
                apo_structures[apo_key] = self.structure_processor.process_structure(
                    apo_struct
                )

        return {
            "receptor": receptor_data,
            "ligands": ligands_data,
            "pockets": pockets_data,
            "apo_structures": apo_structures if apo_paths else None,
        }
