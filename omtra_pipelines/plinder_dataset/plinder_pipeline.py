import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np
from biotite.structure.io.pdbx import CIFFile, get_structure
from omtra.data.xace_ligand import MoleculeTensorizer
from plinder.core import PlinderSystem
from rdkit import Chem

logger = logging.getLogger(__name__)


@dataclass
class StructureData:
    coords: np.ndarray
    atom_names: np.ndarray
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
    sdf: str


class PDBWriter:
    def __init__(self, chain_mapping: Optional[Dict[str, str]] = None):
        self.chain_mapping = chain_mapping

    def write(self, struct_data: StructureData, output_path: str):
        logger.info("Writing structure to %s", output_path)
        struct = struc.AtomArray(len(struct_data.coords))

        struct.coord = struct_data.coords
        struct.atom_name = struct_data.atom_names
        struct.res_id = struct_data.res_ids
        struct.res_name = struct_data.res_names
        struct.hetero = np.full(len(struct_data.coords), False)

        pdb_file = pdb.PDBFile()
        pdb_file.set_structure(struct)
        pdb_file.write(output_path)


class StructureProcessor:
    def __init__(
        self,
        ligand_atom_map: List[str],
        npnde_atom_map: List[str],
        pocket_cutoff: float = 8.0,
        n_cpus: int = 1,
        raw_data: str = "/net/galaxy/home/koes/tjkatz/.local/share/plinder/2024-06/v2",
    ):
        logger.debug("Initializing StructureProcessor with cutoff=%f", pocket_cutoff)
        self.ligand_atom_map = ligand_atom_map
        self.npnde_atom_map = npnde_atom_map
        self.pocket_cutoff = pocket_cutoff
        self.ligand_tensorizer = MoleculeTensorizer(
            atom_map=ligand_atom_map, n_cpus=n_cpus
        )
        self.npnde_tensorizer = MoleculeTensorizer(
            atom_map=npnde_atom_map, n_cpus=n_cpus
        )
        self.raw_data = Path(raw_data)

    def process_receptor(
        self,
        receptor: struc.AtomArray,
        cif: str,
        chain_mapping: Optional[Dict[str, str]] = None,
    ) -> StructureData:
        receptor = receptor[receptor.res_name != "HOH"]

        raw_cif = Path(cif).relative_to(self.raw_data)

        if chain_mapping is not None:
            chain_ids = [chain_mapping.get(chain, chain) for chain in receptor.chain_id]
            receptor.chain_id = chain_ids

        return StructureData(
            cif=str(raw_cif),
            coords=receptor.coord,
            atom_names=receptor.atom_name,
            res_ids=receptor.res_id,
            res_names=receptor.res_name,
            chain_ids=receptor.chain_id,
        )

    def check_ordering(
        self, receptor: struc.AtomArray, linked_structure: struc.AtomArray
    ) -> bool:
        for i, (rec_atom, linked_atom) in enumerate(zip(receptor, linked_structure)):
            if (
                rec_atom.atom_name != linked_atom.atom_name
                or rec_atom.res_name != linked_atom.res_name
            ):
                return False
        return True

    def create_atom_key(self, atom: struc.Atom) -> str:
        return f"{atom.chain_id}_{atom.res_id}_{atom.res_name}_{atom.atom_name}"

    def reorder(
        self,
        system_id: str,
        link_id: str,
        receptor: struc.AtomArray,
        linked_structure: struc.AtomArray,
    ) -> struc.AtomArray:
        receptor_keys = [self.create_atom_key(atom) for atom in receptor]
        linked_keys = [self.create_atom_key(atom) for atom in linked_structure]

        if set(receptor_keys) != set(linked_keys):
            logger.warning(
                f"Atom key set mismatch between receptor and linked structure {system_id}_{link_id}"
            )
            return None

        reorder_indices = []
        for key in receptor_keys:
            idx = linked_keys.index(key)
            reorder_indices.append(idx)

        if len(reorder_indices) != len(set(reorder_indices)):
            logger.warning(f"Failed reordering for {system_id}_{link_id}")
            return None

        reordered = linked_structure[reorder_indices]

        return reordered

    def process_linked_structure(
        self, system: PlinderSystem, linked_id: str
    ) -> Dict[str, StructureData]:
        holo = system.holo_structure
        linked_structure = system.alternate_structures[linked_id]
        linked_structure.set_chain(holo.protein_chain_ordered[0])

        holo_cropped, linked_cropped = holo.align_common_sequence(
            linked_structure, renumber_residues=True
        )
        linked_cropped_superposed, raw_rmsd, refined_rmsd = linked_cropped.superimpose(
            holo_cropped
        )

        aligned = self.check_ordering(
            holo_cropped.protein_atom_array,
            linked_cropped_superposed.protein_atom_array,
        )
        if not aligned:
            reordered_arr = self.reorder(
                system.system_id,
                linked_id,
                holo_cropped.protein_atom_array,
                linked_cropped_superposed.protein_atom_array,
            )
            if reordered_arr:
                linked_cropped_superposed.protein_atom_array = reordered_arr
            else:
                return None

        holo_data = self.process_receptor(
            holo_cropped.protein_atom_array,
            str(holo_cropped.protein_path),
            system.chain_mapping,
        )
        linked_data = self.process_receptor(
            linked_cropped_superposed.protein_atom_array,
            str(linked_cropped_superposed.protein_path),
            system.chain_mapping,
        )

        return {"holo": holo_data, linked_id: linked_data}

    def process_ligands(
        self, system: PlinderSystem, ligand_mols: Dict[str, Chem.rdchem.Mol]
    ) -> Dict[str, LigandData]:
        keys = list(ligand_mols.keys())
        mols = list(ligand_mols.values())

        (positions, atom_types, atom_charges, bond_types, bond_idxs, _, failed_idxs) = (
            self.ligand_tensorizer.featurize_molecules(mols)
        )

        for i in failed_idxs:
            logger.warning("Failed to tensorize ligand %s", keys[i])

        ligand_keys = [key for i, key in enumerate(keys) if i not in failed_idxs]

        ligands_data = {}
        for i, key in enumerate(ligand_keys):
            raw_sdf = Path(system.ligand_sdfs[key]).relative_to(self.raw_data)

            ligands_data[key] = LigandData(
                sdf=str(raw_sdf),
                coords=np.array(positions[i], dtype=np.float32),
                atom_types=atom_types[i],
                atom_charges=atom_charges[i],
                bond_types=bond_types[i],
                bond_indices=bond_idxs[i],
            )

        return ligands_data

    def process_npndes(
        self, system: PlinderSystem, npnde_mols: Dict[str, Chem.rdchem.Mol]
    ) -> Dict[str, LigandData]:
        keys = list(npnde_mols.keys())
        mols = list(npnde_mols.values())

        (positions, atom_types, atom_charges, bond_types, bond_idxs, _, failed_idxs) = (
            self.npnde_tensorizer.featurize_molecules(mols)
        )
        for i in failed_idxs:
            logger.warning("Failed to tensorize npnde %s", keys[i])

        npnde_keys = [key for i, key in enumerate(keys) if i not in failed_idxs]

        npnde_data = {}
        for i, key in enumerate(npnde_keys):
            raw_sdf = Path(system.ligand_sdfs[key]).relative_to(self.raw_data)
            npnde_data[key] = LigandData(
                sdf=str(raw_sdf),
                coords=np.array(positions[i], dtype=np.float32),
                atom_types=atom_types[i],
                atom_charges=atom_charges[i],
                bond_types=bond_types[i],
                bond_indices=bond_idxs[i],
            )

        return npnde_data

    def extract_pocket(
        self,
        receptor: struc.AtomArray,
        ligand_coords: np.ndarray,
        chain_mapping: Optional[Dict[str, str]] = None,
    ) -> StructureData:
        logger.debug("Extracting pocket")
        receptor = receptor[receptor.res_name != "HOH"]
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

        if len(pocket_indices) == 0:
            return None

        if chain_mapping is not None:
            chain_ids = [chain_mapping.get(chain, chain) for chain in receptor.chain_id]
            receptor.chain_id = chain_ids

        return StructureData(
            coords=receptor.coord[pocket_indices],
            atom_names=receptor.atom_name[pocket_indices],
            res_ids=receptor.res_id[pocket_indices],  # original residue ids
            res_names=receptor.res_name[pocket_indices],
            chain_ids=receptor.chain_id[pocket_indices],
        )


class SystemProcessor:
    def __init__(
        self,
        ligand_atom_map: List[str],
        npnde_atom_map: List[str],
        pocket_cutoff: float = 8.0,
        raw_data: str = "/net/galaxy/home/koes/tjkatz/.local/share/plinder/2024-06/v2",
    ):
        logger.debug("Initializing SystemProcessor with cutoff=%f", pocket_cutoff)
        self.structure_processor = StructureProcessor(
            ligand_atom_map=ligand_atom_map,
            npnde_atom_map=npnde_atom_map,
            pocket_cutoff=pocket_cutoff,
            raw_data=raw_data,
        )
        self.pdb_writer = None

    def filter_ligands(
        self, system: PlinderSystem
    ) -> (Dict[str, Chem.rdchem.Mol], Dict[str, Chem.rdchem.Mol]):
        system_annotation = system.system
        system_structure = system.holo_structure

        proper_ligand_keys = []
        npnde_keys = []

        for ligand in system_annotation["ligands"]:
            lig_key = str(ligand["instance"]) + "." + ligand["asym_id"]

            if ligand["is_ion"] or ligand["is_artifact"]:
                npnde_keys.append(lig_key)
            else:
                num_interacting_res = 0
                for chain, res_list in ligand["interacting_residues"].items():
                    num_interacting_res += len(res_list)

                if num_interacting_res > 0:
                    proper_ligand_keys.append(lig_key)

        if len(proper_ligand_keys) < 1:
            logger.warning("Skipping %s due to no proper ligands", system.system_id)
            return None, None

        ligand_mols = {}
        npnde_mols = {}
        for key, mol in system_structure.resolved_ligand_mols.items():
            if key in proper_ligand_keys:
                ligand_mols[key] = mol
            elif key in npnde_keys:
                npnde_mols[key] = mol

        return ligand_mols, npnde_mols

    def process_system(
        self, system_id: str, save_pockets: bool = False
    ) -> Dict[str, Any]:
        logger.info("Processing system %s", system_id)

        system = PlinderSystem(system_id=system_id)

        ligand_mols, npnde_mols = self.filter_ligands(system)

        if not ligand_mols:
            return None

        # Get apo ids
        apo_ids = system.linked_structures[system.linked_structures["kind"] == "apo"][
            "id"
        ].tolist()

        # Get pred ids
        pred_ids = system.linked_structures[system.linked_structures["kind"] == "pred"][
            "id"
        ].tolist()

        result = self.process_structures(
            system_id=system_id,
            system=system,
            ligand_mols=ligand_mols,
            npnde_mols=npnde_mols,
            apo_ids=apo_ids,
            pred_ids=pred_ids,
            chain_mapping=system.chain_mapping,
            save_pockets=save_pockets,
        )

        if not result:
            logger.warning("Skipping system %s due to no ligands remaining", system_id)
            return None

        result["system_annotation"] = system.system
        result["system_id"] = system_id

        num_npndes = 0
        if result["npndes"]:
            num_npndes = len(result["npndes"])

        logger.info(
            "Processed system %s with %d ligands and %d npndes",
            system_id,
            len(result["ligands"]),
            num_npndes,
        )

        return result

    def process_structures(
        self,
        system_id: str,
        system: PlinderSystem,
        ligand_mols: Dict[str, Chem.rdchem.Mol],
        npnde_mols: Optional[Dict[str, Chem.rdchem.Mol]] = None,
        apo_ids: Optional[List[str]] = None,
        pred_ids: Optional[List[str]] = None,
        chain_mapping: Optional[Dict[str, str]] = None,
        save_pockets: bool = False,
    ) -> Dict[str, Any]:
        if save_pockets:
            self.pdb_writer = PDBWriter(chain_mapping)

        system_structure = system.holo_structure

        # Process receptor
        receptor_data = self.structure_processor.process_receptor(
            system_structure.protein_atom_array,
            str(system_structure.protein_path),
            chain_mapping,
        )

        # Process ligands
        ligands_data = self.structure_processor.process_ligands(system, ligand_mols)

        if npnde_mols:
            npnde_data = self.structure_processor.process_npndes(system, npnde_mols)

        # Process pockets
        pockets_data = {}
        ligands_to_remove = []
        for ligand_key, ligand in ligands_data.items():
            pocket_data = self.structure_processor.extract_pocket(
                system_structure.protein_atom_array, ligand.coords, system.chain_mapping
            )

            if not pocket_data:
                logger.warning("No pocket extracted for %s", ligand.sdf)
                ligands_to_remove.append(ligand_key)
                continue

            logger.info(
                "Extracted pocket with %d atoms for %s",
                len(pocket_data.coords),
                ligand.sdf,
            )

            if save_pockets:
                output_dir = os.path.dirname(system_structure.protein_path)
                pocket_path = os.path.join(output_dir, f"pocket_{ligand_key}.pdb")
                self.pdb_writer.write(pocket_data, pocket_path)

            pockets_data[ligand_key] = pocket_data

        for ligand_key in ligands_to_remove:
            del ligands_data[ligand_key]

        if len(ligands_data) < 1:
            return None

        # Process apo structures
        apo_structures = {}
        if apo_ids:
            for id in apo_ids:
                logger.info(
                    "Processing apo structure %s linked to system %s", id, system_id
                )
                apo = self.structure_processor.process_linked_structure(system, id)
                if apo:
                    apo_structures[id] = apo

        # Process pred structures
        pred_structures = {}
        if pred_ids:
            for id in pred_ids:
                logger.info(
                    "Processing pred structure %s linked to system %s", id, system_id
                )
                pred = self.structure_processor.process_linked_structure(system, id)
                if pred:
                    pred_structures[id] = pred

        return {
            "receptor": receptor_data,
            "ligands": ligands_data,
            "npndes": npnde_data if npnde_mols else None,
            "pockets": pockets_data,
            "apo_structures": apo_structures if apo_ids else None,
            "pred_structures": pred_structures if pred_ids else None,
        }
