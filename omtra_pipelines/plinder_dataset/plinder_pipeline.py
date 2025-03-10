import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np
from biotite.structure.io.pdbx import CIFFile, get_structure
from omtra.data.plinder import LigandData, PharmacophoreData, StructureData, SystemData
from omtra.data.pharmacophores import get_pharmacophores
from omtra.data.xace_ligand import MoleculeTensorizer
from omtra_pipelines.plinder_dataset.utils import _DEFAULT_DISTANCE_RANGE, setup_logger
from omtra_pipelines.plinder_dataset.filter import filter
from plinder.core import PlinderSystem
from rdkit import Chem

logger = setup_logger(
    __name__,
)


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
            elements=receptor.element,
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
        self,
        system: PlinderSystem,
        linked_id: str,
        ligand_data: Dict[str, LigandData],
        has_covalent: bool = False,
    ) -> (Dict[str, Any], Dict[int, int]):
        holo = system.holo_structure
        linked_structure = system.alternate_structures[linked_id]
        linked_structure.set_chain(holo.protein_chain_ordered[0])

        holo_cropped, linked_cropped = holo.align_common_sequence(
            linked_structure, renumber_residues=True
        )

        res_id_map = {}
        if has_covalent:
            holo_cropped_og, _ = holo.align_common_sequence(
                linked_structure, renumber_residues=False
            )
            for i, atom in enumerate(holo_cropped_og.protein_atom_array):
                if atom.chain_id not in res_id_map:
                    res_id_map[atom.chain_id] = {}
                if atom.res_id not in res_id_map[atom.chain_id]:
                    res_id_map[atom.chain_id][atom.res_id] = (
                        holo_cropped.protein_atom_array[i].res_id
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
            if reordered_arr is not None:
                linked_cropped_superposed.protein_atom_array = reordered_arr
            else:
                return None, None

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
        pockets_data = {}
        for key, ligand in ligand_data.items():
            pocket = self.extract_pocket(
                receptor=holo_cropped.protein_atom_array,
                ligand_coords=ligand.coords,
                chain_mapping=system.chain_mapping,
            )
            if pocket:
                pockets_data[key] = pocket
        return {
            "holo": holo_data,
            linked_id: linked_data,
            "pockets": pockets_data,
        }, res_id_map

    def infer_covalent_linkages(
        self, system: PlinderSystem, ligand_id: str
    ) -> List[str]:
        system_cif = CIFFile.read(system.system_cif)
        system_struc = get_structure(system_cif, model=1, include_bonds=True)
        ligand = system_struc[system_struc.chain_id == ligand_id]

        receptor_cif = CIFFile.read(system.receptor_cif)
        receptor = get_structure(receptor_cif, model=1, include_bonds=True)
        receptor = receptor[receptor.res_name != "HOH"]

        linkages = []
        dists = struc.distance(
            ligand.coord[:, np.newaxis, :], receptor.coord[np.newaxis, :, :]
        )
        for i, lig_atom in enumerate(ligand):
            for j, rec_atom in enumerate(receptor):
                dist_range = _DEFAULT_DISTANCE_RANGE.get(
                    (lig_atom.element, rec_atom.element)
                ) or _DEFAULT_DISTANCE_RANGE.get((rec_atom.element, lig_atom.element))
                if dist_range is None:
                    continue
                else:
                    min_dist, max_dist = dist_range
                dist = dists[i, j]
                if dist >= min_dist and dist <= max_dist:
                    rec_assym_id = rec_atom.chain_id.split(".")[1]
                    lig_assym_id = lig_atom.chain_id.split(".")[1]
                    prtnr1 = f"{rec_atom.res_id}:{rec_atom.res_name}:{rec_assym_id}:{rec_atom.res_id}:{rec_atom.atom_name}"
                    prtnr2 = f"{lig_atom.res_id}:{lig_atom.res_name}:{lig_assym_id}:.:{lig_atom.atom_name}"
                    linkage = "__".join([prtnr1, prtnr2])
                    linkages.append(linkage)
                    logger.info(
                        f"Covalent linkage detected in {system.system_id}: {linkage}"
                    )
        return linkages

    def process_ligands(
        self, system: PlinderSystem, ligand_mols: Dict[str, Chem.rdchem.Mol]
    ) -> Tuple[
        Dict[str, LigandData], Dict[str, PharmacophoreData], Dict[str, Chem.rdchem.Mol]
    ]:
        keys = list(ligand_mols.keys())
        mols = list(ligand_mols.values())

        receptor_mol = Chem.MolFromPDBFile(system.receptor_pdb)

        (xace_mols, failed_idxs, failure_counts, tcv_counts) = (
            self.ligand_tensorizer.featurize_molecules(mols)
        )
        annotation = system.system
        failed_mols = {}
        for i in failed_idxs:
            failed_mols[keys[i]] = ligand_mols[keys[i]]
            logger.warning("Failed to tensorize ligand %s", keys[i])

        ligand_keys = [key for i, key in enumerate(keys) if i not in failed_idxs]

        ligands_data = {}
        pharmacophores_data = {}
        for i, key in enumerate(ligand_keys):
            raw_sdf = Path(system.ligand_sdfs[key]).relative_to(self.raw_data)
            instance, asym_id = key.split(".")
            is_covalent = False
            linkages = None
            ccd = None
            for lig_ann in annotation["ligands"]:
                if (
                    int(lig_ann["instance"]) == int(instance)
                    and lig_ann["asym_id"] == asym_id
                ):
                    ccd = lig_ann["ccd_code"]
                    is_covalent = lig_ann["is_covalent"]
                    if is_covalent:
                        linkages = lig_ann["covalent_linkages"]
                    else:
                        inferred_linkages = self.infer_covalent_linkages(
                            system=system, ligand_id=key
                        )
                        if inferred_linkages:
                            is_covalent = True
                            linkages = inferred_linkages

            P, X, V, I = get_pharmacophores(mol=ligand_mols[key], rec=receptor_mol)
            if not np.isfinite(V).all():
                logger.warning(
                    f"Non-finite pharmacophore vectors found in system {system.system_id} ligand {key}"
                )
                failed_mols[key] = ligand_mols[key]
                continue
            if len(I) != len(P):
                logger.warning(
                    f"Length mismatch with interactions {len(I)} and pharm centers {len(P)} in system {system.system_id} ligand {key}"
                )
                failed_mols[key] = ligand_mols[key]
                continue

            pharmacophores_data[key] = PharmacophoreData(
                coords=P, types=X, vectors=V, interactions=I
            )

            ligands_data[key] = LigandData(
                sdf=str(raw_sdf),
                ccd=ccd,
                coords=np.array(xace_mols[i].positions, dtype=np.float32),
                atom_types=xace_mols[i].atom_types,
                atom_charges=xace_mols[i].atom_charges,
                bond_types=xace_mols[i].bond_types,
                bond_indices=xace_mols[i].bond_idxs,
                is_covalent=is_covalent,
                linkages=linkages,
            )

        return (ligands_data, pharmacophores_data, failed_mols)

    def process_npndes(
        self, system: PlinderSystem, npnde_mols: Dict[str, Chem.rdchem.Mol]
    ) -> Dict[str, LigandData]:
        keys = list(npnde_mols.keys())
        mols = list(npnde_mols.values())

        (xace_mols, failed_idxs, failure_counts, tcv_counts) = (
            self.npnde_tensorizer.featurize_molecules(mols)
        )

        annotation = system.system

        for i in failed_idxs:
            logger.warning("Failed to tensorize npnde %s", keys[i])

        npnde_keys = [key for i, key in enumerate(keys) if i not in failed_idxs]

        npnde_data = {}
        for i, key in enumerate(npnde_keys):
            raw_sdf = Path(system.ligand_sdfs[key]).relative_to(self.raw_data)

            instance, asym_id = key.split(".")
            is_covalent = False
            linkages = None
            ccd = None
            for lig_ann in annotation["ligands"]:
                if (
                    int(lig_ann["instance"]) == int(instance)
                    and lig_ann["asym_id"] == asym_id
                ):
                    ccd = lig_ann["ccd_code"]
                    is_covalent = lig_ann["is_covalent"]
                    if is_covalent:
                        linkages = lig_ann["covalent_linkages"]
                    else:
                        inferred_linkages = self.infer_covalent_linkages(
                            system=system, ligand_id=key
                        )
                        if inferred_linkages:
                            is_covalent = True
                            linkages = inferred_linkages

            npnde_data[key] = LigandData(
                sdf=str(raw_sdf),
                ccd=ccd,
                coords=np.array(xace_mols[i].positions, dtype=np.float32),
                atom_types=xace_mols[i].atom_types,
                atom_charges=xace_mols[i].atom_charges,
                bond_types=xace_mols[i].bond_types,
                bond_indices=xace_mols[i].bond_idxs,
                is_covalent=is_covalent,
                linkages=linkages,
            )

        return npnde_data

    def convert_npnde_map(self, ligand: LigandData) -> LigandData:
        atom_types = [self.ligand_atom_map[i] for i in ligand.atom_types]
        new_atom_types = [self.npnde_atom_map.index(atom) for atom in atom_types]
        npnde = LigandData(
            sdf=ligand.sdf,
            ccd=ligand.ccd,
            coords=ligand.coords,
            atom_types=np.array(new_atom_types, dtype=np.int32),
            atom_charges=ligand.atom_charges,
            bond_types=ligand.bond_types,
            bond_indices=ligand.bond_indices,
            is_covalent=ligand.is_covalent,
            linkages=ligand.linkages,
        )
        return npnde

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
            elements=receptor.element[pocket_indices],
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
        link_type: str = None,
        raw_data: str = "/net/galaxy/home/koes/tjkatz/.local/share/plinder/2024-06/v2",
    ):
        logger.debug("Initializing SystemProcessor with cutoff=%f", pocket_cutoff)
        self.structure_processor = StructureProcessor(
            ligand_atom_map=ligand_atom_map,
            npnde_atom_map=npnde_atom_map,
            pocket_cutoff=pocket_cutoff,
            raw_data=raw_data,
        )
        self.link_type = link_type
        self.pdb_writer = None

    def filter_ligands(
        self, system: PlinderSystem
    ) -> (Dict[str, Chem.rdchem.Mol], Dict[str, Chem.rdchem.Mol]):
        ligand_mols, npnde_mols = filter(system.system_id)
        return ligand_mols, npnde_mols

    def process_system(
        self, system_id: str, save_pockets: bool = False
    ) -> Dict[str, Any]:
        logger.info("Processing system %s", system_id)

        system = PlinderSystem(system_id=system_id)

        ligand_mols, npnde_mols = self.filter_ligands(system)

        if not ligand_mols:
            return None

        if self.link_type == "apo":
            # Get apo ids
            apo_ids = system.linked_structures[
                system.linked_structures["kind"] == "apo"
            ]["id"].tolist()

            if not apo_ids:
                logger.warning(
                    f"Skipping system {system_id} due to no linked apo structures"
                )
                return None

        elif self.link_type == "pred":
            # Get pred ids
            pred_ids = system.linked_structures[
                system.linked_structures["kind"] == "pred"
            ]["id"].tolist()

            if not pred_ids:
                logger.warning(
                    f"Skipping system {system_id} due to no linked pred structures"
                )
                return None

        elif self.link_type is not None:
            raise NotImplementedError("link_type must be None, apo, or pred")

        result = self.process_structures(
            system_id=system_id,
            system=system,
            ligand_mols=ligand_mols,
            npnde_mols=npnde_mols,
            apo_ids=apo_ids if self.link_type == "apo" else None,
            pred_ids=pred_ids if self.link_type == "pred" else None,
            chain_mapping=system.chain_mapping,
            save_pockets=save_pockets,
        )

        if not result:
            logger.warning("Skipping system %s due to no ligands remaining", system_id)
            return None

        return result

    def process_linked_pair(
        self,
        system_id: str,
        system: PlinderSystem,
        ligand_data: Dict[str, LigandData],
        pharmacophore_data: Dict[str, PharmacophoreData],
        link_id: str,
        link_type: str,
        npnde_data: Optional[Dict[str, LigandData]] = None,
        chain_mapping: Optional[Dict[str, str]] = None,
    ) -> List[SystemData]:
        annotation = system.system
        has_covalent = False
        for lig_ann in annotation["ligands"]:
            if lig_ann["is_covalent"]:
                has_covalent = True

        receptor_data, res_id_mapping = (
            self.structure_processor.process_linked_structure(
                system=system,
                linked_id=link_id,
                has_covalent=has_covalent,
                ligand_data=ligand_data,
            )
        )

        if not receptor_data:
            logger.warning(f"Failed to align/crop {system_id} with {link_id}")
            return None
        elif not receptor_data["pockets"]:
            logger.warning(f"No pockets extracted for {system_id} with {link_id}")
            return None

        system_datas = []

        for key, ligand in ligand_data.items():
            if key not in receptor_data["pockets"]:
                continue
            lig_instance, lig_asym_id = key.split(".")
            other_ligands = {k: l for k, l in ligand_data.items() if k != key}
            if other_ligands:
                for k, l in other_ligands.items():
                    other_ligands[k] = self.structure_processor.convert_npnde_map(l)

            if npnde_data:
                temp_npnde_data = npnde_data.copy()
            else:
                temp_npnde_data = {}
            temp_npnde_data.update(other_ligands)
            # TODO: update linkages with new residue numbering
            # "{auth_resid}:{resname}{assym_id}{seq_resid}{atom_name}__{auth_resid}:{resname}{assym_id}{seq_resid}{atom_name}"
            # 'covalent_linkages': ['11:CYS:A:11:SG__86:GSH:B:.:SG2']
            if ligand.is_covalent:
                updated_linkages = []
                rec_chains = set(system.sequences.keys())
                for linkage in ligand.linkages:
                    prtnr1, prtnr2 = linkage.split("__")
                    updated_linkage = None
                    lig_identifier = f"{ligand.ccd}:{lig_asym_id}"
                    if lig_identifier in prtnr1 and lig_identifier in prtnr2:
                        logger.warning(
                            f"Failed to update linkage in system {system_id} ligand {key}"
                        )
                        return None
                    elif lig_identifier in prtnr1:
                        (
                            rec_auth_resid,
                            rec_resname,
                            rec_asym_id,
                            rec_seq_resid,
                            rec_atom_name,
                        ) = prtnr2.split(":")
                        (
                            lig_auth_resid,
                            lig_resname,
                            lig_asym_id,
                            lig_seq_resid,
                            lig_atom_name,
                        ) = prtnr1.split(":")
                        for chain in rec_chains:
                            if rec_asym_id == chain.split(".")[1]:
                                chain_id = chain
                        chain_res_map = res_id_mapping.get(chain_id)
                        if chain_res_map is None:
                            logger.warning(
                                f"Failure in system {system_id} {link_type} {link_id} : chain {chain_id} not in res_id_mapping, og_linkage: {linkage}"
                            )
                            return None

                        old_id = rec_seq_resid
                        rec_seq_resid = chain_res_map.get(int(rec_seq_resid))
                        if rec_seq_resid is None:
                            logger.warning(
                                f"Failure in system {system_id} {link_type} {link_id} : res id {old_id} not in res_id_mapping, og_linkage: {linkage}"
                            )
                            return None
                        prtnr1 = ":".join(
                            [
                                lig_auth_resid,
                                lig_resname,
                                lig_asym_id,
                                lig_seq_resid,
                                lig_atom_name,
                            ]
                        )
                        prtnr2 = ":".join(
                            [
                                rec_auth_resid,
                                rec_resname,
                                rec_asym_id,
                                str(rec_seq_resid),
                                rec_atom_name,
                            ]
                        )
                        updated_linkage = "__".join([prtnr1, prtnr2])
                    elif lig_identifier in prtnr2:
                        (
                            rec_auth_resid,
                            rec_resname,
                            rec_asym_id,
                            rec_seq_resid,
                            rec_atom_name,
                        ) = prtnr1.split(":")
                        (
                            lig_auth_resid,
                            lig_resname,
                            lig_asym_id,
                            lig_seq_resid,
                            lig_atom_name,
                        ) = prtnr2.split(":")
                        for chain in rec_chains:
                            if rec_asym_id == chain.split(".")[1]:
                                chain_id = chain
                        chain_res_map = res_id_mapping.get(chain_id)
                        if chain_res_map is None:
                            logger.warning(
                                f"Failure in system {system_id} {link_type} {link_id} : chain {chain_id} not in res_id_mapping, og_linkage: {linkage}"
                            )
                            return None

                        old_id = rec_seq_resid
                        rec_seq_resid = chain_res_map.get(int(rec_seq_resid))
                        if rec_seq_resid is None:
                            logger.warning(
                                f"Failure in system {system_id} {link_type} {link_id} : res id {old_id} not in res_id_mapping, og_linkage: {linkage}"
                            )
                            return None
                        prtnr2 = ":".join(
                            [
                                lig_auth_resid,
                                lig_resname,
                                lig_asym_id,
                                lig_seq_resid,
                                lig_atom_name,
                            ]
                        )
                        prtnr1 = ":".join(
                            [
                                rec_auth_resid,
                                rec_resname,
                                rec_asym_id,
                                str(rec_seq_resid),
                                rec_atom_name,
                            ]
                        )
                        updated_linkage = "__".join([prtnr1, prtnr2])
                    else:
                        logger.warning(
                            f"Failed to update linkage in system {system_id} ligand {key}"
                        )
                        return None
                    if updated_linkage:
                        updated_linkages.append(updated_linkage)
                    else:
                        updated_linkages.append(linkage)
                ligand.linkages = updated_linkages

            system_data = SystemData(
                system_id=system_id,
                ligand_id=key,
                receptor=receptor_data["holo"],
                ligand=ligand,
                pharmacophore=pharmacophore_data.get(key),
                pocket=receptor_data["pockets"][key],
                npndes=temp_npnde_data if temp_npnde_data else None,
                link_type=link_type,
                link_id=link_id,
                link=receptor_data[link_id] if link_type else None,
            )
            system_datas.append(system_data)
        return system_datas

    def process_structures_no_links(
        self,
        system_id: str,
        system: PlinderSystem,
        ligand_data: Dict[str, LigandData],
        pharmacophore_data: Dict[str, PharmacophoreData],
        npnde_data: Optional[Dict[str, LigandData]] = None,
        chain_mapping: Optional[Dict[str, str]] = None,
        save_pockets: bool = False,
    ) -> List[SystemData]:
        if save_pockets:
            self.pdb_writer = PDBWriter(chain_mapping)

        system_structure = system.holo_structure

        # Process receptor
        receptor_data = self.structure_processor.process_receptor(
            system_structure.protein_atom_array,
            str(system_structure.protein_path),
            chain_mapping,
        )

        # Process pockets
        pockets_data = {}
        ligands_to_remove = []
        for ligand_key, ligand in ligand_data.items():
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
            del ligand_data[ligand_key]

        if len(ligand_data) < 1:
            return None

        system_datas = []

        for key, ligand in ligand_data.items():
            other_ligands = {k: l for k, l in ligand_data.items() if k != key}
            if other_ligands:
                for k, l in other_ligands.items():
                    other_ligands[k] = self.structure_processor.convert_npnde_map(l)

            if npnde_data:
                temp_npnde_data = npnde_data.copy()
            else:
                temp_npnde_data = {}
            temp_npnde_data.update(other_ligands)

            system_data = SystemData(
                system_id=system_id,
                ligand_id=key,
                receptor=receptor_data,
                ligand=ligand,
                pharmacophore=pharmacophore_data.get(key),
                pocket=pockets_data[key],
                npndes=temp_npnde_data if temp_npnde_data else None,
            )
            system_datas.append(system_data)
        return system_datas

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
        # Process ligands
        ligands_data, pharmacophores_data, failed_mols = (
            self.structure_processor.process_ligands(system, ligand_mols)
        )

        if failed_mols:
            npnde_mols.update(failed_mols)

        if npnde_mols:
            npnde_data = self.structure_processor.process_npndes(system, npnde_mols)

        if not self.link_type:
            systems_data = self.process_structures_no_links(
                system_id=system_id,
                system=system,
                ligand_data=ligands_data,
                pharmacophore_data=pharmacophores_data,
                npnde_data=npnde_data if npnde_mols else None,
                chain_mapping=chain_mapping,
                save_pockets=save_pockets,
            )
            if systems_data:
                return {
                    system_id: systems_data,
                    "links": False,
                    "annotation": system.system,
                }
            else:
                return None
        else:
            systems_data = {}
            links = False
            apos = {}
            if self.link_type == "apo":
                for apo_id in apo_ids:
                    systems_list = self.process_linked_pair(
                        system_id=system_id,
                        system=system,
                        ligand_data=ligands_data,
                        pharmacophore_data=pharmacophores_data,
                        npnde_data=npnde_data if npnde_mols else None,
                        link_id=apo_id,
                        link_type="apo",
                        chain_mapping=chain_mapping,
                    )
                    if systems_list:
                        links = True
                        apos[apo_id] = systems_list
                if apos:
                    systems_data["apo"] = apos
            preds = {}
            if self.link_type == "pred":
                for pred_id in pred_ids:
                    systems_list = self.process_linked_pair(
                        system_id=system_id,
                        system=system,
                        ligand_data=ligands_data,
                        pharmacophore_data=pharmacophores_data,
                        npnde_data=npnde_data if npnde_mols else None,
                        link_id=pred_id,
                        link_type="pred",
                        chain_mapping=chain_mapping,
                    )
                    if systems_list:
                        links = True
                        preds[pred_id] = systems_list
                if preds:
                    systems_data["pred"] = preds
            if systems_data:
                systems_data["links"] = links
                systems_data["annotation"] = system.system
                return systems_data
            else:
                return None
