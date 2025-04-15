import logging
import os
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

import biotite
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np
from biotite.structure.io.pdbx import CIFFile, get_structure
from omtra.data.plinder import (
    LigandData,
    PharmacophoreData,
    StructureData,
    SystemData,
    BackboneData,
)
from omtra.data.pharmacophores import get_pharmacophores
from omtra.data.xace_ligand import MoleculeTensorizer
from omtra.utils.misc import bad_mol_reporter
from omtra.constants import lig_atom_type_map, npnde_atom_type_map, aa_substitutions, residue_to_single
from omtra_pipelines.plinder_dataset.utils import _DEFAULT_DISTANCE_RANGE, setup_logger
from omtra_pipelines.plinder_dataset.filter import filter
from plinder.core import PlinderSystem
from rdkit import Chem

import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    LogitsConfig,
    LogitsOutput,
    )
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.structure.protein_complex import ProteinComplex

EMBEDDING_CONFIG = LogitsConfig(
    sequence=False, return_embeddings=True, return_hidden_states=False
)
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


class SystemProcessor:
    def __init__(
        self,
        system_id: str,
        link_type: Optional[str] = None,
        ligand_atom_map: List[str] = lig_atom_type_map,
        npnde_atom_map: List[str] = npnde_atom_type_map,
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

        self.system_id = system_id
        self.system = PlinderSystem(system_id=self.system_id)

        self.link_type = link_type
        self.pdb_writer = None

    def extract_backbone(
        self,
        backbone: struc.AtomArray,
    ) -> BackboneData:
        compound_keys = np.array(
            [f"{chain}_{res}" for chain, res in zip(backbone.chain_id, backbone.res_id)]
        )
        unique_compound_keys = np.unique(compound_keys)
        num_residues = len(unique_compound_keys)

        coords = np.zeros((num_residues, 3, 3))
        res_ids = np.zeros(num_residues, dtype=int)
        res_names_list = []
        chain_ids_list = []

        for i, compound_key in enumerate(unique_compound_keys):
            chain_id, res_id = compound_key.split("_")
            res_id = int(res_id)

            res_mask = (backbone.chain_id == chain_id) & (backbone.res_id == res_id)
            res_atoms = backbone[res_mask]

            res_ids[i] = res_id
            res_names_list.append(res_atoms.res_name[0])
            chain_ids_list.append(chain_id)

            for j, atom_name in enumerate(["N", "CA", "C"]):
                atom_mask = res_atoms.atom_name == atom_name
                if np.any(atom_mask):
                    coords[i, j] = res_atoms.coord[atom_mask][0]
                else:
                    logger.warning(f"Error with {self.system_id} backbone extraction")
                    return None

        res_names = np.array(res_names_list)
        chain_ids = np.array(chain_ids_list)

        backbone_data = BackboneData(
            coords=coords,
            res_ids=res_ids,
            res_names=res_names,
            chain_ids=chain_ids,
        )
        return backbone_data

    def process_receptor(
        self,
        receptor: struc.AtomArray,
        cif: str,
        chain_mapping: Optional[Dict[str, str]] = None,
    ) -> StructureData:
        receptor = receptor[receptor.res_name != "HOH"]
        receptor = receptor[receptor.element != "H"]

        raw_cif = Path(cif).relative_to(self.raw_data)

        if chain_mapping is not None:
            chain_ids = [chain_mapping.get(chain, chain) for chain in receptor.chain_id]
            receptor.chain_id = chain_ids

        backbone = receptor[struc.filter_peptide_backbone(receptor)]
        backbone_data = self.extract_backbone(backbone)
        if backbone_data is None:
            return None

        receptor = self.check_backbone_order(receptor)
        if receptor is None:
            return None

        bb_mask = struc.filter_peptide_backbone(receptor)
        return StructureData(
            cif=str(raw_cif),
            coords=receptor.coord,
            atom_names=receptor.atom_name,
            elements=receptor.element,
            res_ids=receptor.res_id,
            res_names=receptor.res_name,
            chain_ids=receptor.chain_id,
            backbone_mask=bb_mask,
            backbone=backbone_data,
        )

    def check_backbone_order(self, receptor: struc.AtomArray) -> struc.AtomArray:
        unique_residues = list(set(zip(receptor.chain_id, receptor.res_id)))
        reordering_needed = False

        for chain_id, res_id in unique_residues:
            residue_mask = (receptor.chain_id == chain_id) & (receptor.res_id == res_id)
            residue_atoms = receptor[residue_mask]

            n_indices = np.where(residue_atoms.atom_name == "N")[0]
            ca_indices = np.where(residue_atoms.atom_name == "CA")[0]
            c_indices = np.where(residue_atoms.atom_name == "C")[0]

            if len(n_indices) == 0 or len(ca_indices) == 0 or len(c_indices) == 0:
                continue

            full_indices = np.where(residue_mask)[0]
            n_idx = full_indices[n_indices[0]]
            ca_idx = full_indices[ca_indices[0]]
            c_idx = full_indices[c_indices[0]]

            if not (n_idx < ca_idx < c_idx):
                reordering_needed = True
                break

        if reordering_needed:
            logger.warning(f"System {self.system_id} requires backbone atom reordering")
            return self.reorder_backbone_atoms(receptor, unique_residues)
        else:
            return receptor

    def reorder_backbone_atoms(
        self, receptor: struc.AtomArray, unique_residues
    ) -> struc.AtomArray:
        reordered_atoms = []

        for chain_id, res_id in unique_residues:
            residue_mask = (receptor.chain_id == chain_id) & (receptor.res_id == res_id)
            residue_atoms = receptor[residue_mask]

            n_mask = residue_atoms.atom_name == "N"
            ca_mask = residue_atoms.atom_name == "CA"
            c_mask = residue_atoms.atom_name == "C"
            backbone_mask = n_mask | ca_mask | c_mask

            n_idx = np.where(n_mask)[0][0] if np.any(n_mask) else -1
            ca_idx = np.where(ca_mask)[0][0] if np.any(ca_mask) else -1
            c_idx = np.where(c_mask)[0][0] if np.any(c_mask) else -1

            new_order = []

            if n_idx > 0:
                new_order.extend(list(range(n_idx)))

            if n_idx != -1:
                new_order.append(n_idx)

            if n_idx != -1 and ca_idx != -1:
                for i in range(n_idx + 1, ca_idx):
                    if not backbone_mask[i]:
                        new_order.append(i)

            if ca_idx != -1:
                new_order.append(ca_idx)

            if ca_idx != -1 and c_idx != -1:
                for i in range(ca_idx + 1, c_idx):
                    if not backbone_mask[i]:
                        new_order.append(i)

            if c_idx != -1:
                new_order.append(c_idx)

            if c_idx != -1 and c_idx < len(residue_atoms) - 1:
                new_order.extend(range(c_idx + 1, len(residue_atoms)))

            for idx in new_order:
                reordered_atoms.append(residue_atoms[idx])

        if reordered_atoms:
            return struc.stack(reordered_atoms)
        else:
            logger.warning(f"Failed to reorder backbone {self.system_id}")
            return None

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
        link_id: str,
        receptor: struc.AtomArray,
        linked_structure: struc.AtomArray,
    ) -> struc.AtomArray:
        receptor_keys = [self.create_atom_key(atom) for atom in receptor]
        linked_keys = [self.create_atom_key(atom) for atom in linked_structure]

        if set(receptor_keys) != set(linked_keys):
            logger.warning(
                f"Atom key set mismatch between receptor and linked structure {self.system_id}_{link_id}"
            )
            return None

        reorder_indices = []
        for key in receptor_keys:
            idx = linked_keys.index(key)
            reorder_indices.append(idx)

        if len(reorder_indices) != len(set(reorder_indices)):
            logger.warning(f"Failed reordering for {self.system_id}_{link_id}")
            return None

        reordered = linked_structure[reorder_indices]

        return reordered

    def process_linked_structure(
        self,
        linked_id: str,
        ligand_data: Dict[str, LigandData],
        has_covalent: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[int, int]]:
        holo = self.system.holo_structure
        linked_structure = self.system.alternate_structures[linked_id]
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
            self.system.chain_mapping,
        )
        linked_data = self.process_receptor(
            linked_cropped_superposed.protein_atom_array,
            str(linked_cropped_superposed.protein_path),
            self.system.chain_mapping,
        )
        if holo_data is None or linked_data is None:
            return None, None
        pockets_data = {}
        for key, ligand in ligand_data.items():
            pocket = self.extract_pocket(
                receptor=holo_cropped.protein_atom_array,
                ligand_coords=ligand.coords,
                chain_mapping=self.system.chain_mapping,
            )
            if pocket:
                pockets_data[key] = pocket
        return {
            "holo": holo_data,
            linked_id: linked_data,
            "pockets": pockets_data,
        }, res_id_map

    def infer_covalent_linkages(self, ligand_id: str) -> List[str]:
        system_cif = CIFFile.read(self.system.system_cif)
        system_struc = get_structure(system_cif, model=1, include_bonds=True)
        ligand = system_struc[system_struc.chain_id == ligand_id]

        receptor_cif = CIFFile.read(self.system.receptor_cif)
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
                        f"Covalent linkage detected in {self.system_id}: {linkage}"
                    )
        return linkages

    def process_ligands(
        self, ligand_mols: Dict[str, Chem.rdchem.Mol]
    ) -> Tuple[
        Dict[str, LigandData], Dict[str, PharmacophoreData], Dict[str, Chem.rdchem.Mol]
    ]:
        keys = list(ligand_mols.keys())
        mols = list(ligand_mols.values())

        receptor_mol = Chem.MolFromPDBFile(self.system.receptor_pdb)
        if not receptor_mol:
            receptor_mol = Chem.MolFromPDBFile(self.system.receptor_pdb, sanitize=False)

        (xace_mols, failed_idxs, failure_counts, tcv_counts) = (
            self.ligand_tensorizer.featurize_molecules(mols)
        )
        annotation = self.system.system
        failed_mols = {}
        for i in failed_idxs:
            failed_mols[keys[i]] = ligand_mols[keys[i]]
            logger.warning("Failed to tensorize ligand %s", keys[i])

        ligand_keys = [key for i, key in enumerate(keys) if i not in failed_idxs]

        ligands_data = {}
        pharmacophores_data = {}
        for i, key in enumerate(ligand_keys):
            raw_sdf = Path(self.system.ligand_sdfs[key]).relative_to(self.raw_data)
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
                        inferred_linkages = self.infer_covalent_linkages(ligand_id=key)
                        if inferred_linkages:
                            is_covalent = True
                            linkages = inferred_linkages

            P, X, V, I = get_pharmacophores(mol=ligand_mols[key], rec=receptor_mol)
            if not np.isfinite(V).all():
                logger.warning(
                    f"Non-finite pharmacophore vectors found in system {self.system_id} ligand {key}"
                )
                bad_mol_reporter(
                    ligand_mols[key],
                    note="Pharmacophore vectors contain non-finite values",
                )
                failed_mols[key] = ligand_mols[key]
                continue
            if len(I) != len(P):
                logger.warning(
                    f"Length mismatch with interactions {len(I)} and pharm centers {len(P)} in system {self.system_id} ligand {key}"
                )
                bad_mol_reporter(
                    ligand_mols[key], note="Length mismatch interactions/pharm centers"
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
        self, npnde_mols: Dict[str, Chem.rdchem.Mol]
    ) -> Dict[str, LigandData]:
        keys = list(npnde_mols.keys())
        mols = list(npnde_mols.values())

        (xace_mols, failed_idxs, failure_counts, tcv_counts) = (
            self.npnde_tensorizer.featurize_molecules(mols)
        )

        annotation = self.system.system

        for i in failed_idxs:
            logger.warning("Failed to tensorize npnde %s", keys[i])

        npnde_keys = [key for i, key in enumerate(keys) if i not in failed_idxs]

        npnde_data = {}
        for i, key in enumerate(npnde_keys):
            raw_sdf = Path(self.system.ligand_sdfs[key]).relative_to(self.raw_data)

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
                        inferred_linkages = self.infer_covalent_linkages(ligand_id=key)
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


    def embed_protein_complex(self, model: ESM3InferenceClient, protein_complex: ProteinComplex) -> np.ndarray:
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model =  model.to(device)

        protein = ESMProtein.from_protein_complex(protein_complex)
        protein_tensor = model.encode(protein)
        output = model.logits(protein_tensor, EMBEDDING_CONFIG)
        if device == torch.device("cuda"):
            model.to(torch.device("cpu"))
        return output.embeddings.cpu().numpy()

    def embed_chain(self, model: ESM3InferenceClient, protein_chain: ProteinChain) -> np.ndarray:
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model =  model.to(device)

        protein = ESMProtein.from_protein_chain(protein_chain)
        protein_tensor = model.encode(protein)
        output = model.logits(protein_tensor, EMBEDDING_CONFIG)
        if device == torch.device("cuda"):
            model.to(torch.device("cpu"))
        return output.embeddings.cpu().numpy()


    def ESM3_embed(self, res_name:np.ndarray, chain_id:np.ndarray, backbone_data: np.ndarray, bb_mask: np.ndarray[bool]) -> np.ndarray:  
        
        model = ESM3.from_pretrained("esm3-open", device=torch.device("cpu")) 
        
        residue_names = res_name[bb_mask]
        chain_ids = chain_id[bb_mask]
        coords = backbone_data

        # check if we need to split pocket sequence by chain_id to concatenate for protein_complex
        if len(set(chain_ids)) > 1:
            unique_chain_id = set()
            unique_chain_id = [chain for chain in chain_ids if chain not in unique_chain_id]
            chain_mask = []
            split_seq = []
            for chain in unique_chain_id:
                chain_mask.append(np.where(chain_ids == chain)[0])
                split_seq.append(residue_names[chain_mask[-1]])
        else:
            chain_mask, split_seq = None, None


        if split_seq:
            concat_seq = []
            esm_chains = []
            layers = list(coords)
            for seq in range(len(split_seq)):
                temp = []
                if len(layers) == 0: 
                    break 
                for i in range(0, len(seq),3):
                    if residue_names[i] not in aa_substitutions:
                        temp.append(residue_to_single[seq[i]]) 
                    else: 
                        try: 
                            temp.append(aa_substitutions[seq[i]]) 
                        except: 
                            temp.append(residue_to_single['UNK'])
                concat_seq.append(temp)

                esm_chains.append(ProteinChain.from_backbone_atom_coordinates(layers[0:len(temp)], sequence=temp))
                layers = layers[len(temp)+1:]

            concat_seq = '|'.join(concat_seq)
            protein_complex = ProteinComplex.from_chains(esm_chains)
            return self.embed_protein_complex(model, protein_complex)

        else:
            sequence = []
            for i in range(0, len(residue_names),3):
                if residue_names[i] not in aa_substitutions:
                    sequence.append(residue_to_single[residue_names[i]]) 
                else: 
                    try: 
                        sequence.append(aa_substitutions[residue_names[i]]) 
                    except: 
                        sequence.append(residue_to_single['UNK'])
            
            chain_seq = ''.join(sequence)
            chain = ProteinChain.from_backbone_atom_coordinates(coords, sequence=chain_seq)
            return self.embed_chain(model, chain)

    def extract_pocket(
        self,
        receptor: struc.AtomArray,
        ligand_coords: np.ndarray,
        chain_mapping: Optional[Dict[str, str]] = None,
    ) -> StructureData:
        logger.debug("Extracting pocket")

        receptor = receptor[receptor.res_name != "HOH"]
        receptor = receptor[receptor.element != "H"]
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

        pocket = receptor[pocket_indices]
        backbone = pocket[struc.filter_peptide_backbone(pocket)]
        backbone_data = self.extract_backbone(backbone)

        pocket = self.check_backbone_order(pocket)
        if pocket is None:
            return None

        bb_mask = struc.filter_peptide_backbone(pocket)

        embedding = self.ESM3_embed(pocket.res_name, pocket.chain_id, backbone_data.coords, bb_mask)

        return StructureData(
            coords=pocket.coord,
            atom_names=pocket.atom_name,
            elements=pocket.element,
            res_ids=pocket.res_id,  # original residue ids
            res_names=pocket.res_name,
            chain_ids=pocket.chain_id,
            backbone_mask=bb_mask,
            backbone=backbone_data,
            pocket_embedding=embedding,
        )

    def filter_ligands(
        self,
    ) -> Tuple[Dict[str, Chem.rdchem.Mol], Dict[str, Chem.rdchem.Mol]]:
        # ligand_mols, npnde_mols = filter(self.system_id)
        filter_parquet = Path(
            "/net/galaxy/home/koes/tjkatz/OMTRA/omtra_pipelines/plinder_dataset/plinder_filtered.parquet"
        )
        df = pd.read_parquet(filter_parquet)

        system_df = df[df["system_id"] == self.system_id]

        system_structure = self.system.holo_structure
        ligand_mols = {}
        npnde_mols = {}

        ligand_rows = system_df[system_df["ligand_type"] == "ligand"]
        for _, row in ligand_rows.iterrows():
            ligand_id = row["ligand_id"]
            if ligand_id in system_structure.resolved_ligand_mols:
                ligand_mols[ligand_id] = system_structure.resolved_ligand_mols[
                    ligand_id
                ]

        npnde_rows = system_df[system_df["ligand_type"] == "npnde"]
        for _, row in npnde_rows.iterrows():
            ligand_id = row["ligand_id"]
            if ligand_id in system_structure.resolved_ligand_mols:
                npnde_mols[ligand_id] = system_structure.resolved_ligand_mols[ligand_id]

        return ligand_mols, npnde_mols

    def process_system(self, save_pockets: bool = False) -> Dict[str, Any]:
        logger.info("Processing system %s", self.system_id)

        ligand_mols, npnde_mols = self.filter_ligands()

        if not ligand_mols:
            return None

        if self.link_type == "apo":
            # Get apo ids
            apo_ids = self.system.linked_structures[
                self.system.linked_structures["kind"] == "apo"
            ]["id"].tolist()

            if not apo_ids:
                logger.warning(
                    f"Skipping system {self.system_id} due to no linked apo structures"
                )
                return None

        elif self.link_type == "pred":
            # Get pred ids
            pred_ids = self.system.linked_structures[
                self.system.linked_structures["kind"] == "pred"
            ]["id"].tolist()

            if not pred_ids:
                logger.warning(
                    f"Skipping system {self.system_id} due to no linked pred structures"
                )
                return None

        elif self.link_type is not None:
            raise NotImplementedError("link_type must be None, apo, or pred")

        result = self.process_structures(
            ligand_mols=ligand_mols,
            npnde_mols=npnde_mols,
            apo_ids=apo_ids if self.link_type == "apo" else None,
            pred_ids=pred_ids if self.link_type == "pred" else None,
            chain_mapping=self.system.chain_mapping,
            save_pockets=save_pockets,
        )

        if not result:
            logger.warning(
                "Skipping system %s due to no ligands remaining", self.system_id
            )
            return None

        return result

    def process_linked_pair(
        self,
        ligand_data: Dict[str, LigandData],
        pharmacophore_data: Dict[str, PharmacophoreData],
        link_id: str,
        link_type: str,
        npnde_data: Optional[Dict[str, LigandData]] = None,
        chain_mapping: Optional[Dict[str, str]] = None,
    ) -> List[SystemData]:
        annotation = self.system.system
        has_covalent = False
        for lig_ann in annotation["ligands"]:
            if lig_ann["is_covalent"]:
                has_covalent = True
        for ligand in ligand_data.values():
            if ligand.is_covalent:
                has_covalent = True

        receptor_data, res_id_mapping = self.process_linked_structure(
            linked_id=link_id,
            has_covalent=has_covalent,
            ligand_data=ligand_data,
        )

        if not receptor_data:
            logger.warning(f"Failed to align/crop {self.system_id} with {link_id}")
            return None
        elif not receptor_data["pockets"]:
            logger.warning(f"No pockets extracted for {self.system_id} with {link_id}")
            return None

        system_datas = []

        for key, ligand in ligand_data.items():
            if key not in receptor_data["pockets"]:
                continue
            lig_instance, lig_asym_id = key.split(".")
            other_ligands = {k: l for k, l in ligand_data.items() if k != key}
            if other_ligands:
                for k, l in other_ligands.items():
                    other_ligands[k] = self.convert_npnde_map(l)

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
                rec_chains = set(self.system.sequences.keys())
                for linkage in ligand.linkages:
                    prtnr1, prtnr2 = linkage.split("__")
                    updated_linkage = None
                    lig_identifier = f"{ligand.ccd}:{lig_asym_id}"
                    if lig_identifier in prtnr1 and lig_identifier in prtnr2:
                        logger.warning(
                            f"Failed to update linkage in system {self.system_id} ligand {key}"
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
                                f"Failure in system {self.system_id} {link_type} {link_id} : chain {chain_id} not in res_id_mapping, og_linkage: {linkage}"
                            )
                            return None

                        old_id = rec_seq_resid
                        rec_seq_resid = chain_res_map.get(int(rec_seq_resid))
                        if rec_seq_resid is None:
                            logger.warning(
                                f"Failure in system {self.system_id} {link_type} {link_id} : res id {old_id} not in res_id_mapping, og_linkage: {linkage}"
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
                                f"Failure in system {self.system_id} {link_type} {link_id} : chain {chain_id} not in res_id_mapping, og_linkage: {linkage}"
                            )
                            return None

                        old_id = rec_seq_resid
                        rec_seq_resid = chain_res_map.get(int(rec_seq_resid))
                        if rec_seq_resid is None:
                            logger.warning(
                                f"Failure in system {self.system_id} {link_type} {link_id} : res id {old_id} not in res_id_mapping, og_linkage: {linkage}"
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
                            f"Failed to update linkage in system {self.system_id} ligand {key}"
                        )
                        return None
                    if updated_linkage:
                        logger.info(
                            f"Updated linkage for {self.system_id} ligand {key}"
                        )
                        updated_linkages.append(updated_linkage)
                    else:
                        updated_linkages.append(linkage)
                ligand.linkages = updated_linkages

            system_data = SystemData(
                system_id=self.system_id,
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
        ligand_data: Dict[str, LigandData],
        pharmacophore_data: Dict[str, PharmacophoreData],
        npnde_data: Optional[Dict[str, LigandData]] = None,
        chain_mapping: Optional[Dict[str, str]] = None,
        save_pockets: bool = False,
    ) -> List[SystemData]:
        if save_pockets:
            self.pdb_writer = PDBWriter(chain_mapping)

        system_structure = self.system.holo_structure

        # Process receptor
        receptor_data = self.process_receptor(
            system_structure.protein_atom_array,
            str(system_structure.protein_path),
            chain_mapping,
        )
        if not receptor_data:
            return None

        # Process pockets
        pockets_data = {}
        ligands_to_remove = []
        for ligand_key, ligand in ligand_data.items():
            pocket_data = self.extract_pocket(
                system_structure.protein_atom_array,
                ligand.coords,
                self.system.chain_mapping,
            )

            if not pocket_data:
                logger.warning(
                    f"No pocket extracted for system {self.system_id} ligand {ligand_key}"
                )
                ligands_to_remove.append(ligand_key)
                continue

            logger.info(f"Extracted pocket for {self.system_id} ligand {ligand_key}")

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
                    other_ligands[k] = self.convert_npnde_map(l)

            if npnde_data:
                temp_npnde_data = npnde_data.copy()
            else:
                temp_npnde_data = {}
            temp_npnde_data.update(other_ligands)

            system_data = SystemData(
                system_id=self.system_id,
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
        ligand_mols: Dict[str, Chem.rdchem.Mol],
        npnde_mols: Optional[Dict[str, Chem.rdchem.Mol]] = None,
        apo_ids: Optional[List[str]] = None,
        pred_ids: Optional[List[str]] = None,
        chain_mapping: Optional[Dict[str, str]] = None,
        save_pockets: bool = False,
    ) -> Dict[str, Any]:
        # Process ligands
        ligands_data, pharmacophores_data, failed_mols = self.process_ligands(
            ligand_mols
        )

        if failed_mols:
            npnde_mols.update(failed_mols)

        if npnde_mols:
            npnde_data = self.process_npndes(npnde_mols)

        if not self.link_type:
            systems_data = self.process_structures_no_links(
                ligand_data=ligands_data,
                pharmacophore_data=pharmacophores_data,
                npnde_data=npnde_data if npnde_mols else None,
                chain_mapping=chain_mapping,
                save_pockets=save_pockets,
            )
            if systems_data:
                return {
                    "systems_list": systems_data,
                    "links": False,
                    "annotation": self.system.system,
                }
            else:
                return None
        else:
            systems_data = {}
            links = False
            apos = {}
            if self.link_type == "apo":
                for apo_id in apo_ids:
                    logger.info(f"Processing {self.system_id} {apo_id}")
                    systems_list = self.process_linked_pair(
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
                    logger.info(f"Processing {self.system_id} {pred_id}")
                    systems_list = self.process_linked_pair(
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
                systems_data["annotation"] = self.system.system
                return systems_data
            else:
                return None
