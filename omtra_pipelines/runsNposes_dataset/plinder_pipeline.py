import logging
import os
import tempfile
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
from omtra.constants import lig_atom_type_map, npnde_atom_type_map, residue_to_single
from omtra_pipelines.runsNposes_dataset.utils import _DEFAULT_DISTANCE_RANGE, setup_logger
from rdkit import Chem

import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    LogitsConfig,
    LogitsOutput,
)

# Global ESM3 model cache — loaded once per worker process, shared across all systems
_esm3_model_cache = None

def _get_esm3_model():
    global _esm3_model_cache
    if _esm3_model_cache is None:
        _esm3_model_cache = ESM3.from_pretrained("esm3-open", device=torch.device("cpu"))
    return _esm3_model_cache
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


class RnPSystem:
    """Lightweight local file reader that replaces PlinderSystem.

    Loads receptor and ligand structures from the ground_truth directory.

    Expected layout:
        ground_truth/{system_id}/
            receptor.cif
            ligand_files/{ligand_id}.sdf
    """

    def __init__(self, system_id: str, ground_truth_dir: str):
        self.system_id = system_id
        self.system_dir = Path(ground_truth_dir) / system_id

        if not self.system_dir.exists():
            raise FileNotFoundError(
                f"System directory not found: {self.system_dir}"
            )

        # Receptor CIF
        self.receptor_cif_path = self.system_dir / "receptor.cif"
        if not self.receptor_cif_path.exists():
            raise FileNotFoundError(
                f"receptor.cif not found in {self.system_dir}"
            )

        # Load receptor structure via biotite
        cif_file = CIFFile.read(str(self.receptor_cif_path))
        self._protein_atom_array = get_structure(cif_file, model=1)
        # Keep only protein residues (filter out water, ligands, etc.)
        aa_mask = struc.filter_amino_acids(self._protein_atom_array)
        self._protein_atom_array = self._protein_atom_array[aa_mask]

        # Discover ligand SDF files
        self.ligand_sdfs: Dict[str, str] = {}
        self.resolved_ligand_mols: Dict[str, Chem.rdchem.Mol] = {}
        ligand_dir = self.system_dir / "ligand_files"
        if ligand_dir.exists():
            for sdf_path in sorted(ligand_dir.glob("*.sdf")):
                ligand_id = sdf_path.stem  # e.g. "1.B"
                self.ligand_sdfs[ligand_id] = str(sdf_path)
                mol = Chem.MolFromMolFile(str(sdf_path), removeHs=True)
                if mol is None:
                    mol = Chem.MolFromMolFile(
                        str(sdf_path), removeHs=True, sanitize=False
                    )
                if mol is not None:
                    self.resolved_ligand_mols[ligand_id] = mol

        # Temporary PDB path (created on demand for RDKit)
        self._receptor_pdb_path: Optional[str] = None

    @property
    def protein_atom_array(self) -> struc.AtomArray:
        return self._protein_atom_array

    @property
    def protein_path(self) -> str:
        return str(self.receptor_cif_path)

    def get_receptor_pdb_path(self) -> str:
        """Write a temporary PDB from the receptor CIF for RDKit loading."""
        if self._receptor_pdb_path is not None and Path(self._receptor_pdb_path).exists():
            return self._receptor_pdb_path

        tmp = tempfile.NamedTemporaryFile(
            suffix=".pdb", prefix=f"rnp_{self.system_id}_", delete=False
        )
        # PDB format only supports single-character chain IDs.
        # Truncate multi-char chain IDs from CIF for RDKit compatibility.
        atom_array = self._protein_atom_array.copy()
        atom_array.chain_id = np.array(
            [cid[:1] for cid in atom_array.chain_id]
        )
        pdb_file = pdb.PDBFile()
        pdb_file.set_structure(atom_array)
        pdb_file.write(tmp.name)
        self._receptor_pdb_path = tmp.name
        return self._receptor_pdb_path

    def cleanup(self):
        """Remove temporary files."""
        if self._receptor_pdb_path and Path(self._receptor_pdb_path).exists():
            os.unlink(self._receptor_pdb_path)
            self._receptor_pdb_path = None


class SystemProcessor:
    def __init__(
        self,
        system_id: str,
        ground_truth_dir: str,
        parquet_df: pd.DataFrame,
        ligand_atom_map: List[str] = lig_atom_type_map,
        npnde_atom_map: List[str] = npnde_atom_type_map,
        pocket_cutoff: float = 8.0,
        n_cpus: int = 1,
    ):
        logger.debug("Initializing SystemProcessor for %s with cutoff=%f", system_id, pocket_cutoff)
        self.ligand_atom_map = ligand_atom_map
        self.npnde_atom_map = npnde_atom_map
        self.pocket_cutoff = pocket_cutoff
        self.ligand_tensorizer = MoleculeTensorizer(
            atom_map=ligand_atom_map, n_cpus=n_cpus
        )
        self.npnde_tensorizer = MoleculeTensorizer(
            atom_map=npnde_atom_map, n_cpus=n_cpus
        )
        self.ground_truth_dir = ground_truth_dir

        self.system_id = system_id
        self.system = RnPSystem(system_id=self.system_id, ground_truth_dir=ground_truth_dir)
        self.parquet_df = parquet_df
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

        coords_list = []
        res_ids_list = []
        res_names_list = []
        chain_ids_list = []
        dropped = 0

        for compound_key in unique_compound_keys:
            chain_id, res_id = compound_key.split("_")
            res_id = int(res_id)

            res_mask = (backbone.chain_id == chain_id) & (backbone.res_id == res_id)
            res_atoms = backbone[res_mask]

            residue_coords = np.zeros((3, 3))
            complete = True
            for j, atom_name in enumerate(["N", "CA", "C"]):
                atom_mask = res_atoms.atom_name == atom_name
                if np.any(atom_mask):
                    residue_coords[j] = res_atoms.coord[atom_mask][0]
                else:
                    complete = False
                    break

            if not complete:
                dropped += 1
                continue

            coords_list.append(residue_coords)
            res_ids_list.append(res_id)
            res_names_list.append(res_atoms.res_name[0])
            chain_ids_list.append(chain_id)

        if dropped > 0:
            logger.warning(
                f"{self.system_id}: dropped {dropped}/{num_residues} incomplete backbone residues"
            )

        if len(coords_list) == 0:
            logger.warning(f"No complete backbone residues in {self.system_id}")
            return None

        coords = np.array(coords_list)
        res_ids = np.array(res_ids_list, dtype=int)
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
    ) -> StructureData:
        receptor = receptor[receptor.res_name != "HOH"]
        receptor = receptor[receptor.element != "H"]

        raw_cif = Path(cif).relative_to(self.ground_truth_dir)

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
        all_reordered_indices = []

        for chain_id, res_id in unique_residues:
            residue_mask = (receptor.chain_id == chain_id) & (receptor.res_id == res_id)
            full_indices = np.where(residue_mask)[0]
            residue_atoms = receptor[residue_mask]

            n_mask = residue_atoms.atom_name == "N"
            ca_mask = residue_atoms.atom_name == "CA"
            c_mask = residue_atoms.atom_name == "C"

            n_idx = np.where(n_mask)[0][0] if np.any(n_mask) else -1
            ca_idx = np.where(ca_mask)[0][0] if np.any(ca_mask) else -1
            c_idx = np.where(c_mask)[0][0] if np.any(c_mask) else -1

            # Swap backbone atoms into N, CA, C order within their existing slots
            # while keeping all non-backbone atoms in their original positions
            bb_slots = sorted([idx for idx in [n_idx, ca_idx, c_idx] if idx != -1])
            desired_atoms = [idx for idx in [n_idx, ca_idx, c_idx] if idx != -1]
            slot_to_desired = dict(zip(bb_slots, desired_atoms))

            new_order = []
            for i in range(len(residue_atoms)):
                if i in slot_to_desired:
                    new_order.append(slot_to_desired[i])
                else:
                    new_order.append(i)

            all_reordered_indices.extend(full_indices[new_order])

        if all_reordered_indices:
            return receptor[all_reordered_indices]
        else:
            logger.warning(f"Failed to reorder backbone {self.system_id}")
            return None

    def infer_covalent_linkages(self, ligand_mol: Chem.rdchem.Mol, ligand_id: str) -> List[str]:
        """Detect covalent bonds between a ligand and the receptor.

        Uses the RDKit conformer for ligand coordinates and the biotite
        receptor atom array, checking inter-atomic distances against
        Allen et al. bond-length ranges.
        """
        conf = ligand_mol.GetConformer()
        lig_coords = conf.GetPositions()
        lig_elements = [atom.GetSymbol().upper() for atom in ligand_mol.GetAtoms()]

        receptor = self.system.protein_atom_array
        rec_coords = receptor.coord
        rec_elements = receptor.element

        linkages = []
        for i in range(len(lig_coords)):
            dists = np.linalg.norm(rec_coords - lig_coords[i], axis=1)
            for j in np.where(dists < 3.0)[0]:  # coarse pre-filter
                pair = tuple(sorted([lig_elements[i], rec_elements[j].upper()]))
                dist_range = _DEFAULT_DISTANCE_RANGE.get(pair)
                if dist_range is None:
                    continue
                min_dist, max_dist = dist_range
                if min_dist <= dists[j] <= max_dist:
                    rec_atom = receptor[j]
                    prtnr1 = (
                        f"{rec_atom.res_id}:{rec_atom.res_name}:"
                        f"{rec_atom.chain_id}:{rec_atom.res_id}:{rec_atom.atom_name}"
                    )
                    lig_atom = ligand_mol.GetAtomWithIdx(i)
                    prtnr2 = (
                        f".:{ligand_id.split('.')[0]}:"
                        f"{ligand_id}:.:{lig_atom.GetSymbol()}"
                    )
                    linkage = "__".join([prtnr1, prtnr2])
                    linkages.append(linkage)
                    logger.info(
                        f"Covalent linkage detected in {self.system_id}: {linkage}"
                    )
        return linkages

    def _get_ccd_for_ligand(self, ligand_id: str) -> Optional[str]:
        """Look up CCD code from the parquet for a given ligand."""
        system_df = self.parquet_df[self.parquet_df["system_id"] == self.system_id]
        row = system_df[system_df["ligand_id"] == ligand_id]
        if len(row) > 0 and "ccd_code" in row.columns:
            val = row.iloc[0]["ccd_code"]
            return val if pd.notna(val) else None
        return None

    def process_ligands(
        self, ligand_mols: Dict[str, Chem.rdchem.Mol]
    ) -> Tuple[
        Dict[str, LigandData], Dict[str, PharmacophoreData], Dict[str, Chem.rdchem.Mol]
    ]:
        keys = list(ligand_mols.keys())
        mols = list(ligand_mols.values())

        # Load receptor as RDKit mol (convert CIF → temp PDB first)
        receptor_pdb = self.system.get_receptor_pdb_path()
        receptor_mol = Chem.MolFromPDBFile(receptor_pdb)
        if not receptor_mol:
            receptor_mol = Chem.MolFromPDBFile(receptor_pdb, sanitize=False)

        (xace_mols, failed_idxs, failure_counts, tcv_counts) = (
            self.ligand_tensorizer.featurize_molecules(mols)
        )
        failed_mols = {}
        for i in failed_idxs:
            failed_mols[keys[i]] = ligand_mols[keys[i]]
            logger.warning("Failed to tensorize ligand %s", keys[i])

        ligand_keys = [key for i, key in enumerate(keys) if i not in failed_idxs]

        ligands_data = {}
        pharmacophores_data = {}
        for i, key in enumerate(ligand_keys):
            raw_sdf = Path(self.system.ligand_sdfs[key]).relative_to(self.ground_truth_dir)

            ccd = self._get_ccd_for_ligand(key)

            # Covalent linkage detection
            is_covalent = False
            linkages = None
            inferred_linkages = self.infer_covalent_linkages(ligand_mol=ligand_mols[key], ligand_id=key)
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

        for i in failed_idxs:
            logger.warning("Failed to tensorize npnde %s", keys[i])

        npnde_keys = [key for i, key in enumerate(keys) if i not in failed_idxs]

        npnde_data = {}
        for i, key in enumerate(npnde_keys):
            raw_sdf = Path(self.system.ligand_sdfs[key]).relative_to(self.ground_truth_dir)

            ccd = self._get_ccd_for_ligand(key)

            is_covalent = False
            linkages = None
            inferred_linkages = self.infer_covalent_linkages(ligand_mol=npnde_mols[key], ligand_id=key)
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


    def _res_names_to_sequence(self, res_names: np.ndarray) -> str:
        """Convert residue names to a single-letter sequence string. One name per residue."""
        sequence = []
        for rn in res_names:
            if rn in residue_to_single:
                sequence.append(residue_to_single[rn])
            else:
                sequence.append(residue_to_single['UNK'])
        return ''.join(sequence)

    def ESM3_embed(self, res_names:np.ndarray, chain_ids:np.ndarray, coords: np.ndarray) -> np.ndarray:

        model = _get_esm3_model()

        residue_names = res_names

        # check if we need to split pocket sequence by chain_id to concatenate for protein_complex
        unique_chains = list(dict.fromkeys(chain_ids))  # preserves order, deduplicates

        if len(unique_chains) > 1:
            esm_chains = []
            for chain in unique_chains:
                mask = np.where(chain_ids == chain)[0]
                chain_res_names = residue_names[mask]
                chain_bb_coords = coords[mask]

                chain_seq = self._res_names_to_sequence(chain_res_names)
                if not chain_seq:
                    continue

                esm_chains.append(ProteinChain.from_backbone_atom_coordinates(
                    chain_bb_coords, sequence=chain_seq
                ))

            if esm_chains:
                protein_complex = ProteinComplex.from_chains(esm_chains)
                return self.embed_protein_complex(model, protein_complex)
            else:
                # Fallback: treat as single chain
                chain_seq = self._res_names_to_sequence(residue_names)
                chain = ProteinChain.from_backbone_atom_coordinates(coords, sequence=chain_seq)
                return self.embed_chain(model, chain)

        else:
            chain_seq = self._res_names_to_sequence(residue_names)
            chain = ProteinChain.from_backbone_atom_coordinates(coords, sequence=chain_seq)
            return self.embed_chain(model, chain)

    def extract_pocket(
        self,
        receptor: struc.AtomArray,
        ligand_coords: np.ndarray,
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

        pocket = receptor[pocket_indices]
        backbone = pocket[struc.filter_peptide_backbone(pocket)]
        backbone_data = self.extract_backbone(backbone)

        pocket = self.check_backbone_order(pocket)
        if pocket is None:
            return None

        bb_mask = struc.filter_peptide_backbone(pocket)

        embedding = self.ESM3_embed(backbone_data.res_names, backbone_data.chain_ids, backbone_data.coords)

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
        """Classify ligands as focal (ligand) vs NPNDE using the parquet."""
        system_df = self.parquet_df[self.parquet_df["system_id"] == self.system_id]
        all_mols = self.system.resolved_ligand_mols

        ligand_mols = {}
        npnde_mols = {}

        ligand_rows = system_df[system_df["ligand_type"] == "ligand"]
        for _, row in ligand_rows.iterrows():
            ligand_id = row["ligand_id"]
            if ligand_id in all_mols:
                ligand_mols[ligand_id] = all_mols[ligand_id]

        npnde_rows = system_df[system_df["ligand_type"] == "npnde"]
        for _, row in npnde_rows.iterrows():
            ligand_id = row["ligand_id"]
            if ligand_id in all_mols:
                npnde_mols[ligand_id] = all_mols[ligand_id]

        # Also add any ligand SDFs not listed in the parquet as NPNDEs
        known_ids = set(ligand_mols.keys()) | set(npnde_mols.keys())
        for lig_id, mol in all_mols.items():
            if lig_id not in known_ids:
                npnde_mols[lig_id] = mol

        return ligand_mols, npnde_mols

    def process_system(self, save_pockets: bool = False) -> Dict[str, Any]:
        logger.info("Processing system %s", self.system_id)

        try:
            ligand_mols, npnde_mols = self.filter_ligands()

            if not ligand_mols:
                return None

            result = self.process_structures(
                ligand_mols=ligand_mols,
                npnde_mols=npnde_mols,
                save_pockets=save_pockets,
            )

            if not result:
                logger.warning(
                    "Skipping system %s due to no ligands remaining", self.system_id
                )
                return None

            return result
        finally:
            self.system.cleanup()

    def process_structures_no_links(
        self,
        ligand_data: Dict[str, LigandData],
        pharmacophore_data: Dict[str, PharmacophoreData],
        npnde_data: Optional[Dict[str, LigandData]] = None,
        save_pockets: bool = False,
    ) -> List[SystemData]:
        if save_pockets:
            self.pdb_writer = PDBWriter()

        # Process receptor from local ground_truth files
        receptor_data = self.process_receptor(
            self.system.protein_atom_array,
            self.system.protein_path,
        )
        if not receptor_data:
            return None

        # Process pockets
        pockets_data = {}
        ligands_to_remove = []
        for ligand_key, ligand in ligand_data.items():
            pocket_data = self.extract_pocket(
                self.system.protein_atom_array,
                ligand.coords,
            )

            if not pocket_data:
                logger.warning(
                    f"No pocket extracted for system {self.system_id} ligand {ligand_key}"
                )
                ligands_to_remove.append(ligand_key)
                continue

            logger.info(f"Extracted pocket for {self.system_id} ligand {ligand_key}")

            if save_pockets:
                output_dir = os.path.dirname(self.system.protein_path)
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
        save_pockets: bool = False,
    ) -> Dict[str, Any]:
        # Process ligands
        ligands_data, pharmacophores_data, failed_mols = self.process_ligands(
            ligand_mols
        )

        if failed_mols:
            if npnde_mols is None:
                npnde_mols = {}
            npnde_mols.update(failed_mols)

        npnde_data = None
        if npnde_mols:
            npnde_data = self.process_npndes(npnde_mols)

        systems_data = self.process_structures_no_links(
            ligand_data=ligands_data,
            pharmacophore_data=pharmacophores_data,
            npnde_data=npnde_data,
            save_pockets=save_pockets,
        )
        if systems_data:
            return {
                "systems_list": systems_data,
                "links": False,
            }
        else:
            return None
