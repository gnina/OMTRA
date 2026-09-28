"""Process DynamicBind PDBBind holo/ligand/AF2 files into Plinder-compatible SystemData."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import biotite.structure as struc
import numpy as np
from biotite.structure import filter_amino_acids
from biotite.structure.io.pdb import PDBFile
from omtra.constants import lig_atom_type_map, npnde_atom_type_map
from omtra.data.extra_ligand_features import fragment_molecule, ligand_properties
from omtra.data.pharmacophores import get_pharmacophores
from omtra.data.plinder import (
    LigandData,
    PharmacophoreData,
    StructureData,
    SystemData,
)
from omtra.data.xace_ligand import MoleculeTensorizer
from omtra.utils.misc import bad_mol_reporter
from omtra_pipelines.crossdocked_dataset.pipeline_components import SystemProcessor
from omtra_pipelines.plinder_dataset.filter import check_atom_map
from omtra_pipelines.plinder_dataset.utils import setup_logger
from rdkit import Chem

logger = setup_logger(__name__)


class PdbbindSystemProcessor(SystemProcessor):
    """File-based processor for DynamicBind PDBBind pocket_aligned_fill_missing data."""

    def __init__(
        self,
        system_id: str,
        holo_pdb: str,
        ligand_sdf: str,
        pred_pdb: str,
        link_id: str,
        pocket_cutoff: float = 12.0,
        raw_data_root: Optional[str] = None,
        ligand_atom_map=lig_atom_type_map,
        npnde_atom_map=npnde_atom_type_map,
    ):
        self.system_id = system_id
        self.holo_pdb = Path(holo_pdb)
        self.ligand_sdf = Path(ligand_sdf)
        self.pred_pdb = Path(pred_pdb)
        self.link_id = link_id
        self.pocket_cutoff = pocket_cutoff
        self.ligand_atom_map = ligand_atom_map
        self.npnde_atom_map = npnde_atom_map
        self.raw_data = Path(
            raw_data_root or "~/.local/share/dynamicbind_data"
        ).expanduser().resolve()

        self.receptor_path = self.holo_pdb
        self.ligand_path = self.ligand_sdf

        self.ligand_tensorizer = MoleculeTensorizer(atom_map=ligand_atom_map, n_cpus=1)
        self.npnde_tensorizer = MoleculeTensorizer(atom_map=npnde_atom_map, n_cpus=1)
        self.pdb_writer = None

    def _rel_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.raw_data))
        except ValueError:
            return path.name

    def create_atom_key(self, atom: struc.Atom) -> str:
        return f"{atom.chain_id}_{atom.res_id}_{atom.res_name}_{atom.atom_name}"

    @staticmethod
    def _ordered_residue_groups(arr: struc.AtomArray):
        """Return [(chain_id, res_id, res_name, atom_subarray), ...] in file order."""
        groups = []
        seen = set()
        for atom in arr:
            key = (str(atom.chain_id), int(atom.res_id))
            if key in seen:
                continue
            seen.add(key)
            mask = (arr.chain_id == atom.chain_id) & (arr.res_id == atom.res_id)
            sub = arr[mask]
            groups.append((key[0], key[1], sub.res_name[0], sub))
        return groups

    @staticmethod
    def _residue_one_letter(res_name: str) -> str:
        from biotite.sequence import ProteinSequence

        try:
            return ProteinSequence.convert_letter_3to1(res_name)
        except Exception:
            return "X"

    def align_pred_to_holo(
        self,
        holo: struc.AtomArray,
        pred: struc.AtomArray,
    ) -> Optional[struc.AtomArray]:
        """Map AF2 pred coords onto holo atom order via sequence alignment.

        DynamicBind releases holo/AF2 PDBs already pocket-aligned in the same
        coordinate frame, but chain IDs and residue numbering often differ.
        Exact atom-key matching therefore fails; sequence alignment is used
        instead (same idea as Plinder's align_common_sequence).
        """
        from biotite.sequence import ProteinSequence
        from biotite.sequence.align import Alignment, align_optimal
        from biotite.sequence.align import SubstitutionMatrix

        if len(holo) == 0 or len(pred) == 0:
            logger.error(f"{self.system_id}: empty holo or pred structure")
            return None

        matrix = SubstitutionMatrix.std_protein_matrix()
        aligned = holo.copy()
        n_from_pred = 0
        n_from_holo = 0

        holo_chains = sorted({str(c) for c in holo.chain_id})
        pred_chains = sorted({str(c) for c in pred.chain_id})
        if len(holo_chains) != len(pred_chains):
            logger.warning(
                f"{self.system_id}: chain count mismatch holo={holo_chains} pred={pred_chains}"
            )
        chain_pairs = list(zip(holo_chains, pred_chains))

        for holo_chain, pred_chain in chain_pairs:
            holo_groups = self._ordered_residue_groups(holo[holo.chain_id == holo_chain])
            pred_groups = self._ordered_residue_groups(pred[pred.chain_id == pred_chain])
            if not holo_groups or not pred_groups:
                continue

            holo_seq = ProteinSequence(
                "".join(self._residue_one_letter(g[2]) for g in holo_groups)
            )
            pred_seq = ProteinSequence(
                "".join(self._residue_one_letter(g[2]) for g in pred_groups)
            )
            alignment: Alignment = align_optimal(holo_seq, pred_seq, matrix)[0]

            for holo_res_i, pred_res_i in alignment.trace:
                if holo_res_i == -1:
                    continue
                _, _, _, holo_res_atoms = holo_groups[holo_res_i]
                if pred_res_i == -1:
                    n_from_holo += len(holo_res_atoms)
                    continue

                _, _, _, pred_res_atoms = pred_groups[pred_res_i]
                pred_coords = {a.atom_name: a.coord for a in pred_res_atoms}
                for atom in holo_res_atoms:
                    mask = (
                        (holo.chain_id == atom.chain_id)
                        & (holo.res_id == atom.res_id)
                        & (holo.atom_name == atom.atom_name)
                    )
                    idx = np.where(mask)[0]
                    if len(idx) != 1:
                        continue
                    if atom.atom_name in pred_coords:
                        aligned.coord[idx[0]] = pred_coords[atom.atom_name]
                        n_from_pred += 1
                    else:
                        n_from_holo += 1

        if n_from_pred == 0:
            logger.error(
                f"{self.system_id}: sequence alignment matched 0 pred atoms to holo"
            )
            return None

        if n_from_holo:
            logger.debug(
                f"{self.system_id}: {n_from_holo} holo atoms missing from pred "
                f"(filled from holo), {n_from_pred} from pred"
            )
        return aligned

    def _load_holo_pdb(
        self, path: Path
    ) -> Tuple[Optional[struc.AtomArray], Dict[str, Chem.rdchem.Mol]]:
        """Load holo receptor, splitting protein from NPNDEs (cofactors, ions, etc.)."""
        if not path.exists():
            logger.error(f"Missing PDB: {path}")
            return None, {}
        receptor_array = PDBFile.read(path).get_structure(model=1)
        receptor_array.bonds = struc.connect_via_residue_names(receptor_array)
        holo_array, npnde_mols = self.filter_npndes(receptor_array)
        holo_array = holo_array[holo_array.res_name != "HOH"]
        holo_array = holo_array[holo_array.element != "H"]
        holo_array = holo_array[holo_array.element != "D"]
        return holo_array, npnde_mols

    def _load_protein_pdb(self, path: Path) -> Optional[struc.AtomArray]:
        if not path.exists():
            logger.error(f"Missing PDB: {path}")
            return None
        receptor = PDBFile.read(path).get_structure(model=1)
        receptor = receptor[filter_amino_acids(receptor)]
        receptor = receptor[receptor.res_name != "HOH"]
        receptor = receptor[receptor.element != "H"]
        receptor = receptor[receptor.element != "D"]
        return receptor

    def _load_ligand(self) -> Optional[Chem.rdchem.Mol]:
        if not self.ligand_sdf.exists():
            logger.error(f"Missing ligand SDF: {self.ligand_sdf}")
            return None
        ligand_mol = Chem.MolFromMolFile(str(self.ligand_sdf), sanitize=False)
        if ligand_mol is None:
            logger.error(f"Failed to parse ligand: {self.ligand_sdf}")
            return None
        try:
            Chem.SanitizeMol(ligand_mol)
        except Exception as exc:
            logger.warning(f"Ligand sanitize failed for {self.system_id}: {exc}")
            return None
        if not check_atom_map(ligand_mol):
            logger.warning(f"Ligand atom map check failed for {self.system_id}")
            return None
        return ligand_mol

    def process_system(self) -> Optional[Dict[str, Any]]:
        ligand_mol = self._load_ligand()
        holo_array, npnde_mols = self._load_holo_pdb(self.holo_pdb)
        pred_array = self._load_protein_pdb(self.pred_pdb)
        if ligand_mol is None or holo_array is None or pred_array is None:
            return None

        npnde_data = self.process_npndes(npnde_mols) if npnde_mols else {}

        ligand_path = self._rel_path(self.ligand_sdf)
        ligand_data, pharm_data = self.process_ligand(
            ligand_mol=ligand_mol,
            receptor_array=holo_array,
            ligand_path=ligand_path,
        )
        if ligand_data is None:
            return None

        holo_data = self.process_receptor(holo_array, self._rel_path(self.holo_pdb))
        if holo_data is None:
            return None

        pocket_data = self.extract_pocket(holo_array, ligand_data.coords)
        if pocket_data is None:
            return None

        reordered_pred = self.align_pred_to_holo(holo_array, pred_array)
        if reordered_pred is None:
            return None

        pred_data = self.process_receptor(
            reordered_pred, self._rel_path(self.pred_pdb)
        )
        if pred_data is None:
            return None

        system_data = SystemData(
            system_id=self.system_id,
            ligand_id="1",
            ligand=ligand_data,
            receptor=holo_data,
            pharmacophore=pharm_data,
            pocket=pocket_data,
            link_id=self.link_id,
            link_type="pred",
            link=pred_data,
            npndes=npnde_data,
        )

        return {"pred": {self.link_id: [system_data]}}
