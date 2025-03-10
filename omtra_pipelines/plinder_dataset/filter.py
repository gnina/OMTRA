# Volume overlap filtering adapted from Buttenschoen et al. 2023
# https://pubs.rsc.org/en/content/articlehtml/2024/sc/d3sc04185a

import argparse
import logging
import multiprocessing as mp
import os
import pickle
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from biotite.structure.io.pdbx import CIFFile, get_structure
from omtra_pipelines.plinder_dataset.utils import LIGAND_MAP, NPNDE_MAP, setup_logger
from plinder.core import PlinderSystem
from rdkit import Chem
from tqdm import tqdm

logger = setup_logger(
    __name__,
)


def check_atom_map(mol: Chem.rdchem.Mol, allowed_atoms: List[str] = LIGAND_MAP) -> bool:
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed_atoms:
            return False
    return True


def filter(system_id: str) -> (Dict[str, Chem.rdchem.Mol], Dict[str, Chem.rdchem.Mol]):
    from omtra_pipelines.plinder_dataset.plinder_pipeline import StructureProcessor

    try:
        output_path = Path(
            f"/net/galaxy/home/koes/tjkatz/for_omtra/preprocessed_pkls/{system_id}.pkl"
        )
        if output_path.exists():
            with open(output_path, "rb") as f:
                preprocessed = pickle.load(f)
            system = PlinderSystem(system_id=system_id)
            system_structure = system.holo_structure
            ligand_mols = {}
            npnde_mols = {}
            npnde_keys = preprocessed.get("npndes")
            for key in preprocessed["ligands"].keys():
                ligand_mols[key] = system_structure.resolved_ligand_mols[key]
            if npnde_keys:
                for key in npnde_keys.keys():
                    npnde_mols[key] = system_structure.resolved_ligand_mols[key]
            return ligand_mols, npnde_mols

        system = PlinderSystem(system_id=system_id)

        system_annotation = system.system
        system_entry = system.entry
        holo = system.holo_structure

        ligand_mols = {}
        npnde_mols = {}

        determination_method = system_entry.get("determination_method")
        validation = system_entry.get("validation")
        resolution, r, rfree, r_minus_rfree = None, None, None, None
        if validation:
            resolution = validation.get("resolution")
            r = validation.get("r")
            rfree = validation.get("rfree")
            r_minus_rfree = validation.get("r_minus_rfree")

        if resolution and resolution > 3.5:
            if determination_method != "ELECTRON MICROSCOPY":
                return None, None
        if r and r > 0.4:
            return None, None
        if rfree and rfree > 0.45:
            return None, None
        if r_minus_rfree and r_minus_rfree > 0.075:
            return None, None

        structure_processor = StructureProcessor(
            ligand_atom_map=LIGAND_MAP, npnde_atom_map=NPNDE_MAP
        )

        for lig_ann in system_annotation["ligands"]:
            ligand = {}

            key = str(lig_ann["instance"]) + "." + lig_ann["asym_id"]

            plip_type = lig_ann["plip_type"]
            ccd_code = lig_ann["ccd_code"]
            num_heavy_atoms = lig_ann["num_heavy_atoms"]
            num_unresolved_heavy_atoms = lig_ann["num_unresolved_heavy_atoms"]

            is_covalent = lig_ann["is_covalent"]
            is_ion = lig_ann["is_ion"]
            is_artifact = lig_ann["is_artifact"]
            is_cofactor = lig_ann["is_cofactor"]
            is_fragment = lig_ann["is_fragment"]

            crystal_contacts = True if lig_ann["crystal_contacts"] else False

            volume_overlap_protein = lig_ann["posebusters_result"].get(
                "volume_overlap_protein"
            )
            volume_overlap_organic_cofactors = lig_ann["posebusters_result"].get(
                "volume_overlap_organic_cofactors"
            )
            volume_overlap_inorganic_cofactors = lig_ann["posebusters_result"].get(
                "volume_overlap_inorganic_cofactors"
            )

            num_interacting_res = 0
            for chain, res_list in lig_ann["interacting_residues"].items():
                num_interacting_res += len(res_list)

            single_atom_ion = is_ion and (num_heavy_atoms == 1)

            if (
                volume_overlap_protein
                and not single_atom_ion
                and volume_overlap_protein >= 0.075
            ):
                return None, None
            if (
                volume_overlap_organic_cofactors
                and volume_overlap_organic_cofactors >= 0.075
            ):
                return None, None
            if (
                volume_overlap_inorganic_cofactors
                and volume_overlap_inorganic_cofactors >= 0.075
            ):
                return None, None

            num_covalent = 0
            if is_covalent:
                num_covalent = len(lig_ann["covalent_linkages"])
            if not is_covalent:
                inferred_linkages = structure_processor.infer_covalent_linkages(
                    system=system, ligand_id=key
                )
                if inferred_linkages:
                    is_covalent = True
                    num_covalent = len(inferred_linkages)
                    ligand["is_covalent"] = is_covalent

            lig_mol = holo.resolved_ligand_mols[key]
            in_mapping = check_atom_map(lig_mol)

            if (
                not in_mapping
                or is_ion
                or is_artifact
                or (is_covalent and plip_type == "SACCHARIDE")
                or (is_covalent and "NAG" in ccd_code)
                or (num_heavy_atoms > 120)
                or (num_covalent > 1)
                or (num_unresolved_heavy_atoms > 0)
                or crystal_contacts
                or num_interacting_res < 1
            ):
                npnde_mols[key] = lig_mol
            else:
                ligand_mols[key] = lig_mol

        return ligand_mols, npnde_mols

    except Exception as e:
        return None, None
