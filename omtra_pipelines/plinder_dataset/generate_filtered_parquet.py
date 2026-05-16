#!/usr/bin/env python3
"""
Generate plinder_filtered.parquet for the time-split systems.

Runs the same filtering logic as filter.py across all systems in
plinder-time.parquet and outputs a parquet with ligand classifications.

Usage:
    python generate_filtered_parquet.py --input plinder-time.parquet --output plinder_filtered_time.parquet --num_cpus 16
"""
import argparse
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from plinder.core import PlinderSystem
from rdkit import Chem
from tqdm import tqdm

from omtra_pipelines.plinder_dataset.utils import LIGAND_MAP, NPNDE_MAP, setup_logger

logger = setup_logger(__name__)


def check_atom_map(mol: Chem.rdchem.Mol, allowed_atoms: List[str] = LIGAND_MAP) -> bool:
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed_atoms:
            return False
    return True


def infer_covalent_linkages(system, ligand_id):
    """Check for covalent linkages by distance using SystemProcessor."""
    try:
        from omtra_pipelines.plinder_dataset.plinder_pipeline import SystemProcessor
        sp = SystemProcessor(system_id=system.system_id, link_type=None)
        return sp.infer_covalent_linkages(ligand_id=ligand_id)
    except Exception:
        return None


def process_system(system_id: str) -> List[dict]:
    """Process a single system and return rows for the filtered parquet."""
    rows = []
    try:
        system = PlinderSystem(system_id=system_id)
        system_annotation = system.system
        system_entry = system.entry
        holo = system.holo_structure

        # Entry-level quality checks
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
                return rows
        if r and r > 0.4:
            return rows
        if rfree and rfree > 0.45:
            return rows
        if r_minus_rfree and r_minus_rfree > 0.075:
            return rows

        # Skip linked_structures lookup — links parquet not available locally
        # and attempting it causes minutes-long timeouts per system
        apo_ids = []
        pred_ids = []

        for lig_ann in system_annotation["ligands"]:
            key = str(lig_ann["instance"]) + "." + lig_ann["asym_id"]

            plip_type = lig_ann["plip_type"]
            ccd_code = lig_ann["ccd_code"]
            num_heavy_atoms = lig_ann["num_heavy_atoms"]
            num_unresolved_heavy_atoms = lig_ann["num_unresolved_heavy_atoms"]
            frac_within_4A = lig_ann.get("frac_within_4A_receptor")

            is_covalent = lig_ann["is_covalent"]
            is_ion = lig_ann["is_ion"]
            is_artifact = lig_ann["is_artifact"]
            is_cofactor = lig_ann.get("is_cofactor", False)
            is_fragment = lig_ann.get("is_fragment", False)

            crystal_contacts = True if lig_ann.get("crystal_contacts") else False

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

            # Volume overlap filters — skip entire system
            if (
                volume_overlap_protein
                and not single_atom_ion
                and volume_overlap_protein >= 0.075
            ):
                return rows
            if (
                volume_overlap_organic_cofactors
                and volume_overlap_organic_cofactors >= 0.075
            ):
                return rows
            if (
                volume_overlap_inorganic_cofactors
                and volume_overlap_inorganic_cofactors >= 0.075
            ):
                return rows

            num_covalent = 0
            if is_covalent:
                num_covalent = len(lig_ann["covalent_linkages"])
            if not is_covalent:
                try:
                    inferred_linkages = infer_covalent_linkages(
                        system=system, ligand_id=key
                    )
                    if inferred_linkages:
                        is_covalent = True
                        num_covalent = len(inferred_linkages)
                except Exception:
                    pass

            # Check atom mapping
            try:
                lig_mol = holo.resolved_ligand_mols[key]
                in_mapping = check_atom_map(lig_mol)
            except Exception:
                in_mapping = False

            # Classify ligand
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
                ligand_type = "npnde"
            else:
                ligand_type = "ligand"

            rows.append({
                "system_id": system_id,
                "ligand_id": key,
                "ligand_type": ligand_type,
                "ccd_code": ccd_code,
                "plip_type": plip_type,
                "num_heavy_atoms": num_heavy_atoms,
                "num_unresolved_heavy_atoms": num_unresolved_heavy_atoms,
                "frac_within_4A_receptor": frac_within_4A,
                "is_covalent": is_covalent,
                "is_ion": is_ion,
                "is_artifact": is_artifact,
                "is_cofactor": is_cofactor,
                "is_fragment": is_fragment,
                "crystal_contacts": crystal_contacts,
                "num_interacting_res": num_interacting_res,
                "volume_overlap_protein": volume_overlap_protein,
                "volume_overlap_organic_cofactors": volume_overlap_organic_cofactors,
                "volume_overlap_inorganic_cofactors": volume_overlap_inorganic_cofactors,
                "determination_method": determination_method,
                "resolution": resolution,
                "r": r,
                "rfree": rfree,
                "r_minus_rfree": r_minus_rfree,
                "apo_ids": str(apo_ids) if apo_ids else None,
                "pred_ids": str(pred_ids) if pred_ids else None,
            })

    except Exception as e:
        logger.debug(f"Error processing {system_id}: {e}")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to plinder-time.parquet")
    parser.add_argument("--output", type=str, required=True, help="Output parquet path")
    parser.add_argument("--num_cpus", type=int, default=16)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    split_map = dict(zip(df["system_id"], df["split"]))
    system_ids = list(df["system_id"].unique())

    # Resume support: load partial results if they exist
    partial_path = Path(args.output).with_suffix(".partial.parquet")
    done_ids = set()
    all_rows = []
    if partial_path.exists():
        partial_df = pd.read_parquet(partial_path)
        all_rows = partial_df.to_dict("records")
        done_ids = set(partial_df["system_id"].unique())
        print(f"Resuming: {len(done_ids)} systems already done, {len(all_rows)} rows loaded")

    remaining = [sid for sid in system_ids if sid not in done_ids]
    print(f"Processing {len(remaining)}/{len(system_ids)} systems with {args.num_cpus} workers...")

    with mp.Pool(processes=args.num_cpus) as pool:
        for i, rows in enumerate(pool.imap_unordered(process_system, remaining, chunksize=50)):
            all_rows.extend(rows)
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(remaining)} systems, {len(all_rows)} ligand rows so far")
                # Save partial progress
                partial_df = pd.DataFrame(all_rows)
                for row in all_rows:
                    if "split" not in row:
                        row["split"] = split_map.get(row["system_id"], "unknown")
                partial_df = pd.DataFrame(all_rows)
                partial_df.to_parquet(partial_path, index=False)

    print(f"Total: {len(all_rows)} ligand rows from {len(system_ids)} systems")

    # Add split info
    for row in all_rows:
        if "split" not in row:
            row["split"] = split_map.get(row["system_id"], "unknown")

    result_df = pd.DataFrame(all_rows)
    result_df.to_parquet(args.output, index=False)
    print(f"Saved to {args.output}")

    # Clean up partial file
    if partial_path.exists():
        partial_path.unlink()

    # Summary
    if len(result_df) > 0:
        print(f"\nSummary:")
        print(f"  Systems with ligands: {result_df['system_id'].nunique()}")
        print(f"  Ligand type counts:")
        print(result_df["ligand_type"].value_counts().to_string())
        print(f"  Split counts:")
        print(result_df.groupby('split')['system_id'].nunique().to_string())
    else:
        print("\nWARNING: No ligand rows generated!")


if __name__ == "__main__":
    mp.set_start_method(os.environ.get("OMTRA_MP_START_METHOD", "fork"), force=True)
    main()
