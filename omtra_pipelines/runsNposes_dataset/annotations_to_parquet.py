"""Convert Runs N' Poses annotations.csv into a parquet matching
the plinder_filtered.parquet schema used by the processing pipeline.

All systems are assigned to the 'test' split since Runs N' Poses is
a benchmark dataset.  Ligands listed in the annotations are marked as
ligand_type='ligand'; any additional SDF files discovered in the
ground_truth directory that are not in the annotations are marked as
ligand_type='npnde'.

Expected ground_truth directory layout (after extracting ground_truth.tar.gz):
    ground_truth/
      {system_id}/
        receptor.cif
        ligand_files/
          {ligand_id}.sdf        (e.g. 1.B.sdf)

Usage:
    python annotations_to_parquet.py \
        --annotations annotations.csv \
        --ground_truth /path/to/ground_truth \
        --output rnp_systems.parquet
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Runs N' Poses annotations.csv to pipeline parquet"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="Path to Runs N' Poses annotations.csv",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default=None,
        help="Path to extracted ground_truth directory. "
        "If provided, scans for additional ligand SDFs to include as NPNDEs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rnp_systems.parquet",
        help="Output parquet path",
    )
    return parser.parse_args()


def discover_ligand_ids_from_dir(system_dir: Path) -> set:
    """Discover ligand IDs from SDF files in a system's ground_truth directory."""
    ligand_dir = system_dir / "ligand_files"
    if not ligand_dir.exists():
        return set()
    return {p.stem for p in ligand_dir.glob("*.sdf")}


def main():
    args = parse_args()

    ann = pd.read_csv(args.annotations)
    print(f"Loaded annotations: {len(ann)} rows, {ann['system_id'].nunique()} unique systems")

    # Restrict to drug-like ligands — matches the RnP paper's evaluable set.
    ann = ann[ann["ligand_is_proper"] == True]
    print(f"After ligand_is_proper filter: {len(ann)} rows, {ann['system_id'].nunique()} systems")

    # Deduplicate: one row per (system_id, ligand_instance_chain)
    focal = (
        ann[["system_id", "ligand_instance_chain", "ligand_ccd_code"]]
        .drop_duplicates(subset=["system_id", "ligand_instance_chain"])
        .rename(columns={
            "ligand_instance_chain": "ligand_id",
            "ligand_ccd_code": "ccd_code",
        })
    )
    focal["ligand_type"] = "ligand"
    focal["split"] = "test"

    print(f"Focal ligands (from annotations): {len(focal)} entries")

    # Optionally discover NPNDEs from ground_truth directory
    npnde_rows = []
    if args.ground_truth:
        gt_dir = Path(args.ground_truth)
        if not gt_dir.exists():
            print(f"WARNING: ground_truth directory {gt_dir} does not exist, skipping NPNDE discovery")
        else:
            focal_lookup = set(zip(focal["system_id"], focal["ligand_id"]))
            system_ids = focal["system_id"].unique()

            for system_id in system_ids:
                system_dir = gt_dir / system_id
                if not system_dir.exists():
                    continue
                all_lig_ids = discover_ligand_ids_from_dir(system_dir)
                focal_ids = {lid for sid, lid in focal_lookup if sid == system_id}
                npnde_ids = all_lig_ids - focal_ids

                for npnde_id in npnde_ids:
                    npnde_rows.append({
                        "system_id": system_id,
                        "ligand_id": npnde_id,
                        "ccd_code": None,
                        "ligand_type": "npnde",
                        "split": "test",
                    })

            print(f"NPNDEs discovered from ground_truth: {len(npnde_rows)} entries")

    # Combine
    if npnde_rows:
        npnde_df = pd.DataFrame(npnde_rows)
        df = pd.concat([focal, npnde_df], ignore_index=True)
    else:
        df = focal

    # Merge in proper-ligand rows for systems not present in annotations.csv.
    # The Zenodo annotations.csv ships 2,579 systems but the paper's inputs.json
    # has 2,600; the missing 21 systems' proper ligands are recovered from the
    # per-method prediction CSVs and stored alongside this script.
    missing_path = Path(__file__).parent / "missing_proper_ligands.csv"
    if missing_path.exists():
        missing_df = pd.read_csv(missing_path)
        before = len(df)
        df = pd.concat([df, missing_df], ignore_index=True)
        print(f"Merged {len(missing_df)} supplementary rows from {missing_path.name} ({before} -> {len(df)})")

    # Report
    print(f"\nFinal parquet: {len(df)} rows, {df['system_id'].nunique()} unique systems")
    print(f"  ligand:  {len(df[df['ligand_type'] == 'ligand'])}")
    print(f"  npnde:   {len(df[df['ligand_type'] == 'npnde'])}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(output_path), index=False)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
