import argparse
import logging
import multiprocessing as mp
import time
from pathlib import Path
import pandas as pd
import os

os.environ["LOG_FILE_PATH"] = (
    "/net/galaxy/home/koes/tjkatz/for_omtra/logs/plinder_link_storage_train.log"
)

from omtra_pipelines.plinder_dataset.plinder_pipeline import SystemProcessor
from omtra_pipelines.plinder_dataset.plinder_links_zarr import PlinderLinksZarrConverter
from omtra.constants import lig_atom_type_map, npnde_atom_type_map
from omtra_pipelines.plinder_dataset.utils import setup_logger


logger = setup_logger(
    __name__,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process plinder linked structures and store in zarr"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to systems parquet"
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Split to process (train/val/test/removed/unassigned)",
    )
    parser.add_argument("--output", type=str, required=True, help="Path to zarr store")
    parser.add_argument(
        "--log_dir",
        type=str,
        required=False,
        default="/net/galaxy/home/koes/tjkatz/for_omtra/logs",
        help="Path to log directory",
    )
    parser.add_argument(
        "--pocket_cutoff",
        type=int,
        required=False,
        default=8,
        help="Angstrom cutoff for pocket extraction",
    )
    parser.add_argument(
        "--num_systems", type=int, required=False, help="Number of systems to process"
    )
    parser.add_argument(
        "--num_cpus", type=int, required=False, default=1, help="Number of cpus"
    )
    parser.add_argument(
        "--max_pending",
        type=int,
        required=False,
        default=None,
        help="Maximum number of pending jobs (default: 2*num_cpus)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    logger.info(
        f"Starting processing with {args.num_cpus} CPUs, max_pending={args.max_pending or args.num_cpus * 2}"
    )

    apo_processor = SystemProcessor(
        ligand_atom_map=lig_atom_type_map,
        npnde_atom_map=npnde_atom_type_map,
        pocket_cutoff=args.pocket_cutoff,
        link_type="apo",
    )
    apo_converter = PlinderLinksZarrConverter(
        output_path=args.output,
        system_processor=apo_processor,
        num_workers=args.num_cpus,
        category="apo",
    )

    df = pd.read_parquet(args.data).drop_duplicates("system_id")
    df = df[df["split"] == args.split]
    apo_df = df[df["apo_ids"].notna()]
    pred_df = df[df["pred_ids"].notna()]
    apo_system_ids = list(apo_df["system_id"])
    pred_system_ids = list(pred_df["system_id"])

    if args.num_systems:
        apo_system_ids = apo_system_ids[: args.num_systems]
        pred_system_ids = pred_system_ids[: args.num_systems]

    apo_converter.process_dataset(apo_system_ids, max_pending=args.max_pending)

    pred_processor = SystemProcessor(
        ligand_atom_map=lig_atom_type_map,
        npnde_atom_map=npnde_atom_type_map,
        pocket_cutoff=args.pocket_cutoff,
        link_type="pred",
    )
    pred_converter = PlinderLinksZarrConverter(
        output_path=args.output,
        system_processor=pred_processor,
        num_workers=args.num_cpus,
        category="pred",
    )

    pred_converter.process_dataset(pred_system_ids, max_pending=args.max_pending)

    end_time = time.time()
    logger.info(f"Processing completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
