import argparse
import logging
import multiprocessing as mp
import time
from pathlib import Path
import pandas as pd
import os

log_file_path = os.environ.get("LOG_FILE_PATH", "plinder_storage.log")
os.environ["LOG_FILE_PATH"] = log_file_path

from omtra_pipelines.plinder_dataset.plinder_pipeline import SystemProcessor
from omtra_pipelines.plinder_dataset.plinder_unlink_zarr import (
    PlinderNoLinksZarrConverter,
)
from omtra.constants import lig_atom_type_map, npnde_atom_type_map
from omtra_pipelines.plinder_dataset.utils import setup_logger


logger = setup_logger(
    __name__,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process plinder structures (no links) and store in zarr"
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
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="If set, generate embeddings (default: False)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    logger.info(
        f"Starting processing with {args.num_cpus} CPUs, max_pending={args.max_pending or args.num_cpus * 2}"
    )

    converter = PlinderNoLinksZarrConverter(
        output_path=args.output,
        num_workers=args.num_cpus,
        category=None,
        embeddings=args.embeddings
    )

    df = pd.read_parquet(args.data).drop_duplicates("system_id")
    df = df[df["split"] == args.split]
    system_ids = list(df["system_id"])

    if args.num_systems:
        system_ids = system_ids[: args.num_systems]

    converter.process_dataset(system_ids, max_pending=args.max_pending)

    end_time = time.time()
    logger.info(f"Processing completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
