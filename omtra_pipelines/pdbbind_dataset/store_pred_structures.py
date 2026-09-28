"""Store PDBBind pred-linked structures in Plinder-compatible zarr format."""

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import pandas as pd

from omtra_pipelines.pdbbind_dataset.pdbbind_links_zarr import PdbbindLinksZarrConverter
from omtra_pipelines.plinder_dataset.utils import setup_logger

log_file_path = os.environ.get("LOG_FILE_PATH", "pdbbind_storage.log")
os.environ["LOG_FILE_PATH"] = log_file_path

logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process PDBBind pred structures and store in zarr"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/pdbbind_12A/manifest.parquet",
        help="Path to manifest parquet",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Split to process (train/val/test)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to zarr store",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default=os.environ.get(
            "PDBBIND_RAW_ROOT",
            "~/.local/share/dynamicbind_data",
        ),
        help="Local root for DynamicBind preprocessed files",
    )
    parser.add_argument(
        "--pocket_cutoff",
        type=float,
        default=12.0,
        help="Angstrom cutoff for pocket extraction",
    )
    parser.add_argument(
        "--num_systems",
        type=int,
        default=None,
        help="Limit number of systems (for testing)",
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=1,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--max_pending",
        type=int,
        default=None,
        help="Maximum pending jobs (default: 2*num_cpus)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    raw_root = str(Path(args.raw_root).expanduser().resolve())

    logger.info(
        f"Starting PDBBind pred store with {args.num_cpus} CPUs, "
        f"pocket_cutoff={args.pocket_cutoff}Å"
    )

    manifest = pd.read_parquet(args.manifest)
    manifest = manifest[manifest["split"] == args.split]
    if "files_ok" in manifest.columns:
        manifest = manifest[manifest["files_ok"]]

    converter = PdbbindLinksZarrConverter(
        manifest_df=manifest,
        raw_data_root=raw_root,
        output_path=args.output,
        num_workers=args.num_cpus,
        category="pred",
        pocket_cutoff=args.pocket_cutoff,
    )

    system_ids = list(manifest["system_id"].unique())
    if args.num_systems:
        system_ids = system_ids[: args.num_systems]

    converter.process_dataset(system_ids, max_pending=args.max_pending)

    elapsed = time.time() - start_time
    logger.info(f"Processing completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
