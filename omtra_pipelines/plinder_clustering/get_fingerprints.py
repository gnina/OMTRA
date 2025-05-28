#!/usr/bin/env python3

import argparse
from pathlib import Path
from omtra_pipelines.plinder_clustering.tools.zarr_to_fp import PlinderLigandExtractor
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a block of a Zarr ligand store into RDKit fingerprints."
    )
    parser.add_argument(
        "processed_data_dir",
        type=Path,
        help="Path to the input Zarr store containing ligands"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        required=True,
        help="Number of molecules to process in this block"
    )
    parser.add_argument(
        "--block-index",
        type=int,
        default=0,
        help="Index of the block to process (zero‐based)"
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('./fp_blocks/')
    )
    return parser.parse_args()

def main(args):
    """
    1. Open the Zarr store at args.zarr_path
    2. Read the slice corresponding to args.block_index and args.block_size
    3. Convert each entry to RDKit Mol
    4. Compute fingerprints and write out (e.g., to disk or stdout)
    """
    ds = PlinderLigandExtractor(split='train', processed_data_dir=args.processed_data_dir)
    start_idx = args.block_index * args.block_size
    end_idx = min(start_idx + args.block_size, len(ds))


    fps = []
    for idx in range(start_idx, end_idx):
        fps.apend(ds.get_fingerprint(idx))
    fps = np.array(fps).astype(bool)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f"fps_block_{args.block_index}.npz"
    np.savez_compressed(output_file, fps=fps)

if __name__ == "__main__":
    args = parse_args()
    main(args)