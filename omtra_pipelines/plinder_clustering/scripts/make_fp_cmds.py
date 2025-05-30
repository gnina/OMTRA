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
        '--output_dir',
        type=Path,
        default=Path('./fp_blocks/')
    )
    parser.add_argument(
        '--cmd_file',
        type=Path,
        default=Path('./make_fp_cmds.sh'),
    )
    return parser.parse_args()

def main(args):
    ds = PlinderLigandExtractor(split='train', processed_data_dir=args.processed_data_dir)

    n_blocks, n_mols_last_block = divmod(len(ds), args.block_size)
    if n_mols_last_block > 0:
        n_blocks += 1

    cmds = []
    for block_index in range(n_blocks):
        cmd = f"python -m omtra_pipelines.plinder_clustering.scripts.get_fingerprints " \
              f"{args.processed_data_dir} --block-size {args.block_size} " \
              f"--block-index {block_index} --output_dir {args.output_dir}"
        cmds.append(cmd)

    with open(args.cmd_file, 'w') as f:
        f.write('\n'.join(cmds)+'\n')

if __name__ == "__main__":
    args = parse_args()
    main(args)