#!/usr/bin/env python3
import os
import argparse
import itertools
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Generate a bash file to compute pairwise block distances."
    )
    parser.add_argument(
        "npz_dir",
        help="Directory containing block .npz files (named e.g. 0.npz, 1.npz, …)"
    )
    parser.add_argument(
        "--output_dir",
        default="distances",
        help="Directory to store per‐pair output files"
    )
    parser.add_argument(
        "--bash_file",
        default="run_all_block_cdists.sh",
        help="Name of the generated bash script"
    )
    args = parser.parse_args()

    # find all .npz files and sort by filename
    files = sorted([
        f for f in os.listdir(args.npz_dir)
        if f.endswith(".npz")
    ], key=lambda x: int(Path(x).stem.split('_')[-1]))

    # ensure output_dir exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.bash_file, "w") as bf:
        # bf.write("#!/bin/bash\n\n")
        for f1, f2 in itertools.combinations_with_replacement(files, 2):
            path1 = os.path.join(args.npz_dir, f1)
            path2 = os.path.join(args.npz_dir, f2)
            name1 = os.path.splitext(f1)[0]
            name2 = os.path.splitext(f2)[0]
            out_name = f"{name1}_{name2}_dist.npy"
            out_path = os.path.join(args.output_dir, out_name)
            cmd = (
                f"python -m omtra_pipelines.plinder_clustering.compute_block_dist "
                f"--input1 {path1} "
                f"--input2 {path2} "
                f"--output {out_path}"
            )
            bf.write(cmd + "\n")

    print(f"Generated {args.bash_file} with {len(files)*(len(files)+1)//2} commands.")

if __name__ == "__main__":
    main()