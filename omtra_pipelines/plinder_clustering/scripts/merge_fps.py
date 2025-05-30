#!/usr/bin/env python3
import argparse
import os
import glob
import numpy as np
from pathlib import Path

# from rdkit import Chem
import bitbirch.bitbirch as bb
# import bitbirch.plotting_utils as plotting_utils
# import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Concatenate all 'fps' arrays from .npz files in a directory"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing .npz files with an 'fps' array"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Output .npz filepath (e.g. concatenated_fps.npz)"
    )
    return parser.parse_args()

def cluster_mols(fps, smiles):
    fps = fps.astype(np.int64)
    branching_factor = 50

    # do an intial clustering round
    bb.set_merge('diameter')
    brc = bb.BitBirch(threshold=0.65, branching_factor=branching_factor)
    brc.fit(fps)

    # do second clustering round with higher threshold
    rest, big = brc.prepare_data_BFs(fps)

    # Delete the initial tree to save memory
    #del brc

    # Create a new tree, this time with the data from the first tree and using tolerance merging and a higher threshold
    bb.set_merge('tolerance', 0.00)
    brc_refined = bb.BitBirch(threshold=0.74, branching_factor=branching_factor)

    # Fit the new tree, notice that for this we are using fit_BFs, since we took the BitFeatures from the first tree
    brc_refined.fit_BFs(rest)
    brc_refined.fit_BFs(big)

    # print some info about clusters
    mol_ids = brc_refined.get_cluster_mol_ids()
    print("Number of clusters: ", len(mol_ids))

    # Get the clusters with more than 10 molecules
    clusters = [cluster for cluster in mol_ids if len(cluster) > 10]
    print("Number of clusters with more than 10 molecules: ", len(clusters))

    # get cluster assignments
    assignments = brc_refined.get_assignments(n_mols=fps.shape[0])

    return assignments

def main():
    args = parse_args()

    # find all .npz files in the input directory
    pattern = os.path.join(args.input_dir, "*.npz")
    files = sorted(glob.glob(pattern), 
    key=lambda x: int(Path(x).stem.split('_')[-1]) )
    if not files:
        print(f"Error: no .npz files found in '{args.input_dir}'")
        return

    all_fps = []
    all_smiles = []
    for fp_file in files:
        print(Path(fp_file).name)
        with np.load(fp_file) as data:
            if "fps" not in data:
                print(f"Warning: skipping '{fp_file}' (no key 'fps')")
                continue
            arr = data["fps"]
            all_fps.append(arr)
            all_smiles.append(data['smiles'])

    if not all_fps:
        print("Error: no 'fps' arrays were loaded.")
        return

    # concatenate
    concat_fps = np.concatenate(all_fps, axis=0)
    concat_smiles = np.concatenate(all_smiles, axis=0)

    # deduplicate 
    print('deduplicating concatenated array...')
    unique_fps, first_idx, mapping = np.unique(concat_fps, axis=0, return_inverse=True, return_index=True)
    unique_smiles = concat_smiles[first_idx]
    print(f"Dataset has {concat_fps.shape[0]:,} rows, deduplicated to {unique_fps.shape[0]:,} unique rows.")

    # run clustering
    print('clustering unique fingerprints...')
    unique_mol_cluster_assignments = cluster_mols(unique_fps, unique_smiles)

    # get cluster assignments for all concatenated fingerprints
    all_mol_cluster_assignments = unique_mol_cluster_assignments[mapping]

    np.savez_compressed(args.output_file, 
        unique_fps=unique_fps, 
        unique_smiles=unique_smiles, 
        unique_mapping=mapping,
        unique_mol_cluster_assignments=unique_mol_cluster_assignments,
        all_mol_cluster_assignments=all_mol_cluster_assignments,
    )

    print(
        f"Saved concatenated array of shape {unique_fps.shape}, "
        f"to '{args.output_file}', deduplication reduced rows by {(concat_fps.shape[0] - unique_fps.shape[0]) / concat_fps.shape[0] * 100:.2f}%"
    )

if __name__ == "__main__":
    main()