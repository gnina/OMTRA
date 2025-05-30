#!/usr/bin/env python3
import argparse
import numpy as np
from scipy.spatial.distance import cdist

def load_array(npz_path):
    """Load the first array found in a .npz file."""
    archive = np.load(npz_path)
    # take the first array in the archive
    key = next(iter(archive.files))
    return archive[key]

def main():
    parser = argparse.ArgumentParser(
        description="Compute pairwise distances between two blocks of vectors."
    )
    parser.add_argument(
        "--input1",
        required=True,
        help="Path to first .npz file of shape (n1, d)"
    )
    parser.add_argument(
        "--input2",
        required=True,
        help="Path to second .npz file of shape (n2, d)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npy file to write the (n1 × n2) distance matrix"
    )
    args = parser.parse_args()

    # load data
    data1 = load_array(args.input1)
    data2 = load_array(args.input2)
    print('data loaded')

    # sanity check
    if data1.ndim != 2 or data2.ndim != 2:
        raise ValueError("Both input arrays must be 2D (shape: (n, d)).")

    # compute distances
    distances = cdist(data1, data2, metric='jaccard').astype(np.float32)

    # save result
    np.savez_compressed(args.output, distances)
    print(f"Saved distances to {args.output}")

if __name__ == "__main__":
    main()