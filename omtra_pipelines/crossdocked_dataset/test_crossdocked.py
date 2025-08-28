import os
from pathlib import Path
from omtra.dataset.crossdocked import CrossdockedDataset
from rdkit import Chem
import logging
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import time
import traceback
import sys
sys.path.append("/net/galaxy/home/koes/jmgupta/OMTRA")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/net/galaxy/home/koes/jmgupta/OMTRA/omtra_pipelines/crossdocked_dataset/zarr_data_with_npnde", help="Root directory for crossdocked data")
    #parser.add_argument("--link_version", type=str, default="", help="Optional link version (e.g., 'apo', 'pred')")
    parser.add_argument("--split", type=str, default="internal_split0/train", help="Split name and zarr file name")
    parser.add_argument("--n_samples", type=int, default=5, help="number of systems to load")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    # set zarr path
    zarr_path = f"{args.root_dir}/{args.split}" #split should be for example train_zarr/train0_output.zarr (folder name and file name)

    # Initialize the CrossdockedDataset
    dataset = CrossdockedDataset(
        split=args.split,
        processed_data_dir=args.root_dir,
    )
    print(dataset.system_lookup.columns)
    print(dataset.system_lookup.head())
    graph = dataset[("protein_ligand_denovo", 0)]

    print("Node types:", graph.ntypes)
    print("Edge types:", graph.etypes)

    # Search for a sample with npnde_to_npnde edges
    # Search specifically for systems with npndes
    print("Searching for systems with npndes...")

    systems_with_npndes = dataset.system_lookup[
        dataset.system_lookup['npnde_idxs'].notna() & 
        (dataset.system_lookup['npnde_idxs'] != '[]')
    ]['system_idx'].tolist()

    print(f"Found {len(systems_with_npndes)} systems with npndes")
    print(f"First few system indices with npndes: {systems_with_npndes[:10]}")

    # Test the first system with npndes
    if systems_with_npndes:
        test_idx = systems_with_npndes[0]
        print(f"\nTesting system {test_idx} (known to have npndes)")
        
        graph = dataset[("protein_ligand_denovo", test_idx)]
        npnde_edges = graph.edges['npnde_to_npnde']
        
        print(f"npnde_to_npnde edges: {npnde_edges.data['e_1_true'].shape[0]}")
        print("Edge data keys:", list(npnde_edges.data.keys()))
        
        for key in npnde_edges.data.keys():
            print(f"  {key}: shape {npnde_edges.data[key].shape}")