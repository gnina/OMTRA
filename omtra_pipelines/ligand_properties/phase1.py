import argparse
from pathlib import Path
import numpy as np
import zarr


def parse_args():
    p = argparse.ArgumentParser(description='Create new Zarr array for additional atom features.')

    p.add_argument('--pharmit_path', type=str, help='Path to the Pharmit Zarr store.', default='/net/galaxy/home/koes/ltoft/OMTRA/data/pharmit_dev')   # /net/galaxy/home/koes/icd3/moldiff/OMTRA/data/pharmit
    p.add_argument('--store_name', type=str, help='Name of the Zarr store.', default='train.zarr')
    p.add_argument('--n_feats', type=int, default=6, help='Number of additional features per molecule.')
    p.add_argument('--array_name', type=str, default='extra_feats', help='Name of the new Zarr array.')
    
    args = p.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    n_feats = args.n_feats
    array_name = args.array_name

    store_path = args.pharmit_path+'/'+args.store_name
    root = zarr.open(store_path, mode='a')

    lig_node_group = root['lig/node']
    n_atoms = lig_node_group['x'].shape[0]
    nodes_per_chunk = lig_node_group['x'].chunks[0]

    # Create array if it doesn't exist
    if array_name not in lig_node_group:
        lig_node_group.create_array(array_name, shape=(n_atoms, n_feats), chunks=(nodes_per_chunk, n_feats), dtype=np.int8, overwrite=False)
    
        