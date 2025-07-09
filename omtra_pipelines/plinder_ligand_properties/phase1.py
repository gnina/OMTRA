import argparse
import numpy as np
import zarr


def parse_args():
    p = argparse.ArgumentParser(description='Create new Zarr array for additional atom features for Plinder ligand.')

    p.add_argument('--plinder_path', type=str, help='Path to the Plinder Zarr store.', default='/net/galaxy/home/koes/ltoft/OMTRA/data/plinder')
    p.add_argument('--store_name', type=str, help='Name of the Zarr store.', default='train')
    p.add_argument('--n_feats', type=int, default=6, help='Number of additional features per molecule.')
    p.add_argument('--array_name', type=str, default='extra_feats', help='Name of the new Zarr array.')
    p.add_argument('--feat_names', type=list, default=['impl_H', 'aro', 'hyb', 'ring', 'chiral', 'frag'], help='Name of the new Zarr array.')
    
    args = p.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    n_feats = args.n_feats
    array_name = args.array_name

    for version in ['exp', 'no_links', 'pred']:
        print(f"Creating Zarr array for Plinder version '{version}'")

        store_path = args.plinder_path+'/'+version+'/'+args.store_name+'.zarr'
        root = zarr.open(store_path, mode='a')

        lig_node_group = root['ligand']
        n_atoms = lig_node_group['coords'].shape[0]
        nodes_per_chunk = lig_node_group['coords'].chunks[0]

        # Create array if it doesn't exist
        if array_name not in lig_node_group:
            array = lig_node_group.create_array(array_name, shape=(n_atoms, n_feats), chunks=(nodes_per_chunk, n_feats), dtype=np.int8, overwrite=False)
            array.attrs['features'] = args.feat_names    # add attribute

    print(f"Finished creating Zarr array '{args.array_name}'.")
    
        