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
    p.add_argument('--flat', action='store_true', help='Store path is {plinder_path}/{store_name}.zarr (no version subdirs, e.g. PDBbind).')
    p.add_argument('--versions', nargs='*', default=None, help='Link versions to process (default: exp no_links pred). Ignored when --flat is set.')

    args = p.parse_args()

    return args


def store_path_for(args, version=None):
    if args.flat:
        return f"{args.plinder_path}/{args.store_name}.zarr"
    return f"{args.plinder_path}/{version}/{args.store_name}.zarr"


def create_extra_feats_array(store_path, array_name, n_feats, feat_names):
    root = zarr.open(store_path, mode='r+')
    lig_node_group = root['ligand']
    n_atoms = lig_node_group['coords'].shape[0]
    nodes_per_chunk = lig_node_group['coords'].chunks[0]

    if array_name not in lig_node_group:
        array = lig_node_group.create_array(
            array_name,
            shape=(n_atoms, n_feats),
            chunks=(nodes_per_chunk, n_feats),
            dtype=np.int8,
            overwrite=False,
        )
        array.attrs['features'] = feat_names
        print(f"Created '{array_name}' at {store_path} with shape ({n_atoms}, {n_feats})")
    else:
        print(f"'{array_name}' already exists at {store_path}")


if __name__ == '__main__':
    args = parse_args()
    versions = [None] if args.flat else (args.versions or ['exp', 'no_links', 'pred'])

    for version in versions:
        label = 'flat' if args.flat else version
        print(f"Creating Zarr array for store '{label}'")
        create_extra_feats_array(
            store_path_for(args, version),
            args.array_name,
            args.n_feats,
            args.feat_names,
        )

    print(f"Finished creating Zarr array '{args.array_name}'.")
    
        