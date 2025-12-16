import argparse
from pathlib import Path
import numpy as np
import zarr


def parse_args():
    p = argparse.ArgumentParser(description='Create new Zarr array for additional atom features.')

    p.add_argument('--pharmit_path', type=str, help='Path to the Pharmit Zarr store.', default='/net/galaxy/home/koes/ltoft/OMTRA/data/pharmit_dev')
    p.add_argument('--store_name', type=str, help='Name of the Zarr store.', default='train')
    p.add_argument('--n_feats', type=int, default=6, help='Number of additional features per molecule.')
    p.add_argument('--array_name', type=str, default='extra_feats', help='Name of the new Zarr array.')
    p.add_argument("--entity_type", type=str, default='atom', help="Entity type of the feature: node or edge")
    p.add_argument("--feat_names", type=str, nargs="+", default=['impl_H', 'aro', 'hyb', 'ring', 'chiral', 'frag'], help="Feature names.")
    
    args = p.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    n_feats = args.n_feats
    array_name = args.array_name

    store_path = args.pharmit_path+'/'+args.store_name+'.zarr'
    root = zarr.open(store_path, mode='a')

    lig_node_group = root['lig/node']
    n_atoms = lig_node_group['x'].shape[0]
    nodes_per_chunk = lig_node_group['x'].chunks[0]

    lig_edge_group = root['lig/edge']
    n_edges = lig_edge_group['edge_index'].shape[0]
    edges_per_chunk = lig_edge_group['edge_index'].chunks[0]

    # Create array if it doesn't exist
    
    if (args.entity_type == 'atom') and (array_name not in lig_node_group):
        array = lig_node_group.create_array(array_name, shape=(n_atoms, n_feats), chunks=(nodes_per_chunk, n_feats), dtype=np.int8, overwrite=False)
    
    elif (args.entity_type == 'edge') and (array_name not in lig_edge_group):
        array = lig_edge_group.create_array(array_name, shape=(n_edges, n_feats), chunks=(edges_per_chunk, n_feats), dtype=np.int8, overwrite=False)
    
    else:
        if args.entity_type not in ['atom', 'edge']:
            raise NotImplementedError(f'{args.entity_type} is not a valid entity. Select from atom or edges.')
        else:
            print(f'{array_name} array already exists for {args.entity_type}')
        
    array.attrs['features'] = args.feat_names    # add attribute
    print(f"Finished creating Zarr array {array_name}.")
    
        