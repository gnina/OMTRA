import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time
import pickle
import zarr

def parse_args():
    p = argparse.ArgumentParser(description='Get ligand atom counts, mean x-coord, and x-coord standard deviation for Pharmit ligands.')

    p.add_argument('--pharmit_path', type=Path, help='Path to the Pharmit Zarr store.', default='/net/galaxy/home/koes/icd3/moldiff/OMTRA/data/pharmit')
    p.add_argument('--split', type=str, help='Data split: train or val.', default='train')
    p.add_argument('--n_seeds', type=int, help='Number of indices to sample.', default=50)
    p.add_argument('--batch_size', type=int, help='Number of ligands to sample per seed index.', default=100000)
    p.add_argument('--output_dir', type=Path, help='Output directory for processed data.', default=Path('/net/galaxy/home/koes/ltoft/OMTRA/omtra_pipelines/pharmit_ligand_properties/outputs/lig_x_std'))
 
    return p.parse_args()


def rand_lig_x_summary(atom_x, graph_lookup, n_seeds, batch_size):
    """
    Summarize random ligands as (lig idx, # atoms, coord std dev) tuples.
    """
    total_ligs = graph_lookup.shape[0]

    all_seed_idxs = np.arange(0, total_ligs, batch_size)
    seed_idxs = np.random.choice(all_seed_idxs, size=min(n_seeds, len(all_seed_idxs)), replace=False)

    x_props = []

    # iterate over molecules in chunks
    for seed_idx in tqdm(seed_idxs, desc="Processing", unit="batches"):
        print(f"Processing seed starting at index {seed_idx}")

        batch = graph_lookup[seed_idx : seed_idx + batch_size]
        batch_start = batch[0][0]
        batch_end = batch[-1][1]
        batch_coords = atom_x[batch_start:batch_end]

        for i, (start, end) in enumerate(batch):
            lig_idx = seed_idx + i
            lig_coords = batch_coords[start - batch_start:end - batch_start]
            n_atoms = end - start
            std = lig_coords.std()
            x_props.append((lig_idx, n_atoms, std))

        print(f"done.\n")
    return np.array(x_props, dtype=np.float32)


if __name__ == '__main__':
    args = parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    store_path = args.pharmit_path / f"{args.split}.zarr"
    root = zarr.open(store_path, mode='r')
    lig_node_group = root['lig/node']
    
    # lig data as zarr arrays
    atom_x = lig_node_group['x']
    graph_lookup = lig_node_group['graph_lookup']

    start_time = time.time()
    result = rand_lig_x_summary(atom_x, graph_lookup, args.n_seeds, args.batch_size)
    end_time = time.time()

    with open(args.output_dir / f'{args.split}_x_std.pkl', 'wb') as f:
        pickle.dump(result, f)

    print(f"––––––––––––––––––––––––––––––––––")
    print(f"Total molecules processed: {len(result)}")
    print(f"Total time: {end_time - start_time:.1f} seconds")
     

