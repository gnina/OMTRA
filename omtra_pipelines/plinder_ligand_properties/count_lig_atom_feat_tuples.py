import argparse
from pathlib import Path
import traceback
from tqdm import tqdm
import time
import torch
import zarr
import pickle

from multiprocessing import Pool
from functools import partial

import multiprocessing
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(description='Get ligand atom feature tuple counts in Plinder dataset.')

    p.add_argument('--plinder_path', type=str, help='Path to the Plinder Zarr store.', default='/net/galaxy/home/koes/ltoft/OMTRA/data/plinder')
    p.add_argument('--split', type=str, help='Data split: train or val.', default='train')
    p.add_argument('--array_name', type=str, default='extra_feats', help='Name of the new Zarr array.')
    p.add_argument('--block_size', type=int, default=1000000, help='Number of ligands to process in a block.')
    p.add_argument('--n_cpus', type=int, default=1, help='Number of CPUs to use for parallel processing.')
    p.add_argument('--output_dir', type=Path, help='Output directory for processed data.', default=Path('omtra_pipelines/plinder_ligand_properties/outputs/count_lig_feats'))
 
    return p.parse_args()


multiprocessing.set_start_method('spawn', force=True)

# Global variables for zarr arrays
atom_types = None
atom_charge = None
extra_feats = None

def process_plinder_block(block_start_idx: int, block_end_idx: int):
    """ 
    Parameters:
        block_start_idx (int): Index of the first atom in the block
        block_end_idx (int): Index of the last atom in the block

    Returns:
       Tensor of unique feature tuples in the block
    """
    global atom_types, atom_charges, extra_feats

    atom_feats = torch.cat([torch.tensor(atom_types[block_start_idx: block_end_idx].copy()).unsqueeze(1), 
                            torch.tensor(atom_charges[block_start_idx: block_end_idx].copy()).unsqueeze(1),
                            torch.tensor(extra_feats[block_start_idx: block_end_idx, :-1].copy())], dim=1)
    
    # Get counts of each unique tuple
    feat_counts = defaultdict(int)
    for row in atom_feats:
        key = tuple(row.tolist())
        feat_counts[key] += 1

    return dict(feat_counts)


def worker_initializer(store_path):
    """ Sets plinder dataset instance as a global variable """

    global atom_types, atom_charges, extra_feats

    root = zarr.open(store_path, mode='r')
    lig_node_group = root['ligand']
    atom_types = lig_node_group['atom_types']
    atom_charges = lig_node_group['atom_charges']
    extra_feats = lig_node_group['extra_feats']


def error_and_update(error, block_idx, pbar, error_counter, output_dir):
    """ Handle errors, update error counter, and the progress bar """
    
    error_counter[0] += 1
    pbar.set_postfix({'errors': error_counter[0]})
    print(f"Error!: {error}")
    
    error_log_path = output_dir / 'error_log.txt'

    with open(error_log_path, 'a') as f:
        f.write(f"Error in block {block_idx}:\n{error}\n")
        if hasattr(error, "traceback"):
            f.write(error.traceback)
        else:
            traceback.print_exception(type(error), error, error.__traceback__, file=f)
    
    pbar.update(1)


def save_and_update(result, feature_aggregator, block_idx, pbar, output_dir):
    try:
        feature_aggregator.save_and_update(result)

    except Exception as error:
            print(f"Error saving block #{block_idx}: {error}")
            error_log_path = output_dir / 'error_log.txt'

            with open(error_log_path, 'a') as f:
                f.write(f"Error updating:\n{error}\n")
                if hasattr(error, "traceback"):
                    f.write(error.traceback)
                else:
                    traceback.print_exception(type(error), error, error.__traceback__, file=f)

    pbar.update(1)


class FeatureAggregator:
    def __init__(self):
        self.unique_atom_features = defaultdict(int)

    def save_and_update(self, block_unique_feats):
        for feat, count in block_unique_feats.items():
            self.unique_atom_features[feat] += count
        


def run_parallel(plinder_path: str,
                 split: str,
                 block_size: int,
                 n_cpus: int,
                 output_dir: Path,
                 max_pending: int = None):
    
    if max_pending is None:
        max_pending = n_cpus * 5 

    store_path = f"{plinder_path}/{split}.zarr"
    root = zarr.open(store_path, mode='r')

    n_atoms = root['ligand/atom_types'].shape[0]
    n_blocks = (n_atoms + block_size - 1) // block_size

    pbar = tqdm(total=n_blocks, desc="Processing", unit="blocks")
    aggregator = FeatureAggregator()
    error_counter = [0]

    print(f"Plinder zarr store will be processed in {n_blocks} blocks.\n", flush=True)
    print(f"––––––––––––––––––––––––––––––––––", flush=True)
    
    with Pool(processes=n_cpus, initializer=worker_initializer, initargs=(store_path,), maxtasksperchild=5) as pool:
        pending = []

        for block_idx in range(n_blocks):
            
            # prevent adding more jobs until we are waiting for <max_pending jobs to complete
            while len(pending) >= max_pending:
                pending = [r for r in pending if not r.ready()]
                if len(pending) >= max_pending:
                    time.sleep(0.1)
            
            error_callback_fn = partial(error_and_update,
                                    block_idx=block_idx, 
                                    error_counter=error_counter,
                                    output_dir=output_dir)
        
            callback_fn = partial(save_and_update,
                                  feature_aggregator=aggregator,
                                  block_idx=block_idx,
                                  pbar=pbar,
                                  output_dir=output_dir
                                  )
                          
            block_start_idx = block_idx * block_size
            block_end_idx = min(block_start_idx + block_size, n_atoms)

            result = pool.apply_async(process_plinder_block,
                                      args=(block_start_idx, block_end_idx),
                                      callback=callback_fn,
                                      error_callback=error_callback_fn)   
            pending.append(result)

        # block main process until all async jobs have finished
        for result in pending:
            result.wait()

    print(f"Processing completed with {error_counter[0]} errors.")
    return aggregator.unique_atom_features


def run_single(plinder_path: str,
               split: str,
               block_size: int,
               output_dir: Path):

    store_path = f"{plinder_path}/{split}.zarr"
    root = zarr.open(store_path, mode='r')

    n_atoms = root['ligand/atom_types'].shape[0]
    n_blocks = (n_atoms + block_size - 1) // block_size

    pbar = tqdm(total=n_blocks, desc="Processing", unit="blocks")
    aggregator = FeatureAggregator()
    error_counter = [0]   # error counter

    worker_initializer(store_path)  # initialize global variables for zarr arrays

    print(f"Plinder zarr store will be processed in {n_blocks} blocks.\n")
    print(f"––––––––––––––––––––––––––––––––––")

    for block_idx in range(n_blocks):
        block_start_idx = block_idx * block_size 

        callback_fn = partial(save_and_update,
                                  feature_aggregator=aggregator,
                                  block_idx=block_idx,
                                  pbar=pbar,
                                  output_dir=output_dir
                                  )     
        try:
            result = process_plinder_block(block_start_idx, min(block_start_idx + block_size, n_atoms))
            callback_fn(result)

        except Exception as e:
            print(f"Error processing block {block_idx}: {e}")
            error_log_path = output_dir / 'error_log.txt'

            with open(error_log_path, 'a') as f:
                f.write(f"Error in block {block_idx}:\n{e}\n")
                if hasattr(e, "traceback"):
                    f.write(e.traceback)
                else:
                    traceback.print_exception(type(e), e, e.__traceback__, file=f)
                    error_counter[0] += 1

    pbar.close()

    print(f"Processing completed with {error_counter[0]} errors.")
    return aggregator.unique_atom_features


if __name__ == '__main__':
    args = parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tuple_counts = {}

    start_time = time.time()

    for version in ['exp', 'no_links', 'pred']:

        plinder_path = f"{args.plinder_path}/{version}"

        if args.n_cpus == 1:
            result = run_single(plinder_path, args.split, args.block_size, args.output_dir)
        else:
            result = run_parallel(plinder_path, args.split, args.block_size, args.n_cpus, args.output_dir)
        
        tuple_counts[version] = result


    with open(f'{args.output_dir}/plinder_{args.split}_condensed_atom_types.pkl', 'wb') as f:
        pickle.dump(tuple_counts, f)

    end_time = time.time()

    print(f"––––––––––––––––––––––––––––––––––")
    print(f"Total unique tuples: {len(result)}")
    print(f"Total time: {end_time - start_time:.1f} seconds")
     

