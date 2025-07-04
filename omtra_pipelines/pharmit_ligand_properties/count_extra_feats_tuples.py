import argparse
from pathlib import Path
import traceback
from tqdm import tqdm
import time
import torch
import zarr

from multiprocessing import Pool
from functools import partial

from omtra.load.quick import datamodule_from_config
import omtra.load.quick as quick_load

from omtra_pipelines.pharmit_ligand_properties.phase2 import *

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


def parse_args():
    p = argparse.ArgumentParser(description='Compute new ligand features in parallel and save to Pharmit Zarr store.')

    p.add_argument('--pharmit_path', type=str, help='Path to the Pharmit Zarr store.', default='/net/galaxy/home/koes/ltoft/OMTRA/data/pharmit_dev')
    p.add_argument('--store_name', type=str, help='Name of the Zarr store.', default='train')
    p.add_argument('--array_name', type=str, default='extra_feats', help='Name of the new Zarr array.')
    p.add_argument('--block_size', type=int, default=100000, help='Number of ligands to process in a block.')
    p.add_argument('--n_cpus', type=int, default=1, help='Number of CPUs to use for parallel processing.')
    p.add_argument('--output_dir', type=Path, help='Output directory for processed data.', default=Path('omtra_pipelines/pharmit_ligand_properties/outputs/count_lig_feats'))
 
    return p.parse_args()

# define global variables
atom_types = None
atom_charge = None
extra_feats = None
global_unique_feats = torch.zeros((0, 7)) 

def process_pharmit_block(block_start_idx: int, block_end_idx: int):
    """ 
    Parameters:
        block_start_idx (int): Index of the first atom in the block
        block_end_idx (int): Index of the last atom in the block

    Returns:
        Number of unique extra feature vectors in the block
    """
    
    global atom_types, atom_charges, extra_feats

    atom_feats = torch.cat([torch.tensor(atom_types[block_start_idx: block_end_idx]).unsqueeze(1), 
                            torch.tensor(atom_charges[block_start_idx: block_end_idx]).unsqueeze(1),
                            torch.tensor(extra_feats[block_start_idx: block_end_idx, :-1])], dim=1)
    
    unique_feats = torch.unique(atom_feats, dim=0)

    return unique_feats


def worker_initializer(store_path):
    """ Sets pharmit dataset instance as a global variable """
    global atom_types, atom_charges, extra_feats

    root = zarr.open(store_path, mode='r+')
    lig_node_group = root['lig/node']
    atom_types = lig_node_group['a']
    atom_charges = lig_node_group['c']
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


class FeatureAggregator:
    def __init__(self):
        self.unique_atom_features = torch.zeros((0, 7))

    def save_and_update(self, block_unique_feats, pbar):
        try:
            self.unique_atom_features = torch.unique(torch.cat([self.unique_atom_features, block_unique_feats], dim=0), dim=0)
            pbar.update(1)
        except Exception as e:
            print(f"Callback error: {e}")
            traceback.print_exc()

def run_parallel(pharmit_path: Path,
                 store_name: str,
                 block_size: int,
                 n_cpus: int,
                 output_dir: Path,
                 max_pending: int = None):
    
    if max_pending is None:
        max_pending = n_cpus * 2 

    store_path = pharmit_path+'/'+store_name+'.zarr'
    root = zarr.open(store_path, mode='r+')

    n_atoms = root['lig/node/a'].shape[0]
    n_blocks = (n_atoms + block_size - 1) // block_size

    print(f"Pharmit zarr store will be processed in {n_blocks} blocks.\n")
    print(f"––––––––––––––––––––––––––––––––––")

    aggregator = FeatureAggregator()
    pbar = tqdm(total=n_blocks, desc="Processing", unit="blocks")
    error_counter = [0]
    
    with Pool(processes=n_cpus, initializer=worker_initializer, initargs=(store_path,), maxtasksperchild=5) as pool:
        pending = []

        for block_idx in range(n_blocks):
            
            while len(pending) >= max_pending:
                # Filter out jobs that have finished
                pending = [r for r in pending if not r.ready()]
                if len(pending) >= max_pending:
                    time.sleep(0.1)
            
            callback_fn = partial(aggregator.save_and_update, pbar=pbar)

            error_callback_fn = partial(error_and_update,
                                    block_idx=block_idx, 
                                    pbar=pbar,
                                    error_counter=error_counter,
                                    output_dir=output_dir)
                               
            block_start_idx = block_idx * block_size

            result = pool.apply_async(process_pharmit_block,
                                      args=(block_start_idx, min(block_start_idx + block_size, n_atoms)),
                                      callback=callback_fn,
                                      error_callback=error_callback_fn)   
            pending.append(result)

        for result in pending:
            result.wait() 

        pool.close()
        pool.join()

    print(f"Processing completed with {error_counter[0]} errors.")
    return aggregator.unique_atom_features



def run_single(pharmit_path: Path,
               store_name: str,
               block_size: int,
               output_dir: Path):

    global atom_types, atom_charges, extra_feats
    global global_unique_feats

    store_path = store_path = pharmit_path+'/'+store_name+'.zarr'
    root = zarr.open(store_path, mode='r+')
    lig_node_group = root['lig/node']
    atom_types = lig_node_group['a']
    atom_charges = lig_node_group['c']
    extra_feats = lig_node_group['extra_feats']

    n_atoms = atom_types.shape[0]
    n_blocks = (n_atoms + block_size - 1) // block_size

    print(f"Pharmit zarr store will be processed in {n_blocks} blocks.\n")

    pbar = tqdm(total=n_blocks, desc="Processing", unit="blocks")
    error_counter = [0]   # simple error counter

    for block_idx in range(n_blocks):
        block_start_idx = block_idx * block_size      
        try:
            result = process_pharmit_block(block_start_idx, min(block_start_idx + block_size, n_atoms))
            global_unique_feats = torch.unique(torch.cat([global_unique_feats, result], dim=0), dim=0)

            print(f"{global_unique_feats.shape[0]} unique atom feature tuples found so far.")
            print(f"––––––––––––––––––––––––––––––––––")

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

        pbar.update(1)

    pbar.close()

    print(f"Processing completed with {error_counter[0]} errors.")
    return global_unique_feats


if __name__ == '__main__':
    args = parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    if args.n_cpus == 1:
        result = run_single(args.pharmit_path, args.store_name, args.block_size, args.output_dir)
    else:
        result = run_parallel(args.pharmit_path, args.store_name, args.block_size, args.n_cpus, args.output_dir)

    torch.save(result, args.output_dir+'/unique_feature_tuples.pt')

    end_time = time.time()

    print(f"––––––––––––––––––––––––––––––––––")
    print(f"Total unique tuples: {result.shape[0]}")
    print(f"Total time: {end_time - start_time:.1f} seconds")
     

