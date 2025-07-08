import argparse
from pathlib import Path
import traceback
from tqdm import tqdm
import time
import torch
import zarr

from multiprocessing import Pool
from functools import partial

import multiprocessing
import os


def parse_args():
    p = argparse.ArgumentParser(description='Compute new ligand features in parallel and save to Pharmit Zarr store.')

    p.add_argument('--pharmit_path', type=str, help='Path to the Pharmit Zarr store.', default='/net/galaxy/home/koes/ltoft/OMTRA/data/pharmit_dev')
    p.add_argument('--split', type=str, help='Data split: train or val.', default='train')
    p.add_argument('--array_name', type=str, default='extra_feats', help='Name of the new Zarr array.')
    p.add_argument('--block_size', type=int, default=5000000, help='Number of ligands to process in a block.')
    p.add_argument('--n_cpus', type=int, default=1, help='Number of CPUs to use for parallel processing.')
    p.add_argument('--output_dir', type=Path, help='Output directory for processed data.', default=Path('omtra_pipelines/pharmit_ligand_properties/outputs/count_lig_feats'))
 
    return p.parse_args()


multiprocessing.set_start_method('spawn', force=True)

def process_pharmit_block(store_path: str, block_start_idx: int, block_end_idx: int):
    """ 
    Parameters:
        block_start_idx (int): Index of the first atom in the block
        block_end_idx (int): Index of the last atom in the block

    Returns:
        Number of unique extra feature vectors in the block
    """
    
    root = zarr.open(store_path, mode='r')
    lig_node_group = root['lig/node']
    atom_types = lig_node_group['a']
    atom_charges = lig_node_group['c']
    extra_feats = lig_node_group['extra_feats']

    atom_feats = torch.cat([torch.tensor(atom_types[block_start_idx: block_end_idx].copy()).unsqueeze(1), 
                            torch.tensor(atom_charges[block_start_idx: block_end_idx].copy()).unsqueeze(1),
                            torch.tensor(extra_feats[block_start_idx: block_end_idx, :-1].copy())], dim=1)
    
    unique_feats = torch.unique(atom_feats, dim=0)

    return unique_feats


# def worker_initializer(store_path):
#     """ Sets pharmit dataset instance as a global variable """
#     global atom_types, atom_charges, extra_feats
    # root = zarr.open(store_path, mode='r')
    # lig_node_group = root['lig/node']
    # atom_types = lig_node_group['a']
    # atom_charges = lig_node_group['c']
    # extra_feats = lig_node_group['extra_feats']


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
    def __init__(self, pbar, output_dir):
        self.unique_atom_features = torch.zeros((0, 7))
        self.pbar = pbar
        self.output_dir = output_dir

    def save_and_update(self, block_unique_feats):
        try:
            self.unique_atom_features = torch.unique(torch.cat([self.unique_atom_features, block_unique_feats], dim=0), dim=0)
            self.pbar.update(1)

        except Exception as error:
            error_log_path = self.output_dir / 'error_log.txt'

            with open(error_log_path, 'a') as f:
                f.write(f"Error updating:\n{error}\n")
                if hasattr(error, "traceback"):
                    f.write(error.traceback)
                else:
                    traceback.print_exception(type(error), error, error.__traceback__, file=f)
            
            self.pbar.update(1)


def run_parallel(pharmit_path: str,
                 split: str,
                 block_size: int,
                 n_cpus: int,
                 output_dir: Path,
                 max_pending: int = None):
    
    if max_pending is None:
        max_pending = n_cpus * 5 

    store_path = pharmit_path+'/'+split+'.zarr'
    root = zarr.open(store_path, mode='r')

    n_atoms = root['lig/node/a'].shape[0]
    n_blocks = (n_atoms + block_size - 1) // block_size

    print(f"Pharmit zarr store will be processed in {n_blocks} blocks.\n")
    print(f"––––––––––––––––––––––––––––––––––")

    pbar = tqdm(total=n_blocks, desc="Processing", unit="blocks")
    aggregator = FeatureAggregator(pbar, output_dir)
    error_counter = [0]
    
    with Pool(processes=n_cpus, maxtasksperchild=5) as pool:
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
                               
            block_start_idx = block_idx * block_size
            block_end_idx = min(block_start_idx + block_size, n_atoms)

            result = pool.apply_async(process_pharmit_block,
                                      args=(store_path, block_start_idx, block_end_idx),
                                      callback=aggregator.save_and_update,
                                      error_callback=error_callback_fn)   

            pending.append(result)

        # block main process until all async jobs have finished
        for result in pending:
            result.wait()

    print(f"Processing completed with {error_counter[0]} errors.")
    return aggregator.unique_atom_features


def run_single(pharmit_path: str,
               split: str,
               block_size: int,
               output_dir: Path):

    store_path = store_path = pharmit_path+'/'+split+'.zarr'
    root = zarr.open(store_path, mode='r')

    n_atoms = root['lig/node/a'].shape[0]
    n_blocks = (n_atoms + block_size - 1) // block_size

    print(f"Pharmit zarr store will be processed in {n_blocks} blocks.\n")

    pbar = tqdm(total=n_blocks, desc="Processing", unit="blocks")
    aggregator = FeatureAggregator(pbar, output_dir)
    error_counter = [0]   # simple error counter

    for block_idx in range(n_blocks):
        block_start_idx = block_idx * block_size      
        try:
            result = process_pharmit_block(store_path, block_start_idx, min(block_start_idx + block_size, n_atoms))
            aggregator.save_and_update(result)

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
    return aggregator.unique_atom_features


if __name__ == '__main__':
    args = parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    if args.n_cpus == 1:
        result = run_single(args.pharmit_path, args.split, args.block_size, args.output_dir)
    else:
        result = run_parallel(args.pharmit_path, args.split, args.block_size, args.n_cpus, args.output_dir)

    torch.save(result, args.output_dir / f'{args.split}_unique_feat_tuples.pt')

    end_time = time.time()

    print(f"––––––––––––––––––––––––––––––––––")
    print(f"Total unique tuples: {result.shape[0]}")
    print(f"Total time: {end_time - start_time:.1f} seconds")
     

