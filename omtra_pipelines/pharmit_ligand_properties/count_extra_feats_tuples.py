import argparse
from pathlib import Path
import traceback
from tqdm import tqdm
import time
import torch

from multiprocessing import Pool, Manager
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
    p.add_argument('--block_size', type=int, default=10000, help='Number of ligands to process in a block.')
    p.add_argument('--n_cpus', type=int, default=1, help='Number of CPUs to use for parallel processing.')
    p.add_argument('--output_dir', type=Path, help='Output directory for processed data.', default=Path('omtra_pipelines/pharmit_ligand_properties/outputs/count_lig_feats'))
 
    return p.parse_args()

pharmit_dataset = None


def process_pharmit_block(block_start_idx: int, block_size: int):
    """ 
    Parameters:
        block_start_idx (int): Index of the first ligand in the block
        block_size (int): Number of ligands in the block

    Returns:
        Number of unique extra feature vectors in the block
    """
    
    global pharmit_dataset

    n_mols = len(pharmit_dataset)
    block_end_idx = min(block_start_idx + block_size, n_mols)
    block_graphs = [pharmit_dataset[('denovo_ligand_extra_feats', idx)] for idx in range(block_start_idx, block_end_idx)]
    extra_feats = torch.cat([torch.stack([g.nodes['lig'].data['a_1_true'],
                                          g.nodes['lig'].data['c_1_true'],
                                          g.nodes['lig'].data['impl_H_1_true'],
                                          g.nodes['lig'].data['aro_1_true'],
                                          g.nodes['lig'].data['hyb_1_true'],
                                          g.nodes['lig'].data['ring_1_true'],
                                          g.nodes['lig'].data['chiral_1_true']], dim=1)
                              for g in block_graphs], dim=0)
    
    unique_feats = torch.unique(extra_feats, dim=0)

    return unique_feats


def worker_initializer(pharmit_path, store_name):
    """ Sets pharmit dataset instance as a global variable """
    global pharmit_dataset

    cfg = quick_load.load_cfg(overrides=['task_group=no_protein_extra_feats'], pharmit_path=pharmit_path)
    datamodule = datamodule_from_config(cfg)
    dataset = datamodule.load_dataset(store_name)
    pharmit_dataset = dataset.datasets['pharmit']
 

def save_and_update(block_unique_feats, all_unique_feats, pbar):
    """ Callback to new features for a block and update progress """
    all_unique_feats.append(block_unique_feats)
    print(f"{torch.unique(torch.cat(list(all_unique_feats), dim=0), dim=0).shape[0]} unique features found.")
    pbar.update(1)


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


def run_parallel(pharmit_path: Path,
                 store_name: str,
                 block_size: int,
                 n_cpus: int,
                 output_dir: Path,
                 max_pending: int = None):
    
    if max_pending is None:
        max_pending = n_cpus * 2 

    # Load Pharmit dataset (also needed for number of ligands)
    cfg = quick_load.load_cfg(overrides=['task_group=no_protein_extra_feats'], pharmit_path=pharmit_path)
    datamodule = datamodule_from_config(cfg)
    dataset = datamodule.load_dataset(store_name)
    pharmit_dataset = dataset.datasets['pharmit']

    n_mols = len(pharmit_dataset)
    n_blocks = (n_mols + block_size - 1) // block_size
    print(f"Pharmit zarr store will be processed in {n_blocks} blocks.\n")

    pbar = tqdm(total=n_blocks, desc="Processing", unit="blocks")
 
    error_counter = [0]
    all_unique_feats = Manager().list()

    with Pool(processes=n_cpus, initializer=worker_initializer, initargs=(pharmit_path, store_name), maxtasksperchild=5) as pool:
        pending = []

        for block_idx in range(n_blocks):
            
            while len(pending) >= max_pending:
                # Filter out jobs that have finished
                pending = [r for r in pending if not r.ready()]
                if len(pending) >= max_pending:
                    time.sleep(0.1)
            
            callback_fn = partial(save_and_update,
                                  all_unique_feats=all_unique_feats,
                                  pbar=pbar)

            error_callback_fn = partial(error_and_update,
                                    block_idx=block_idx, 
                                    pbar=pbar,
                                    error_counter=error_counter,
                                    output_dir=output_dir)
                               
            block_start_idx = block_idx * block_size

            result = pool.apply_async(process_pharmit_block,
                                      args=(block_start_idx, block_size),
                                      callback=callback_fn,
                                      error_callback=error_callback_fn)   
            pending.append(result)

        for result in pending:
            result.wait() 

        pool.close()
        pool.join()

    print(f"Processing completed with {error_counter[0]} errors.")
    return torch.unique(torch.cat(all_unique_feats, dim=0), dim=0)



def run_single(pharmit_path: Path,
               store_name: str,
               block_size: int,
               output_dir: Path):

    # Load Pharmit dataset (also needed for number of ligands)
    global pharmit_dataset

    cfg = quick_load.load_cfg(overrides=['task_group=no_protein_extra_feats'], pharmit_path=pharmit_path)
    datamodule = datamodule_from_config(cfg)
    dataset = datamodule.load_dataset(store_name)
    pharmit_dataset = dataset.datasets['pharmit']

    n_mols = len(pharmit_dataset)
    n_blocks = (n_mols + block_size - 1) // block_size

    print(f"Pharmit zarr store will be processed in {n_blocks} blocks.\n")

    pbar = tqdm(total=n_blocks, desc="Processing", unit="blocks")
    error_counter = [0]   # simple error counter

    all_unique_feats = []

    for block_idx in range(n_blocks):
        block_start_idx = block_idx * block_size      
        try:
            result = process_pharmit_block(block_start_idx, block_size)
            all_unique_feats.append(result)

            print(f"{torch.unique(torch.cat(all_unique_feats, dim=0), dim=0).shape[0]} unique features found.")
            print(f"––––––––––––––––––––––––––––––––––––––––––")
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
    return torch.unique(torch.cat(all_unique_feats, dim=0), dim=0)


if __name__ == '__main__':
    args = parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    if args.n_cpus == 1:
        result= run_single(args.pharmit_path, args.store_name, args.block_size, args.output_dir)
    else:
        result = run_parallel(args.pharmit_path, args.store_name, args.block_size, args.n_cpus, args.output_dir)

    end_time = time.time()

    print(f"––––––––––––––––––––––––––––––––––")
    print(f"Total unique tuples: {result.shape[0]}")
    print(f"Total time: {end_time - start_time:.1f} seconds")
     

