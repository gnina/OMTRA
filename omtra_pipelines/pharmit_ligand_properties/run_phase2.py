import argparse
from pathlib import Path
import traceback
from tqdm import tqdm
import time
import os

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
    p.add_argument('--block_size', type=int, default=10000, help='Number of ligands to process in a block.')
    p.add_argument('--n_cpus', type=int, default=2, help='Number of CPUs to use for parallel processing.')
    p.add_argument('--output_dir', type=Path, help='Output directory for processed data.', default=Path('./outputs/phase2'))
 
    return p.parse_args()


pharmit_dataset = None


def process_pharmit_block(block_start_idx: int, block_size: int):
    """ 
    Parameters:
        block_start_idx (int): Index of the first ligand in the block
        block_size (int): Number of ligands in the block

    Returns:
        new_feats (List[np.ndarray]): Feature arrays per contiguous atom block.
        contig_idxs (List[Tuple[int, int]]): Start/end atom indices for each contiguous block.
        failed_idxs (List[int]): Indices of ligands that failed processing.
    """
    
    global pharmit_dataset

    n_mols = len(pharmit_dataset)
    block_end_idx = min(block_start_idx + block_size, n_mols)

    contig_idxs = []
    new_feats = []
    failed_idxs = []

    cur_contig_feats = []
    contig_start_idx = None
    contig_end_idx = None

    for idx in range(block_start_idx, block_end_idx):
        
        start_idx, end_idx = pharmit_dataset.retrieve_atom_idxs(idx)

        try:
            g = pharmit_dataset[('denovo_ligand', idx)]
            mol = dgl_to_rdkit(g)
            Chem.SanitizeMol(mol)

            atom_props = ligand_properties(mol)                         # (n_atoms, 5)
            fragments = fragment_molecule(mol)                          # (n_atoms, 1)
            atom_props = np.concatenate((atom_props, fragments), axis=1)# (n_atoms, 6)

            assert atom_props.shape[0] == (end_idx - start_idx), f"Mismatch in atom counts: computed properties for {atom_props.shape[0]} atoms but expected {(end_idx - start_idx)}"

            if contig_start_idx is None:
                contig_start_idx = start_idx

            cur_contig_feats.append(atom_props)
            contig_end_idx = end_idx  # always update with latest good molecule

        except Exception as e:
            print(f"Failed to compute features for molecule {idx}: {e}.")
            failed_idxs.append(idx)

            # Close current contiguous chunk (if any)
            if cur_contig_feats:
                feat_array = np.vstack(cur_contig_feats)
                contig_idxs.append((contig_start_idx, contig_end_idx))
                new_feats.append(feat_array)

                # Reset
                cur_contig_feats = []
                contig_start_idx = None
                contig_end_idx = None

    # After final molecule, flush last chunk if present
    if cur_contig_feats:
        atom_props = np.vstack(cur_contig_feats)
        contig_idxs.append((contig_start_idx, contig_end_idx))
        new_feats.append(atom_props)

    return new_feats, contig_idxs, failed_idxs


def worker_initializer(pharmit_path, store_name):
    """ Sets pharmit dataset instance as a global variable """
    global pharmit_dataset

    cfg = quick_load.load_cfg(overrides=['task_group=no_protein', 'fake_atom_p=0.0'], pharmit_path=pharmit_path)
    datamodule = datamodule_from_config(cfg)
    dataset = datamodule.load_dataset(store_name)
    pharmit_dataset = dataset.datasets['pharmit']
 

def save_and_update(result, block_writer, pbar, output_dir):
    """ Callback to new features for a block and update progress """

    new_feats, contig_idxs, failed_idxs = result

    try:
        block_writer.save_chunk(contig_idxs, new_feats)
    except Exception as e:
        print(f"Error during save_chunk: {e}")
        raise

    if failed_idxs:
        failed_path = output_dir / "failed_ligands.txt"
        with open(failed_path, 'a') as f:
            for fid in failed_idxs:
                f.write(f"{fid}\n")
    pbar.update(1)


def error_and_update(error, pbar, error_counter, output_dir):
    """ Handle errors, update error counter, and the progress bar """
    
    error_counter[0] += 1
    pbar.set_postfix({'errors': error_counter[0]})
    
    error_log_path = output_dir / 'error_log.txt'
    with open(error_log_path, 'a') as f:
        f.write(f"Error:\n{error}\n")
        if hasattr(error, "traceback"):
            f.write(error.traceback)
        else:
            traceback.print_exception(type(error), error, error.__traceback__, file=f)
    
    pbar.update(1)


def run_parallel(pharmit_path: Path,
                 store_name: str,
                 block_size: int,
                 n_cpus: int,
                 block_writer: BlockWriter,
                 output_dir: Path,
                 max_pending: int = None):
    
    if max_pending is None:
        max_pending = n_cpus * 2 

    # Load Pharmit dataset (also needed for number of ligands)
    cfg = quick_load.load_cfg(overrides=['task_group=no_protein'], pharmit_path=pharmit_path)
    datamodule = datamodule_from_config(cfg)
    dataset = datamodule.load_dataset(store_name)
    pharmit_dataset = dataset.datasets['pharmit']

    n_mols = len(pharmit_dataset)
    n_blocks = (n_mols + block_size - 1) // block_size
    print(f"Pharmit zarr store will be processed in {n_blocks} blocks.\n")

    pbar = tqdm(total=n_blocks, desc="Processing", unit="blocks")
    error_counter = [0]

    with Pool(processes=n_cpus, initializer=worker_initializer, initargs=(pharmit_path, store_name), maxtasksperchild=5) as pool:
        pending = []

        for block_idx in range(n_blocks):
            
            while len(pending) >= max_pending:
                pending = [r for r in pending if not r.ready()]
                if len(pending) >= max_pending:
                    time.sleep(0.1)
            
            callback_fn = partial(save_and_update,
                              block_writer=block_writer,
                              pbar=pbar,
                              output_dir=output_dir)

            error_callback_fn = partial(error_and_update, 
                                    pbar=pbar,
                                    error_counter=error_counter)
                               
            block_start_idx = block_idx * block_size

            result = pool.apply_async(process_pharmit_block,
                                      args=(block_start_idx, block_size),
                                      callback=callback_fn,
                                      error_callback=error_callback_fn)   
            pending.append(result)

        # block main process until all async jobs have finished
        for result in pending:
            result.wait() 

        pool.close()
        pool.join()

    print(f"Processing completed with {error_counter[0]} errors.")



def run_single(pharmit_path: Path,
               store_name: str,
               block_size: int,
               block_writer: BlockWriter,
               output_dir: Path):

    # Load Pharmit dataset (also needed for number of ligands)
    global pharmit_dataset

    cfg = quick_load.load_cfg(overrides=['task_group=no_protein', 'fake_atom_p=0.0'], pharmit_path=pharmit_path)
    datamodule = datamodule_from_config(cfg)
    dataset = datamodule.load_dataset(store_name)
    pharmit_dataset = dataset.datasets['pharmit']

    n_mols = len(pharmit_dataset)
    n_blocks = (n_mols + block_size - 1) // block_size
    print(f"Pharmit zarr store will be processed in {n_blocks} blocks.")

    pbar = tqdm(total=n_blocks, desc="Processing", unit="blocks")
    error_counter = [0]   # simple error counter

    for block_idx in range(n_blocks):
        block_start_idx = block_idx * block_size      
        try:
            start_time = time.time()
            result = process_pharmit_block(block_start_idx, block_size)
            processing_time = time.time() - start_time
            
            write_start = time.time()
            save_and_update(result, block_writer, pbar, output_dir)
            write_time = time.time() - write_start

            print(f"[Block {block_idx}] "
                  f"Processing time: {processing_time:.2f}s | "
                  f"Write time: {write_time:.2f}s \n")
        except Exception as e:
            print(f"Error processing block {block_idx}: {e}")
            error_counter[0] += 1

        pbar.update(1)

    pbar.close()

    print(f"Processing completed with {error_counter[0]} errors.")


if __name__ == '__main__':
    args = parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    store_path = args.pharmit_path+'/'+args.store_name+'.zarr'
    block_writer = BlockWriter(store_path, args.array_name)

    start_time = time.time()

    if args.n_cpus == 1:
        run_single(args.pharmit_path, args.store_name, args.block_size, block_writer, args.output_dir)
    else:
        run_parallel(args.pharmit_path, args.store_name, args.block_size, args.n_cpus, block_writer, args.output_dir)

    end_time = time.time()

    print(f"Total time: {end_time - start_time:.1f} seconds")

