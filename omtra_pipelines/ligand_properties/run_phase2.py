import argparse
from pathlib import Path
import traceback
from tqdm import tqdm
import time

from multiprocessing import Pool, Queue, Manager
from queue import Empty

from omtra.load.quick import datamodule_from_config
import omtra.load.quick as quick_load

from omtra_pipelines.ligand_properties.phase2 import *


def parse_args():
    p = argparse.ArgumentParser(description='Generate embarassingly parallel processing commands for phase2_1.py')

    p.add_argument('--pharmit_path', type=str, help='Path to the Pharmit Zarr store.', default='/net/galaxy/home/koes/ltoft/OMTRA/data/pharmit_dev')   # /net/galaxy/home/koes/icd3/moldiff/OMTRA/data/pharmit
    p.add_argument('--store_name', type=str, help='Name of the Zarr store.', default='train.zarr')
    p.add_argument('--array_name', type=str, default='extra_feats', help='Name of the new Zarr array.')
    p.add_argument('--block_size', type=int, default=10000, help='Number of ligands to process in a block.')
    p.add_argument('--n_cpus', type=int, default=2, help='Number of CPUs to use for parallel processing.')
    p.add_argument('--output_dir', type=Path, help='Output directory for processed data.', default=Path('./outputs/phase2'))

    return p.parse_args()

def worker_initializer(pharmit_path):
    global pharmit_dataset
    cfg = quick_load.load_cfg(overrides=['task_group=no_protein'], pharmit_path=pharmit_path)
    datamodule = datamodule_from_config(cfg)
    train_dataset = datamodule.load_dataset("val")
    pharmit_dataset = train_dataset.datasets['pharmit']
    

def worker_task(block_start_idx: int, block_size: int, pharmit_dataset):
    """ Task done by each worker: Get new features for the given block """
    try:
        start_time = time.time()
        new_feats, contig_idxs, failed_idxs = process_pharmit_block(block_start_idx, block_size)
        processing_time = time.time() - start_time
        return (new_feats, contig_idxs, failed_idxs, processing_time)
    
    except Exception as e:
        print(f"Worker error at block starting {block_start_idx}: {e}")
        e.traceback = traceback.format_exc()
        raise e
    

def error_and_update(error, pbar, error_counter):
    """ Handle errors, update error counter and the progress bar """

    print(f"Error: {error}")
    print(error.traceback if hasattr(error, "traceback") else traceback.format_exc())
    
    error_counter[0] += 1
    pbar.set_postfix({'errors': error_counter[0]})
    pbar.update(1)
    
    with open('error_log.txt', 'a') as f:
        f.write(f"Error:\n{error}\n")
        if hasattr(error, "traceback"):
            f.write(error.traceback)
        else:
            traceback.print_exception(type(error), error, error.__traceback__, file=f)


def run_parallel(pharmit_path: Path,
                 array_name: str,
                 block_size: int,
                 n_cpus: int,
                 block_writer: BlockWriter,
                 output_dir: Path):

    # Load Pharmit dataset (also needed for number of ligands)
    cfg = quick_load.load_cfg(overrides=['task_group=no_protein'], pharmit_path=pharmit_path)
    datamodule = datamodule_from_config(cfg)
    train_dataset = datamodule.load_dataset("val")
    pharmit_dataset = train_dataset.datasets['pharmit']

    n_mols = len(pharmit_dataset)
    n_blocks = (n_mols + block_size - 1) // block_size
    print(f"Pharmit zarr store will be processed in {n_blocks} blocks.")

    pbar = tqdm(total=n_blocks, desc="Processing", unit="blocks")
    manager = Manager()
    error_counter = manager.list([0])   # Track errors

    # Queue for worker results
    write_queue = Queue(maxsize=n_cpus * 2) # maxsize = # of returns from worker that are stored in queue

    # Start Pool
    pool = Pool(processes=n_cpus, initializer=worker_initializer, initargs=(pharmit_path,), maxtasksperchild=2)

    # Submit all worker jobs
    for block_idx in range(n_blocks):
        block_start_idx = block_idx * block_size
        pool.apply_async(worker_task,
                         args=(block_start_idx, block_size),
                         callback=lambda res: write_queue.put(('success', res)),    # put stalls if queue is full
                         error_callback=lambda err: write_queue.put(('error', err)))

    pool.close()

    # Main loop: consume results and write them safely
    finished_blocks = 0
    start_time = time.time()
    pending_blocks = n_blocks

    print("Starting result collection loop...")

    while finished_blocks < pending_blocks:
        try:
            
            status, payload = write_queue.get(timeout=120)  # Timeout in case of deadlocks or unresponsive workers

            if status == 'success':
                new_feats, contig_idxs, failed_idxs, processing_time = payload
                try:
                    write_start = time.time()
                    block_writer.save_chunk(array_name, contig_idxs, new_feats)
                    write_time = time.time() - write_start

                    print(f"[Block {finished_blocks}] "
                        f"Worker time: {processing_time:.2f}s | "
                        f"Write time: {write_time:.2f}s | "
                        f"Queue size: {write_queue.qsize()}/{write_queue._maxsize}")

                    pbar.update(1)

                    if failed_idxs:
                        with open(f"{output_dir}/failed_ligands.txt", 'a') as f:
                            for fid in failed_idxs:
                                f.write(f"{fid}\n")

                except Exception as e:
                    error_and_update(e, pbar, error_counter)

                finished_blocks += 1

            elif status == 'error':
                error = payload
                error_and_update(error, pbar, error_counter)
                finished_blocks += 1

        except Empty:
            print(f"Timeout waiting for worker result. {finished_blocks}/{pending_blocks} blocks finished.")
            break  # or continue, or retry failed block

    pool.join()
    pbar.close()

    total_time = time.time() - start_time
    print(f"Finished {finished_blocks}/{pending_blocks} blocks in {total_time:.1f} seconds")



def run_single(pharmit_path: Path,
                 array_name: str,
                 block_size: int,
                 block_writer: BlockWriter,
                 output_dir: Path):

    # Load Pharmit dataset (also needed for number of ligands)
    cfg = quick_load.load_cfg(overrides=['task_group=no_protein'], pharmit_path=pharmit_path)
    datamodule = datamodule_from_config(cfg)
    train_dataset = datamodule.load_dataset("val")
    pharmit_dataset = train_dataset.datasets['pharmit']

    n_mols = len(pharmit_dataset)
    n_blocks = (n_mols + block_size - 1) // block_size
    print(f"Pharmit zarr store will be processed in {n_blocks} blocks.")

    pbar = tqdm(total=n_blocks, desc="Processing", unit="blocks")
    error_counter = [0]   # simple error counter

    for block_idx in range(n_blocks):
        block_start_idx = block_idx * block_size        
        try:
            start_time = time.time()
            new_feats, contig_idxs, failed_idxs = process_pharmit_block(block_start_idx, block_size)
            processing_time = time.time() - start_time
            
            write_start = time.time()
            block_writer.save_chunk(array_name, contig_idxs, new_feats)
            write_time = time.time() - write_start

            print(f"[Block {block_idx}] "
                  f"Processing time: {processing_time:.2f}s | "
                  f"Write time: {write_time:.2f}s \n")

            if failed_idxs:
                with open(f"{output_dir}/failed_ligands.txt", 'a') as f:
                    for fid in failed_idxs:
                        f.write(f"{fid}\n")

        except Exception as e:
            print(f"Error processing block {block_idx}: {e}")
            error_counter[0] += 1

        pbar.update(1)

    pbar.close()

    print(f"Processing completed with {error_counter[0]} errors.")


if __name__ == '__main__':
    args = parse_args()

    store_path = args.pharmit_path+'/'+args.store_name
    block_writer = BlockWriter(store_path)

    start_time = time.time()
    run_parallel(args.pharmit_path, args.array_name, args.block_size, args.n_cpus, block_writer, args.output_dir)
    #run_single(args.pharmit_path, args.array_name, args.block_size, block_writer, args.output_dir)
    end_time = time.time()

    print(f"Total time: {end_time - start_time:.1f} seconds")

