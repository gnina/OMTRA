import argparse
from pathlib import Path
import traceback
from tqdm import tqdm
import time

from multiprocessing import Pool
from functools import partial

from omtra.load.quick import datamodule_from_config
import omtra.load.quick as quick_load

from omtra_pipelines.plinder_ligand_properties.phase2 import *

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


def parse_args():
    p = argparse.ArgumentParser(description='Compute new ligand features in parallel and save to Zarr store.')

    p.add_argument('--plinder_path', type=str, help='Path to the Plinder/PDBbind Zarr store root.', default='/net/galaxy/home/koes/ltoft/OMTRA/data/plinder')
    p.add_argument('--store_name', type=str, help='Name of the Zarr store split.', default='train')
    p.add_argument('--array_name', type=str, default='extra_feats', help='Name of the new Zarr array.')
    p.add_argument('--block_size', type=int, default=5000, help='Number of ligands to process in a block.')
    p.add_argument('--n_cpus', type=int, default=2, help='Number of CPUs to use for parallel processing.')
    p.add_argument('--output_dir', type=Path, help='Output directory for processed data.', default=Path('./outputs/phase2'))
    p.add_argument('--dataset', type=str, choices=['plinder', 'pdbbind'], default='plinder', help='Dataset type.')
    p.add_argument('--flat', action='store_true', help='Store path is {plinder_path}/{store_name}.zarr (PDBbind layout).')
    p.add_argument('--task', type=str, default='denovo_ligand', help='Task used to load ligand graphs (must not require extra_feats).')
    p.add_argument('--pdbbind_path', type=str, default=None, help='PDBbind data root (parent of pred/). Defaults to plinder_path parent when --dataset pdbbind.')

    return p.parse_args()


target_dataset = None
task_name = 'denovo_ligand'


def store_path_for(args, version=None):
    if args.flat:
        return f"{args.plinder_path}/{args.store_name}.zarr"
    return f"{args.plinder_path}/{version}/{args.store_name}.zarr"


def versions_for(args):
    if args.flat:
        return [None]
    return ['exp', 'no_links', 'pred']


def load_target_dataset(args, version, store_name):
    if args.dataset == 'pdbbind':
        pdbbind_path = args.pdbbind_path or str(Path(args.plinder_path).parent)
        cfg = quick_load.load_cfg(
            overrides=['task_group=pdbbind_predapo', 'fake_atom_p=0.0'],
            pdbbind_path=pdbbind_path,
        )
        datamodule = datamodule_from_config(cfg)
        dataset = datamodule.load_dataset(store_name)
        link_version = version or 'pred'
        return dataset.datasets['pdbbind'][link_version]

    cfg = quick_load.load_cfg(
        overrides=['task_group=protein', 'fake_atom_p=0.0'],
        plinder_path=args.plinder_path,
    )
    datamodule = datamodule_from_config(cfg)
    dataset = datamodule.load_dataset(store_name)
    return dataset.datasets['plinder'][version]


def process_block(block_start_idx: int, block_size: int):
    global target_dataset, task_name

    n_mols = len(target_dataset)
    block_end_idx = min(block_start_idx + block_size, n_mols)

    contig_idxs = []
    new_feats = []
    failed_idxs = []

    cur_contig_feats = []
    contig_start_idx = None
    contig_end_idx = None

    for idx in range(block_start_idx, block_end_idx):
        start_idx, end_idx = target_dataset.retrieve_atom_idxs(idx)

        try:
            g = target_dataset[(task_name, idx)]
            mol = dgl_to_rdkit(g, task_name=task_name)
            Chem.SanitizeMol(mol)

            atom_props = ligand_properties(mol)
            fragments = fragment_molecule(mol)
            atom_props = np.concatenate((atom_props, fragments), axis=1)

            assert atom_props.shape[0] == (end_idx - start_idx), (
                f"Mismatch in atom counts: computed properties for {atom_props.shape[0]} "
                f"atoms but expected {(end_idx - start_idx)}"
            )

            if contig_start_idx is None:
                contig_start_idx = start_idx

            cur_contig_feats.append(atom_props)
            contig_end_idx = end_idx

        except Exception as e:
            print(f"Failed to compute features for molecule {idx}: {e}. Creating new contig from {contig_start_idx}-{contig_end_idx}")
            failed_idxs.append(idx)

            if cur_contig_feats:
                feat_array = np.vstack(cur_contig_feats)
                contig_idxs.append((contig_start_idx, contig_end_idx))
                new_feats.append(feat_array)

                cur_contig_feats = []
                contig_start_idx = None
                contig_end_idx = None

    if cur_contig_feats:
        atom_props = np.vstack(cur_contig_feats)
        contig_idxs.append((contig_start_idx, contig_end_idx))
        new_feats.append(atom_props)

    return new_feats, contig_idxs, failed_idxs


def worker_initializer(args, version, store_name, load_task_name):
    global target_dataset, task_name
    task_name = load_task_name
    target_dataset = load_target_dataset(args, version, store_name)


def save_and_update(result, block_writer, pbar, output_dir):
    new_feats, contig_idxs, failed_idxs = result

    try:
        block_writer.save_chunk(contig_idxs, new_feats)
    except Exception as e:
        print(f"Error during save_chunk: {e}")
        raise

    if failed_idxs:
        failed_path = Path(output_dir) / "failed_ligands.txt"
        with open(failed_path, 'a') as f:
            for fid in failed_idxs:
                f.write(f"{fid}\n")

    pbar.update(1)


def error_and_update(error, pbar, error_counter, output_dir):
    print(f"Error: {error}")
    print(error.traceback if hasattr(error, "traceback") else traceback.format_exc())

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


def run_parallel(args, version, store_name, block_size, n_cpus, block_writer, output_dir, max_pending=None):
    global target_dataset

    if max_pending is None:
        max_pending = n_cpus * 2

    target_dataset = load_target_dataset(args, version, store_name)

    n_mols = len(target_dataset)
    n_blocks = (n_mols + block_size - 1) // block_size
    print(f"Zarr store will be processed in {n_blocks} blocks.\n")

    pbar = tqdm(total=n_blocks, desc="Processing", unit="blocks")
    error_counter = [0]

    with Pool(
        processes=n_cpus,
        initializer=worker_initializer,
        initargs=(args, version, store_name, args.task),
        maxtasksperchild=5,
    ) as pool:
        pending = []

        for block_idx in range(n_blocks):
            while len(pending) >= max_pending:
                pending = [r for r in pending if not r.ready()]
                if len(pending) >= max_pending:
                    time.sleep(0.1)

            callback_fn = partial(
                save_and_update,
                block_writer=block_writer,
                pbar=pbar,
                output_dir=output_dir,
            )

            error_callback_fn = partial(
                error_and_update,
                pbar=pbar,
                error_counter=error_counter,
                output_dir=output_dir,
            )

            block_start_idx = block_idx * block_size

            result = pool.apply_async(
                process_block,
                args=(block_start_idx, block_size),
                callback=callback_fn,
                error_callback=error_callback_fn,
            )
            pending.append(result)

        for result in pending:
            result.wait()

        pool.close()
        pool.join()

    print(f"Processing completed with {error_counter[0]} errors.")


def run_single(args, version, store_name, block_size, block_writer, output_dir):
    global target_dataset, task_name

    task_name = args.task
    target_dataset = load_target_dataset(args, version, store_name)

    n_mols = len(target_dataset)
    n_blocks = (n_mols + block_size - 1) // block_size
    print(f"Zarr store will be processed in {n_blocks} blocks.")

    pbar = tqdm(total=n_blocks, desc="Processing", unit="blocks")
    error_counter = [0]

    for block_idx in range(n_blocks):
        block_start_idx = block_idx * block_size
        try:
            start_time = time.time()
            new_feats, contig_idxs, failed_idxs = process_block(block_start_idx, block_size)
            processing_time = time.time() - start_time

            write_start = time.time()
            block_writer.save_chunk(contig_idxs, new_feats)
            write_time = time.time() - write_start

            print(
                f"[Block {block_idx}] "
                f"Processing time: {processing_time:.2f}s | "
                f"Write time: {write_time:.2f}s \n"
            )

            if failed_idxs:
                failed_path = Path(output_dir) / 'failed_ligands.txt'
                with open(failed_path, 'a') as f:
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for version in versions_for(args):
        label = 'flat' if args.flat else version
        print("\n–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––")
        print(f"Computing new ligand features for {args.dataset} store '{label}'.")
        print("–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––\n")

        store_path = store_path_for(args, version)
        block_writer = BlockWriter(store_path, args.array_name)

        start_time = time.time()

        if args.n_cpus == 1:
            run_single(args, version, args.store_name, args.block_size, block_writer, args.output_dir)
        else:
            run_parallel(
                args,
                version,
                args.store_name,
                args.block_size,
                args.n_cpus,
                block_writer,
                args.output_dir,
            )

        end_time = time.time()
        print(f"Processed in {end_time - start_time:.1f} seconds")
