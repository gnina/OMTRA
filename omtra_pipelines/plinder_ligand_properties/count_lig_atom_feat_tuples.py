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
    p = argparse.ArgumentParser(description='Get ligand atom feature tuple counts in a Zarr store.')

    p.add_argument('--plinder_path', type=str, help='Path to the Zarr store root.', default='/net/galaxy/home/koes/ltoft/OMTRA/data/plinder')
    p.add_argument('--split', type=str, help='Data split when using a single split.', default='train')
    p.add_argument('--splits', nargs='*', default=None, help='Multiple splits to merge (e.g. train val test).')
    p.add_argument('--array_name', type=str, default='extra_feats', help='Name of the extra feats Zarr array.')
    p.add_argument('--block_size', type=int, default=1000000, help='Number of atoms to process in a block.')
    p.add_argument('--n_cpus', type=int, default=1, help='Number of CPUs to use for parallel processing.')
    p.add_argument('--output_dir', type=Path, help='Output directory for processed data.', default=Path('omtra_pipelines/plinder_ligand_properties/outputs/count_lig_feats'))
    p.add_argument('--flat', action='store_true', help='Store path is {plinder_path}/{split}.zarr (PDBbind layout).')
    p.add_argument('--versions', nargs='*', default=None, help='Link versions to process (default: exp no_links pred). Ignored when --flat is set.')
    p.add_argument('--output_name', type=str, default=None, help='Output pickle basename (without .pkl).')
    p.add_argument('--check_vocab', action='store_true', help='Report tuples not in Pharmit+Plinder vocab.')

    return p.parse_args()


multiprocessing.set_start_method('spawn', force=True)

atom_types = None
atom_charges = None
extra_feats = None


def store_path_for(plinder_path, split, version=None, flat=False):
    if flat:
        return f"{plinder_path}/{split}.zarr"
    return f"{plinder_path}/{version}/{split}.zarr"


def process_block(block_start_idx: int, block_end_idx: int):
    global atom_types, atom_charges, extra_feats

    atom_feats = torch.cat([
        torch.tensor(atom_types[block_start_idx: block_end_idx].copy()).unsqueeze(1),
        torch.tensor(atom_charges[block_start_idx: block_end_idx].copy()).unsqueeze(1),
        torch.tensor(extra_feats[block_start_idx: block_end_idx, :-1].copy()),
    ], dim=1)

    feat_counts = defaultdict(int)
    for row in atom_feats:
        key = tuple(int(x) for x in row.tolist())
        feat_counts[key] += 1

    return dict(feat_counts)


def worker_initializer(store_path):
    global atom_types, atom_charges, extra_feats

    root = zarr.open(store_path, mode='r')
    lig_node_group = root['ligand']
    atom_types = lig_node_group['atom_types']
    atom_charges = lig_node_group['atom_charges']
    extra_feats = lig_node_group['extra_feats']


def error_and_update(error, block_idx, pbar, error_counter, output_dir):
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


def run_parallel(store_path: str, block_size: int, n_cpus: int, output_dir: Path, max_pending: int = None):
    if max_pending is None:
        max_pending = n_cpus * 5

    root = zarr.open(store_path, mode='r')
    n_atoms = root['ligand/atom_types'].shape[0]
    n_blocks = (n_atoms + block_size - 1) // block_size

    pbar = tqdm(total=n_blocks, desc=f"Counting {Path(store_path).name}", unit="blocks")
    aggregator = FeatureAggregator()
    error_counter = [0]

    with Pool(processes=n_cpus, initializer=worker_initializer, initargs=(store_path,), maxtasksperchild=5) as pool:
        pending = []

        for block_idx in range(n_blocks):
            while len(pending) >= max_pending:
                pending = [r for r in pending if not r.ready()]
                if len(pending) >= max_pending:
                    time.sleep(0.1)

            error_callback_fn = partial(
                error_and_update,
                block_idx=block_idx,
                error_counter=error_counter,
                output_dir=output_dir,
            )

            callback_fn = partial(
                save_and_update,
                feature_aggregator=aggregator,
                block_idx=block_idx,
                pbar=pbar,
                output_dir=output_dir,
            )

            block_start_idx = block_idx * block_size
            block_end_idx = min(block_start_idx + block_size, n_atoms)

            result = pool.apply_async(
                process_block,
                args=(block_start_idx, block_end_idx),
                callback=callback_fn,
                error_callback=error_callback_fn,
            )
            pending.append(result)

        for result in pending:
            result.wait()

    print(f"Processing completed with {error_counter[0]} errors.")
    return aggregator.unique_atom_features


def run_single(store_path: str, block_size: int, output_dir: Path):
    root = zarr.open(store_path, mode='r')
    n_atoms = root['ligand/atom_types'].shape[0]
    n_blocks = (n_atoms + block_size - 1) // block_size

    pbar = tqdm(total=n_blocks, desc=f"Counting {Path(store_path).name}", unit="blocks")
    aggregator = FeatureAggregator()
    error_counter = [0]

    worker_initializer(store_path)

    for block_idx in range(n_blocks):
        block_start_idx = block_idx * block_size
        callback_fn = partial(
            save_and_update,
            feature_aggregator=aggregator,
            block_idx=block_idx,
            pbar=pbar,
            output_dir=output_dir,
        )
        try:
            result = process_block(block_start_idx, min(block_start_idx + block_size, n_atoms))
            callback_fn(result)
        except Exception as e:
            print(f"Error processing block {block_idx}: {e}")
            error_counter[0] += 1

    pbar.close()
    print(f"Processing completed with {error_counter[0]} errors.")
    return aggregator.unique_atom_features


def count_store(store_path, block_size, n_cpus, output_dir):
    if n_cpus == 1:
        return run_single(store_path, block_size, output_dir)
    return run_parallel(store_path, block_size, n_cpus, output_dir)


def merge_counts(*count_dicts):
    merged = defaultdict(int)
    for counts in count_dicts:
        for feat, count in counts.items():
            merged[feat] += count
    return dict(merged)


def load_valid_tuples():
    from omtra.data.condensed_atom_typing import CondensedAtomTyper

    typer = CondensedAtomTyper()
    return set(typer.uncond_tuple_to_cond_idx.keys())


def report_unknown_tuples(tuple_counts):
    valid = load_valid_tuples()
    unknown = {t: c for t, c in tuple_counts.items() if t not in valid}
    print(f"Total unique tuples: {len(tuple_counts)}")
    print(f"Unknown tuples (not in Pharmit+Plinder vocab): {len(unknown)}")
    if unknown:
        for t, c in sorted(unknown.items(), key=lambda x: -x[1])[:20]:
            print(f"  {t}: count={c}")
    return unknown


if __name__ == '__main__':
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = args.splits or [args.split]
    versions = [None] if args.flat else (args.versions or ['exp', 'no_links', 'pred'])

    start_time = time.time()
    tuple_counts = {}

    for version in versions:
        version_key = 'pred' if args.flat else version
        merged_split_counts = defaultdict(int)

        for split in splits:
            store_path = store_path_for(args.plinder_path, split, version=version, flat=args.flat)
            print(f"\nCounting tuples in {store_path}")
            split_counts = count_store(store_path, args.block_size, args.n_cpus, args.output_dir)
            for feat, count in split_counts.items():
                merged_split_counts[feat] += count

        tuple_counts[version_key] = dict(merged_split_counts)

    output_name = args.output_name
    if output_name is None:
        if args.flat:
            output_name = f"pdbbind_{'_'.join(splits)}_condensed_atom_types"
        else:
            output_name = f"plinder_{args.split}_condensed_atom_types"

    output_path = args.output_dir / f"{output_name}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(tuple_counts, f)

    if args.check_vocab:
        if args.flat:
            combined = merge_counts(*tuple_counts.values())
            unknown = report_unknown_tuples(combined)
            unknown_path = args.output_dir / f"{output_name}_unknown.pkl"
            with open(unknown_path, 'wb') as f:
                pickle.dump(unknown, f)
            print(f"Wrote unknown tuple report to {unknown_path}")
        else:
            for version_key, counts in tuple_counts.items():
                print(f"\nVocab check for version '{version_key}':")
                report_unknown_tuples(counts)

    end_time = time.time()
    print(f"Wrote {output_path}")
    print(f"Total time: {end_time - start_time:.1f} seconds")
