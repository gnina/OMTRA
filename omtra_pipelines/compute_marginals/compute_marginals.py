"""
Pipeline for computing marginal distributions of discrete modalities from Pharmit zarr stores.

This is used for Stage 2 of the noisy paths experiment: data-marginal corruption.
Instead of corrupting tokens with uniform samples, we corrupt with samples from
the empirical data distribution, which better approximates realistic denoiser errors.

Usage:
    python omtra_pipelines/compute_marginals/compute_marginals.py \
        --pharmit_path data/pharmit \
        --split train \
        --output_path omtra/constants/pharmit_marginals.npz \
        --n_samples 1000000

The output .npz file contains:
    - lig_cond_a_marginal: (n_categories,) probability distribution over condensed atom types
    - lig_e_marginal: (4,) probability distribution over bond types [none, single, double, triple]
    - lig_cond_a_counts: raw counts
    - lig_e_counts: raw counts
    - n_atoms_total: total atoms counted
    - n_edges_total: total edges counted (including inferred "no bond" edges)
"""

import argparse
import numpy as np
import zarr
from pathlib import Path
from tqdm import tqdm
import pickle

from omtra.data.condensed_atom_typing import CondensedAtomTyper
from omtra.constants import num_condensed_atom_types


def compute_marginals(
    pharmit_path: str,
    split: str = "train",
    output_path: str = None,
    n_samples: int = None,
    chunk_size: int = 10000,
    sample_ranges: list = None,
    verbose: bool = True,
):
    """
    Compute marginal distributions over discrete ligand modalities.

    Args:
        pharmit_path: Path to pharmit data directory (contains {split}.zarr)
        split: Which split to use (train, val, test)
        output_path: Where to save the marginals (.npz file)
        n_samples: Number of molecules to sample (None = all)
        chunk_size: Number of molecules to process per batch
        sample_ranges: List of (start, end) tuples for sampling specific ranges
        verbose: Print progress

    Returns:
        dict with marginal distributions and counts
    """
    zarr_path = Path(pharmit_path) / f"{split}.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
    root = zarr.open(store=store, mode='r')

    # Get dataset size
    n_graphs_total = root['lig/node/graph_lookup'].shape[0]

    # Build list of ranges to process
    if sample_ranges is not None:
        ranges_to_process = [(s, min(e, n_graphs_total)) for s, e in sample_ranges]
        n_graphs = sum(e - s for s, e in ranges_to_process)
    elif n_samples is not None:
        n_graphs = min(n_graphs_total, n_samples)
        ranges_to_process = [(0, n_graphs)]
    else:
        n_graphs = n_graphs_total
        ranges_to_process = [(0, n_graphs)]

    if verbose:
        print(f"Computing marginals from {zarr_path}")
        print(f"Processing {n_graphs:,} molecules from {len(ranges_to_process)} ranges")

    # Initialize condensed atom typer
    cond_a_typer = CondensedAtomTyper(fake_atoms=False)
    n_cond_a_types = cond_a_typer.n_real_categories

    # Initialize counts
    # For lig_cond_a: count each condensed atom type
    lig_cond_a_counts = np.zeros(n_cond_a_types, dtype=np.int64)

    # For lig_e: count each bond type (0=none, 1=single, 2=double, 3=triple)
    # Note: zarr stores edges sparsely (only non-zero bonds)
    # We need to infer the number of "no bond" edges from graph structure
    lig_e_counts = np.zeros(4, dtype=np.int64)

    n_atoms_total = 0
    n_edges_stored = 0  # edges in zarr (non-zero bonds only)
    n_edges_possible = 0  # total possible edges (for computing "no bond" count)

    # Process in chunks
    node_lookup = root['lig/node/graph_lookup']
    edge_lookup = root['lig/edge/graph_lookup']
    node_a = root['lig/node/a']
    node_c = root['lig/node/c']
    node_extra = root['lig/node/extra_feats']
    edge_e = root['lig/edge/e']

    # Build iterator over all chunk positions across ranges
    chunk_positions = []
    for range_start, range_end in ranges_to_process:
        for pos in range(range_start, range_end, chunk_size):
            chunk_positions.append((pos, min(pos + chunk_size, range_end)))

    pbar = tqdm(chunk_positions, disable=not verbose, desc="Processing molecules")

    for chunk_start, chunk_end in pbar:
        # Get node slice indices for this chunk
        node_slices = node_lookup[chunk_start:chunk_end]

        # Get all nodes for this chunk
        node_start = int(node_slices[0, 0])
        node_end = int(node_slices[-1, 1])

        chunk_a = node_a[node_start:node_end]
        chunk_c = node_c[node_start:node_end]
        chunk_extra = node_extra[node_start:node_end][:, :-1]  # exclude last col

        # Convert to condensed atom types
        chunk_cond_a = cond_a_typer.feats_to_cond_a(chunk_a, chunk_c, chunk_extra)

        # Count atom types
        for atom_type in chunk_cond_a:
            lig_cond_a_counts[atom_type] += 1
        n_atoms_total += len(chunk_cond_a)

        # Get edge slice indices for this chunk
        edge_slices = edge_lookup[chunk_start:chunk_end]

        # Get all edges for this chunk
        if edge_slices[0, 0] < edge_slices[-1, 1]:
            edge_start = int(edge_slices[0, 0])
            edge_end = int(edge_slices[-1, 1])
            chunk_e = edge_e[edge_start:edge_end]

            # Count bond types (stored edges are all non-zero: 1, 2, or 3)
            for bond_type in [1, 2, 3]:
                lig_e_counts[bond_type] += np.sum(chunk_e == bond_type)
            n_edges_stored += len(chunk_e)

        # Compute number of possible edges (for "no bond" inference)
        # For each molecule, possible undirected edges = n_atoms * (n_atoms - 1) / 2
        for i in range(chunk_end - chunk_start):
            mol_node_start = int(node_slices[i, 0]) - node_start
            mol_node_end = int(node_slices[i, 1]) - node_start
            n_atoms_mol = mol_node_end - mol_node_start
            n_edges_possible += n_atoms_mol * (n_atoms_mol - 1) // 2

    # Infer "no bond" count: possible edges minus stored edges
    # Note: stored edges are directed in the dense representation but sparse stores undirected
    # Actually, looking at the code, edge_index stores directed edges, but e stores bond orders
    # Need to check if edges are stored as undirected or directed
    # From xace_ligand.py, it seems like edges are stored as undirected (sparse representation)
    # and then expanded to directed in sparse_to_dense

    # The stored edges are undirected, so:
    # no_bond_count = possible_edges - stored_edges
    n_no_bond = n_edges_possible - n_edges_stored
    lig_e_counts[0] = n_no_bond

    n_edges_total = n_edges_possible

    if verbose:
        print(f"\nTotal atoms: {n_atoms_total:,}")
        print(f"Total edges (including no-bond): {n_edges_total:,}")
        print(f"Stored edges (non-zero bonds): {n_edges_stored:,}")
        print(f"Inferred no-bond edges: {n_no_bond:,}")

    # Compute marginal distributions
    lig_cond_a_marginal = lig_cond_a_counts / lig_cond_a_counts.sum()
    lig_e_marginal = lig_e_counts / lig_e_counts.sum()

    if verbose:
        print(f"\nlig_cond_a distribution: {n_cond_a_types} types")
        print(f"  Top 5 types: {np.argsort(lig_cond_a_marginal)[-5:][::-1]}")
        print(f"  Top 5 probs: {np.sort(lig_cond_a_marginal)[-5:][::-1]}")

        print(f"\nlig_e distribution:")
        for i, name in enumerate(['none', 'single', 'double', 'triple']):
            print(f"  {name}: {lig_e_marginal[i]:.4f} ({lig_e_counts[i]:,})")

    result = {
        'lig_cond_a_marginal': lig_cond_a_marginal,
        'lig_e_marginal': lig_e_marginal,
        'lig_cond_a_counts': lig_cond_a_counts,
        'lig_e_counts': lig_e_counts,
        'n_atoms_total': n_atoms_total,
        'n_edges_total': n_edges_total,
        'n_cond_a_types': n_cond_a_types,
    }

    # Save if output path specified
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **result)
        if verbose:
            print(f"\nSaved marginals to {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute marginal distributions from Pharmit zarr store"
    )
    parser.add_argument(
        "--pharmit_path",
        type=str,
        required=True,
        help="Path to pharmit data directory (contains {split}.zarr)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Which split to use (default: train)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save marginals (default: {pharmit_path}/{split}_marginals.npz)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of molecules to sample (default: all)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10000,
        help="Batch size for processing (default: 10000)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = str(Path(args.pharmit_path) / f"{args.split}_marginals.npz")

    compute_marginals(
        pharmit_path=args.pharmit_path,
        split=args.split,
        output_path=args.output_path,
        n_samples=args.n_samples,
        chunk_size=args.chunk_size,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
