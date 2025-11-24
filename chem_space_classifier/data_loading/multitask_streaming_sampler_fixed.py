"""
Fixed Multitask Streaming Sampler - Correctly Counts Complete Graph Edges!

Key fix: Counts N*(N-1) edges per molecule (complete directed graph)
instead of just covalent bonds, matching what the GVP model actually uses.

This prevents the 30x underestimation that was causing OOM errors.
"""

import numpy as np
import torch
from torch.utils.data import Sampler
from typing import Optional, List
from tqdm import tqdm
import time


class MultitaskStreamingSamplerFixed(Sampler):
    """
    Streaming sampler that creates batches on-demand.

    Key difference from original MultitaskStreamingSampler:
    - Correctly counts COMPLETE GRAPH edges (N*(N-1))
    - Prevents OOM by accurate memory estimation
    """

    def __init__(
        self,
        dataset,
        edges_per_batch: int = 100_000,
        max_batches: int = 10_000,
        shuffle: bool = True,
        min_capacity_utilization: float = 0.8,
        use_cache: bool = True,
        cache_dir: str = './multitask_cache',
    ):
        """
        Args:
            dataset: PharmitDataset (not wrapper!)
            edges_per_batch: Target number of COMPLETE GRAPH edges per batch
            max_batches: Maximum batches per epoch
            shuffle: Whether to shuffle chunk order and samples
            min_capacity_utilization: Minimum fraction of edges_per_batch to use
            use_cache: Whether to cache indices
            cache_dir: Directory for cache files
        """
        self.dataset = dataset
        self.edges_per_batch = edges_per_batch
        self.max_batches = max_batches
        self.shuffle = shuffle
        self.min_capacity_utilization = min_capacity_utilization

        # Database configuration
        self.database_names = ['CSC', 'MCULE', 'PubChem', 'ZINC', 'MolPort']
        self.n_databases = 5

        # Dataset info
        self.graphs_per_chunk = dataset.graphs_per_chunk
        self.n_chunks = dataset.n_zarr_chunks
        self.total_graphs = len(dataset)

        # Use db/db chunking (molecule labels)
        zarr_root = dataset.root
        db_chunk_size = zarr_root['db/db'].chunks[0]
        self.graphs_per_chunk = db_chunk_size
        self.n_chunks = (self.total_graphs + db_chunk_size - 1) // db_chunk_size

        print(f"{'='*80}")
        print("MultitaskStreamingSamplerFixed")
        print(f"{'='*80}")
        print(f"  Total graphs: {self.total_graphs:,}")
        print(f"  Zarr chunks: {self.n_chunks}")
        print(f"  Target edges/batch: {edges_per_batch:,} (COMPLETE GRAPH edges)")
        print(f"  Max batches/epoch: {max_batches:,}")
        print(f"  Min capacity: {min_capacity_utilization:.0%}")
        print(f"  Shuffle: {shuffle}")
        print()
        print(f"  ⚠️  KEY FIX: Counting complete graph edges N*(N-1)")
        print(f"     NOT just covalent bonds!")
        print()

        # Load or build indices (ONLY indices, NOT batches!)
        self.use_cache = use_cache
        if use_cache:
            from pathlib import Path
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_or_build_indices()
        else:
            self._build_indices()

        print(f"{'='*80}\n")

    def _get_cache_path(self):
        """Get cache file path for indices."""
        import hashlib

        try:
            if hasattr(self.dataset.root, 'store'):
                if hasattr(self.dataset.root.store, 'path'):
                    dataset_path = str(self.dataset.root.store.path)
                else:
                    dataset_path = str(self.dataset.root.store)
            else:
                dataset_path = 'unknown'
        except:
            dataset_path = 'unknown'

        config_str = f"{dataset_path}_multitask_streaming_fixed"
        cache_key = hashlib.md5(config_str.encode()).hexdigest()
        return self.cache_dir / f"multitask_streaming_fixed_{cache_key}.npz"

    def _load_or_build_indices(self):
        """Load indices from cache or build."""
        cache_path = self._get_cache_path()

        if cache_path.exists():
            print(f"📂 Loading indices from cache...")
            data = np.load(cache_path)
            self.valid_indices = data['valid_indices']
            self.edge_counts = data['edge_counts']
            self.database_labels = data['database_labels']

            print(f"  ✓ Loaded {len(self.valid_indices):,} samples")
            self._print_statistics()
        else:
            print(f"  No cache found. Building indices...")
            self._build_indices()

            # Save to cache
            print(f"  💾 Saving to cache...")
            np.savez_compressed(
                cache_path,
                valid_indices=self.valid_indices,
                edge_counts=self.edge_counts,
                database_labels=self.database_labels
            )
            size_mb = cache_path.stat().st_size / (1024**2)
            print(f"  ✓ Cached! Size: {size_mb:.1f} MB")

    def _build_indices(self):
        """Build indices by scanning through dataset."""
        print(f"  Building multitask data...")

        zarr_root = self.dataset.root
        db_data = zarr_root['db/db']  # [N, 5] - all 5 database labels
        node_graph_lookup = zarr_root['lig/node/graph_lookup']  # ← USE THIS for node counts!

        valid_indices = []
        edge_counts = []
        database_labels = []

        # Patterns to skip (problematic patterns with bad coordinates)
        SKIP_PATTERNS = {(0, 0, 0, 0, 1), (1, 1, 0, 0, 1)}
        n_skipped = 0

        # Process in chunks for memory efficiency
        chunk_batch_size = 50
        n_chunk_batches = (self.n_chunks + chunk_batch_size - 1) // chunk_batch_size

        start_time = time.time()

        for chunk_batch_idx in tqdm(range(n_chunk_batches), desc="  Processing"):
            chunk_start = chunk_batch_idx * chunk_batch_size
            chunk_end = min((chunk_batch_idx + 1) * chunk_batch_size, self.n_chunks)

            sample_start = chunk_start * self.graphs_per_chunk
            sample_end = min(chunk_end * self.graphs_per_chunk, self.total_graphs)

            # Read chunk
            chunk_db_labels = db_data[sample_start:sample_end]
            chunk_node_lookup = node_graph_lookup[sample_start:sample_end]

            # ================================================================
            # KEY FIX: Calculate COMPLETE GRAPH edges, not just covalent bonds!
            # ================================================================
            # For a complete directed graph: N * (N-1) edges
            chunk_node_counts = chunk_node_lookup[:, 1] - chunk_node_lookup[:, 0]
            chunk_edge_counts = chunk_node_counts * (chunk_node_counts - 1)
            # ================================================================

            # Filter out problematic patterns
            for i, labels in enumerate(chunk_db_labels):
                pattern = tuple(labels.astype(int))

                # Skip problematic patterns
                if pattern in SKIP_PATTERNS:
                    n_skipped += 1
                    continue

                # Add valid sample
                global_idx = sample_start + i
                valid_indices.append(global_idx)
                edge_counts.append(int(chunk_edge_counts[i]))
                database_labels.append(labels.tolist())

        # Convert to arrays
        self.valid_indices = np.array(valid_indices, dtype=np.int64)
        self.edge_counts = np.array(edge_counts, dtype=np.int32)
        self.database_labels = np.array(database_labels, dtype=np.int8)  # [N, 5]

        elapsed = time.time() - start_time

        print(f"\n  ✓ Data built in {elapsed:.1f}s")
        print(f"    Total samples: {len(self.valid_indices):,}")
        if n_skipped > 0:
            print(f"    Skipped {n_skipped:,} molecules with problematic patterns")
            print(f"    Patterns skipped: {SKIP_PATTERNS}")

        # Calculate average atoms and edges
        avg_atoms = np.sqrt(self.edge_counts.mean() + 0.25) + 0.5  # Inverse of N*(N-1)
        print(f"    Avg atoms/molecule: {avg_atoms:.1f}")
        print(f"    Avg complete graph edges/molecule: {self.edge_counts.mean():.1f}")
        print(f"    Edge range: {self.edge_counts.min():,} - {self.edge_counts.max():,}")

        self._print_statistics()

    def _print_statistics(self):
        """Print dataset statistics."""
        print(f"\n  Label distribution:")
        for db_idx, db_name in enumerate(self.database_names):
            n_pos = (self.database_labels[:, db_idx] == 1).sum()
            pct = 100 * n_pos / len(self.database_labels)
            print(f"    {db_name:>10}: {n_pos:>10,} positive ({pct:>5.2f}%)")

    def __iter__(self):
        """
        Create batches on-the-fly during iteration.

        This is the key innovation: we don't pre-build batches,
        we create them lazily as we iterate!
        """
        # Group indices by chunk (fast - we only store positions, not data)
        chunk_to_positions = {}
        for idx_pos, idx in enumerate(self.valid_indices):
            chunk_id = idx // self.graphs_per_chunk
            if chunk_id not in chunk_to_positions:
                chunk_to_positions[chunk_id] = []
            chunk_to_positions[chunk_id].append(idx_pos)

        # Shuffle chunk order
        chunk_ids = list(chunk_to_positions.keys())
        if self.shuffle:
            np.random.shuffle(chunk_ids)

        # Iterate through chunks and create batches on-the-fly
        batch_count = 0

        for chunk_id in chunk_ids:
            if batch_count >= self.max_batches:
                break

            # Get data for this chunk
            chunk_positions = chunk_to_positions[chunk_id]
            chunk_indices = self.valid_indices[chunk_positions]
            chunk_edges = self.edge_counts[chunk_positions]

            # Shuffle within chunk
            if self.shuffle:
                perm = np.random.permutation(len(chunk_indices))
                chunk_indices = chunk_indices[perm]
                chunk_edges = chunk_edges[perm]

            # Create batches from this chunk using adaptive loading
            offset = 0
            while offset < len(chunk_indices) and batch_count < self.max_batches:
                # Adaptive batch loader (inline for speed)
                remaining_edges = chunk_edges[offset:]
                cumsum = np.cumsum(remaining_edges)
                mask = cumsum <= self.edges_per_batch
                valid_positions = np.where(mask)[0]

                if len(valid_positions) == 0:
                    # First graph too large, take it anyway
                    batch = chunk_indices[offset:offset+1].tolist()
                    offset += 1
                else:
                    # Check capacity utilization
                    n_selected = len(valid_positions)
                    capacity = cumsum[n_selected-1] / self.edges_per_batch

                    if capacity < self.min_capacity_utilization:
                        # Try adding one more graph
                        if n_selected < len(cumsum):
                            if cumsum[n_selected] <= self.edges_per_batch * 1.1:
                                valid_positions = np.arange(n_selected + 1)

                    batch = chunk_indices[offset:offset+len(valid_positions)].tolist()
                    offset += len(valid_positions)

                batch_count += 1
                yield batch

                if batch_count >= self.max_batches:
                    break

    def __len__(self):
        """Return max number of batches per epoch."""
        return self.max_batches


# ============================================================================
# DataLoader Creation
# ============================================================================

def create_multitask_streaming_dataloader_fixed(
    dataset,
    edges_per_batch: int = 100_000,
    max_batches: int = 10_000,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
):
    """
    Create streaming DataLoader for multitask classification.

    FIXED VERSION: Correctly counts complete graph edges!

    Args:
        dataset: PharmitWrapper or PharmitDataset
        edges_per_batch: Target COMPLETE GRAPH edges per batch (default: 100k)
        max_batches: Maximum batches per epoch
        shuffle: Whether to shuffle
        num_workers: Number of workers
        **kwargs: Additional args for sampler

    Returns:
        DataLoader with multitask collate function
    """
    from torch.utils.data import DataLoader
    from omtra.dataset.data_module import omtra_collate_fn

    # Get underlying dataset
    if hasattr(dataset, 'ds'):
        underlying_dataset = dataset.ds
        wrapper = dataset
    else:
        underlying_dataset = dataset
        from omtra.dataset.pharmit import PharmitWrapper
        wrapper = PharmitWrapper(dataset)

    # Create streaming sampler (no batch pre-building!)
    sampler = MultitaskStreamingSamplerFixed(
        dataset=underlying_dataset,
        edges_per_batch=edges_per_batch,
        max_batches=max_batches,
        shuffle=shuffle,
        **kwargs
    )

    # Collate function
    def multitask_collate_fn(batch):
        """Collate function - labels are already [B, 5]."""
        collated = omtra_collate_fn(batch)
        # Labels are already at indices [0,1,2,3,4] for the 5 databases
        # Shape is already [B, 5]
        return collated

    # Create DataLoader
    return DataLoader(
        wrapper,
        batch_sampler=sampler,
        collate_fn=multitask_collate_fn,
        num_workers=num_workers,
        pin_memory=False,  # Disabled to avoid RAM exhaustion with large datasets
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )


# ============================================================================
# Comparison Helper
# ============================================================================

def compare_edge_counting(dataset, n_samples=100):
    """
    Compare old vs new edge counting to show the difference.

    Args:
        dataset: PharmitDataset
        n_samples: Number of molecules to sample
    """
    import random

    zarr_root = dataset.root
    node_lookup = zarr_root['lig/node/graph_lookup']
    edge_lookup = zarr_root['lig/edge/graph_lookup']

    print("="*80)
    print("EDGE COUNTING COMPARISON")
    print("="*80)

    # Sample random molecules
    indices = random.sample(range(len(dataset)), n_samples)

    covalent_bonds = []
    complete_edges = []

    for idx in indices:
        # Covalent bonds (old method)
        edge_start, edge_end = edge_lookup[idx]
        n_bonds = edge_end - edge_start
        covalent_bonds.append(n_bonds)

        # Complete graph edges (new method)
        node_start, node_end = node_lookup[idx]
        n_atoms = node_end - node_start
        n_complete_edges = n_atoms * (n_atoms - 1)
        complete_edges.append(n_complete_edges)

    covalent_bonds = np.array(covalent_bonds)
    complete_edges = np.array(complete_edges)

    ratio = complete_edges / (covalent_bonds + 1e-10)

    print(f"\nAnalyzed {n_samples} random molecules:")
    print(f"\nOLD method (covalent bonds):")
    print(f"  Mean: {covalent_bonds.mean():.1f} edges/molecule")
    print(f"  Range: {covalent_bonds.min()} - {covalent_bonds.max()}")

    print(f"\nNEW method (complete graph):")
    print(f"  Mean: {complete_edges.mean():.1f} edges/molecule")
    print(f"  Range: {complete_edges.min():,} - {complete_edges.max():,}")

    print(f"\nRatio (complete/covalent):")
    print(f"  Mean: {ratio.mean():.1f}x")
    print(f"  Range: {ratio.min():.1f}x - {ratio.max():.1f}x")

    print(f"\n💡 INSIGHT:")
    print(f"   If you set edges_per_batch=50,000 with OLD method,")
    print(f"   you actually get ~{50000 * ratio.mean():,.0f} edges!")
    print(f"   That's why you got OOM!")

    print("="*80)
