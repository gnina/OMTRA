"""
Edge-Based Multitask Sampler for 5-Database Classification

Based on mcule_edge_sampler.py but adapted for multitask learning.

Key features:
1. Batches by EDGES (not molecules) for consistent memory usage
2. Chunk-aware for efficient I/O
3. Maintains approximate target proportions across all 5 databases

Dataset structure:
- Labels at indices [0, 1, 2, 3, 4] = [CSC, MCULE, PubChem, ZINC, MolPort]
- No NA samples (already filtered)
- Multi-label: molecules can be positive for multiple databases
"""

import numpy as np
import torch
from torch.utils.data import Sampler
from tqdm import tqdm
import time
from typing import List, Optional


class MultitaskEdgeSampler(Sampler):
    """
    Edge-based sampler for 5-database multitask learning.

    Similar to EdgeBasedMCuleSampler but handles all 5 databases.
    """

    def __init__(
        self,
        dataset,
        edges_per_batch: int = 300000,
        shuffle: bool = True,
        target_proportions: Optional[List[float]] = None,
        min_capacity_utilization: float = 0.8,
        use_cache: bool = True,
        cache_dir: str = './multitask_cache',
    ):
        """
        Args:
            dataset: PharmitDataset (not wrapper!)
            edges_per_batch: Target number of edges per batch (default: 300k)
            shuffle: Whether to shuffle chunk order and samples
            target_proportions: Target % positive for [CSC, MCULE, PubChem, ZINC, MolPort]
            min_capacity_utilization: Minimum fraction of edges_per_batch to use
            use_cache: Whether to cache indices and edge counts
            cache_dir: Directory for cache files
        """
        self.dataset = dataset
        self.edges_per_batch = edges_per_batch
        self.shuffle = shuffle
        self.min_capacity_utilization = min_capacity_utilization

        # Database configuration
        self.database_names = ['CSC', 'MCULE', 'PubChem', 'ZINC', 'MolPort']
        self.n_databases = 5

        # Target proportions (from dataset analysis)
        if target_proportions is None:
            target_proportions = [0.1991, 0.2995, 0.5022, 0.2087, 0.0407]
        self.target_proportions = np.array(target_proportions)

        # Dataset info
        self.total_graphs = len(dataset)

        # Use db/db chunking (molecule labels) instead of lig/node/x chunking (atoms)
        # This ensures our batches align with how molecules are stored, not atoms
        zarr_root = dataset.root
        db_chunk_size = zarr_root['db/db'].chunks[0]
        self.graphs_per_chunk = db_chunk_size
        self.n_chunks = (self.total_graphs + db_chunk_size - 1) // db_chunk_size

        print(f"{'='*80}")
        print("MultitaskEdgeSampler")
        print(f"{'='*80}")
        print(f"  Total graphs: {self.total_graphs:,}")
        print(f"  Zarr chunks: {self.n_chunks}")
        print(f"  Target edges/batch: {edges_per_batch:,}")
        print(f"  Min capacity: {min_capacity_utilization:.0%}")
        print(f"  Shuffle: {shuffle}")
        print()
        print("Target proportions (% positive per database):")
        for name, prop in zip(self.database_names, target_proportions):
            print(f"  {name:>10}: {prop*100:>5.2f}%")
        print()

        # Cache management
        self.use_cache = use_cache
        if use_cache:
            from pathlib import Path
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_or_build_data()
        else:
            self._build_data()

        # Create batches
        self._create_batches()

    def _get_cache_path(self):
        """Get cache file path."""
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

        config_str = f"{dataset_path}_multitask_edges"
        cache_key = hashlib.md5(config_str.encode()).hexdigest()
        return self.cache_dir / f"multitask_edges_{cache_key}.npz"

    def _load_or_build_data(self):
        """Load from cache or build."""
        cache_path = self._get_cache_path()

        if cache_path.exists():
            print(f"📂 Loading from cache...")
            data = np.load(cache_path)
            self.valid_indices = data['valid_indices']
            self.edge_counts = data['edge_counts']
            self.database_labels = data['database_labels']

            print(f"  ✓ Loaded {len(self.valid_indices):,} samples")
            self._print_statistics()
        else:
            print(f"  No cache found. Building data...")
            self._build_data()

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

    def _build_data(self):
        """Build indices, edge counts, and labels."""
        print(f"  Building multitask data...")

        zarr_root = self.dataset.root
        db_data = zarr_root['db/db']  # [N, 5] - all 5 database labels
        edge_graph_lookup = zarr_root['lig/edge/graph_lookup']

        valid_indices = []
        edge_counts = []
        database_labels = []

        # Patterns to skip (problematic patterns with bad coordinates)
        SKIP_PATTERNS = {(0, 0, 0, 0, 1), (1, 1, 0, 0, 1)}
        n_skipped = 0

        # Process in chunks
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
            chunk_edge_lookup = edge_graph_lookup[sample_start:sample_end]

            # Calculate edge counts
            chunk_edge_counts = chunk_edge_lookup[:, 1] - chunk_edge_lookup[:, 0]

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
        print(f"    Avg edges/molecule: {self.edge_counts.mean():.1f}")

        self._print_statistics()

    def _print_statistics(self):
        """Print dataset statistics."""
        print(f"\n  Label distribution:")
        for db_idx, db_name in enumerate(self.database_names):
            n_pos = (self.database_labels[:, db_idx] == 1).sum()
            pct = 100 * n_pos / len(self.database_labels)
            print(f"    {db_name:>10}: {n_pos:>10,} positive ({pct:>5.2f}%)")

    def _create_batches(self):
        """
        Create edge-based batches with per-database stratification.

        Strategy:
        1. Group molecules by chunk
        2. Shuffle chunk order (if shuffle=True)
        3. Within each chunk, use GREEDY STRATIFIED SAMPLING:
           - For each batch, iteratively select molecules that bring
             current proportions closer to target proportions
           - Respect edge budget constraint
        """
        print(f"\n  Creating edge-based batches...")
        print(f"    🔧 Using adaptive edge-based batching (fast!)")

        # Group by chunk
        chunk_to_data = {}
        for idx, edge_count, labels in zip(self.valid_indices, self.edge_counts, self.database_labels):
            chunk_id = idx // self.graphs_per_chunk

            if chunk_id not in chunk_to_data:
                chunk_to_data[chunk_id] = {
                    'indices': [],
                    'edges': [],
                    'labels': []
                }

            chunk_to_data[chunk_id]['indices'].append(idx)
            chunk_to_data[chunk_id]['edges'].append(edge_count)
            chunk_to_data[chunk_id]['labels'].append(labels)

        # Convert to numpy
        for chunk_id in chunk_to_data:
            chunk_to_data[chunk_id]['indices'] = np.array(chunk_to_data[chunk_id]['indices'])
            chunk_to_data[chunk_id]['edges'] = np.array(chunk_to_data[chunk_id]['edges'])
            chunk_to_data[chunk_id]['labels'] = np.array(chunk_to_data[chunk_id]['labels'])

        print(f"       Data spans {len(chunk_to_data)} chunks")

        # Get chunk IDs
        chunk_ids = sorted(chunk_to_data.keys())

        # Shuffle chunk order
        if self.shuffle:
            np.random.shuffle(chunk_ids)
            print(f"       Shuffled chunk order")

        # Create stratified batches
        self.batch_indices = []
        batch_stats = []

        for chunk_id in tqdm(chunk_ids, desc="       Creating batches"):
            chunk_indices = chunk_to_data[chunk_id]['indices']
            chunk_edges = chunk_to_data[chunk_id]['edges']
            chunk_labels = chunk_to_data[chunk_id]['labels']

            # Shuffle within chunk
            if self.shuffle:
                perm = np.random.permutation(len(chunk_indices))
                chunk_indices = chunk_indices[perm]
                chunk_edges = chunk_edges[perm]
                chunk_labels = chunk_labels[perm]

            # Create batches from this chunk using adaptive loader (much faster!)
            offset = 0
            while offset < len(chunk_indices):
                # Use adaptive batch loader (fast, simple edge-based batching)
                batch_indices = self._adaptive_batch_loader(
                    chunk_indices[offset:],
                    chunk_edges[offset:]
                )

                if len(batch_indices) == 0:
                    break

                self.batch_indices.append(batch_indices.tolist())

                # Compute batch statistics
                batch_mask = np.isin(chunk_indices, batch_indices)
                batch_labels = chunk_labels[batch_mask]
                batch_edges_total = chunk_edges[batch_mask].sum()
                batch_proportions = batch_labels.mean(axis=0)

                batch_stats.append({
                    'n_molecules': len(batch_indices),
                    'n_edges': batch_edges_total,
                    'proportions': batch_proportions
                })

                offset += len(batch_indices)

        # Print statistics
        avg_molecules = np.mean([b['n_molecules'] for b in batch_stats])
        avg_edges = np.mean([b['n_edges'] for b in batch_stats])

        print(f"\n  ✓ Created {len(self.batch_indices)} stratified edge-based batches")
        print(f"    Avg molecules/batch: {avg_molecules:.1f}")
        print(f"    Avg edges/batch: {avg_edges:,.0f} (target: {self.edges_per_batch:,})")

        # Proportion statistics
        print(f"\n    Batch proportion statistics:")
        print(f"    {'Database':<12} {'Target':>8} {'Mean':>8} {'Std':>8}")
        print(f"    {'-'*40}")

        for db_idx, db_name in enumerate(self.database_names):
            target = self.target_proportions[db_idx]
            props = [b['proportions'][db_idx] for b in batch_stats]
            mean_prop = np.mean(props)
            std_prop = np.std(props)

            print(f"    {db_name:<12} {target*100:>7.2f}% {mean_prop*100:>7.2f}% {std_prop*100:>7.2f}%")

        print(f"{'='*80}\n")

    def _greedy_stratified_batch(self, indices, edges, labels):
        """
        Greedily create a stratified batch that maintains target proportions.

        For each molecule, compute a "benefit score" based on how much it
        improves the current proportions toward targets.

        Args:
            indices: Available molecule indices
            edges: Edge counts for each molecule
            labels: Database labels for each molecule [N, 5]

        Returns:
            batch_indices: Selected molecule indices
            total_edges: Total edges in batch
        """
        n_available = len(indices)
        if n_available == 0:
            return np.array([], dtype=np.int64), 0

        # Track selected molecules
        selected = []
        selected_edges = []
        selected_labels = []
        total_edges = 0

        # Available pool (will shrink as we select)
        available_mask = np.ones(n_available, dtype=bool)

        # Greedy selection
        while available_mask.any():
            # Current proportions in batch
            if len(selected_labels) == 0:
                current_props = np.zeros(self.n_databases)
                n_selected = 0
            else:
                current_props = np.array(selected_labels).mean(axis=0)
                n_selected = len(selected_labels)

            # Find best molecule to add
            best_idx = None
            best_score = -np.inf

            for i in np.where(available_mask)[0]:
                # Check edge budget
                if total_edges + edges[i] > self.edges_per_batch * 1.1:  # Allow 10% overflow
                    continue

                # Simulate adding this molecule
                simulated_labels = selected_labels + [labels[i]]
                simulated_props = np.array(simulated_labels).mean(axis=0)

                # Score: negative distance to target (closer = better)
                # Use weighted distance (weight by target to emphasize rare databases)
                distance = np.abs(simulated_props - self.target_proportions)
                weighted_distance = (distance * (1 + self.target_proportions)).sum()
                score = -weighted_distance

                if score > best_score:
                    best_score = score
                    best_idx = i

            # If no molecule can be added (edge budget exceeded), stop
            if best_idx is None:
                break

            # Add best molecule to batch
            selected.append(indices[best_idx])
            selected_edges.append(edges[best_idx])
            selected_labels.append(labels[best_idx])
            total_edges += edges[best_idx]
            available_mask[best_idx] = False

            # Stop if we've reached minimum capacity and proportions are good
            if total_edges >= self.edges_per_batch * self.min_capacity_utilization:
                # Check if proportions are reasonable
                current_props = np.array(selected_labels).mean(axis=0)
                prop_errors = np.abs(current_props - self.target_proportions)

                # If all databases within 10% of target, we can stop
                if (prop_errors < 0.10).all():
                    break

        return np.array(selected, dtype=np.int64), total_edges

    def _adaptive_batch_loader(self, indices: np.ndarray, edge_counts: np.ndarray) -> np.ndarray:
        """
        Adaptive batch loader (from mcule_edge_sampler.py).

        Selects molecules whose cumulative edge count stays ≤ edges_per_batch.
        """
        n_graphs = len(indices)

        if n_graphs == 0:
            return np.array([], dtype=np.int64)

        # Shuffle if needed (already shuffled at chunk level, skip here)
        # Cumulative sum
        cumsum = np.cumsum(edge_counts)

        # Find how many fit
        mask = cumsum <= self.edges_per_batch
        valid_positions = np.where(mask)[0]

        # Handle edge case: first graph too large
        if len(valid_positions) == 0:
            return indices[:1]

        # Check capacity utilization
        n_edges_selected = cumsum[valid_positions[-1]]
        capacity_utilization = n_edges_selected / self.edges_per_batch

        # If utilization too low, try to add one more
        if capacity_utilization < self.min_capacity_utilization:
            next_idx = len(valid_positions)
            if next_idx < len(edge_counts):
                # Allow 10% overflow
                if cumsum[next_idx] <= self.edges_per_batch * 1.1:
                    valid_positions = np.arange(next_idx + 1)

        return indices[valid_positions]

    def __iter__(self):
        """Iterate through batches."""
        for batch in self.batch_indices:
            yield batch

    def __len__(self):
        """Return number of batches."""
        return len(self.batch_indices)


# ============================================================================
# DataLoader Creation
# ============================================================================

def create_multitask_dataloader(
    dataset,
    edges_per_batch: int = 300000,
    shuffle: bool = True,
    target_proportions: Optional[List[float]] = None,
    num_workers: int = 8,
    **kwargs
):
    """
    Create DataLoader for multitask classification with edge-based batching.

    Args:
        dataset: PharmitWrapper or PharmitDataset
        edges_per_batch: Target edges per batch (default: 300k)
        shuffle: Whether to shuffle
        target_proportions: Target proportions for [CSC, MCULE, PubChem, ZINC, MolPort]
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

    # Create sampler
    sampler = MultitaskEdgeSampler(
        dataset=underlying_dataset,
        edges_per_batch=edges_per_batch,
        shuffle=shuffle,
        target_proportions=target_proportions,
        **kwargs
    )

    # Collate function
    def multitask_collate_fn(batch):
        """Collate function - labels are already [B, 5]."""
        collated = omtra_collate_fn(batch)

        # Labels are already at indices [0,1,2,3,4] for the 5 databases
        # Shape is already [B, 5]
        # No need to extract - just use as is!

        return collated

    # Create DataLoader
    return DataLoader(
        wrapper,
        batch_sampler=sampler,
        collate_fn=multitask_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )


# ============================================================================
# Diagnostic Function
# ============================================================================

def diagnose_sampler(sampler: MultitaskEdgeSampler, n_batches: int = 20):
    """
    Diagnose sampler to verify proportions.

    Args:
        sampler: The sampler to diagnose
        n_batches: Number of batches to check
    """
    print("="*80)
    print("SAMPLER DIAGNOSTICS")
    print("="*80)

    batch_stats = []

    for batch_idx, batch_indices in enumerate(sampler.batch_indices[:n_batches]):
        # Get labels for this batch
        batch_mask = np.isin(sampler.valid_indices, batch_indices)
        batch_labels = sampler.database_labels[batch_mask]

        # Compute proportions
        proportions = batch_labels.mean(axis=0)

        batch_stats.append(proportions)

        # Print batch info
        print(f"\nBatch {batch_idx} (size: {len(batch_indices)})")
        for db_idx, db_name in enumerate(sampler.database_names):
            target = sampler.target_proportions[db_idx]
            actual = proportions[db_idx]
            diff = actual - target
            print(f"  {db_name:>10}: {actual*100:>5.2f}% (target: {target*100:>5.2f}%, "
                  f"diff: {diff*100:>+5.2f}%)")

    # Summary
    batch_stats = np.array(batch_stats)
    print(f"\n{'='*80}")
    print(f"SUMMARY (first {n_batches} batches)")
    print(f"{'='*80}")

    for db_idx, db_name in enumerate(sampler.database_names):
        mean_prop = batch_stats[:, db_idx].mean()
        std_prop = batch_stats[:, db_idx].std()
        target = sampler.target_proportions[db_idx]

        print(f"{db_name:>10}: Mean={mean_prop*100:>5.2f}% ± {std_prop*100:>4.2f}% "
              f"(target: {target*100:>5.2f}%)")

    print(f"{'='*80}\n")
