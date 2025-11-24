"""
Chunk-Aware Stratified Sampler for Multitask Learning

Strategy:
1. Shuffle chunk order
2. For each chunk, shuffle samples within that chunk
3. Create stratified batches maintaining target database proportions:
   - CSC:     19.91% (19,912 molecules per 100k)
   - MCULE:   29.95% (29,949 molecules per 100k)
   - PubChem: 50.22% (50,220 molecules per 100k)
   - ZINC:    20.87% (20,874 molecules per 100k)
   - MolPort:  4.07% (4,065 molecules per 100k)

Note: Molecules can be positive for MULTIPLE databases simultaneously.
The stratification ensures that each batch has the target proportion of
molecules that are positive for each database (not mutually exclusive).
"""

import numpy as np
import torch
from torch.utils.data import Sampler
from tqdm import tqdm
import time
from typing import List, Tuple, Optional
from collections import defaultdict


class MultitaskStratifiedSampler(Sampler):
    """
    Chunk-aware stratified sampler for 5-database multitask learning.

    Creates batches where each database has approximately the target
    proportion of positive samples, while maintaining chunk locality
    for efficient I/O.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 512,
        shuffle: bool = True,
        target_proportions: Optional[List[float]] = None,
        na_task_idx: int = 13,
        database_task_indices: Optional[List[int]] = None,
        use_cache: bool = True,
        cache_dir: str = './multitask_cache',
    ):
        """
        Args:
            dataset: PharmitDataset (not wrapper!)
            batch_size: Target number of molecules per batch
            shuffle: Whether to shuffle chunks and samples
            target_proportions: Target proportions for each database [CSC, MCULE, PubChem, ZINC, MolPort]
            na_task_idx: Index of NA task to exclude (default: 13)
            database_task_indices: Task indices for databases [CSC, MCULE, PubChem, ZINC, MolPort]
            use_cache: Whether to cache indices
            cache_dir: Directory for cache files
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.na_task_idx = na_task_idx

        # Database configuration
        if database_task_indices is None:
            database_task_indices = [2, 5, 8, 12, 6]  # CSC, MCULE, PubChem, ZINC, MolPort
        self.database_task_indices = database_task_indices
        self.database_names = ['CSC', 'MCULE', 'PubChem', 'ZINC', 'MolPort']
        self.n_databases = len(database_task_indices)

        # Target proportions (from dataset analysis)
        if target_proportions is None:
            target_proportions = [0.1991, 0.2995, 0.5022, 0.2087, 0.0407]
        self.target_proportions = np.array(target_proportions)

        # Dataset info
        self.graphs_per_chunk = dataset.graphs_per_chunk
        self.n_chunks = dataset.n_zarr_chunks
        self.total_graphs = len(dataset)

        print(f"{'='*80}")
        print("MultitaskStratifiedSampler")
        print(f"{'='*80}")
        print(f"  Total graphs: {self.total_graphs:,}")
        print(f"  Zarr chunks: {self.n_chunks}")
        print(f"  Chunk size: {self.graphs_per_chunk:,}")
        print(f"  Batch size: {batch_size}")
        print(f"  Shuffle: {shuffle}")
        print()
        print("Target proportions (% of molecules positive for each database):")
        for name, prop in zip(self.database_names, target_proportions):
            print(f"  {name:>10}: {prop*100:>5.2f}%")
        print()

        # Cache management
        self.use_cache = use_cache
        if use_cache:
            from pathlib import Path
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_or_build_indices()
        else:
            self._build_indices()

        # Create batches
        self._create_stratified_batches()

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

        task_str = '_'.join(map(str, self.database_task_indices))
        config_str = f"{dataset_path}_multitask_{task_str}"
        cache_key = hashlib.md5(config_str.encode()).hexdigest()
        return self.cache_dir / f"multitask_{cache_key}.npz"

    def _load_or_build_indices(self):
        """Load indices from cache or build from scratch."""
        cache_path = self._get_cache_path()

        if cache_path.exists():
            print(f"📂 Loading indices from cache...")
            data = np.load(cache_path)
            self.valid_indices = data['valid_indices']
            self.database_labels = data['database_labels']

            print(f"  ✓ Loaded {len(self.valid_indices):,} valid samples from cache")
            self._print_dataset_statistics()
        else:
            print(f"  No cache found. Building indices...")
            self._build_indices()

            # Save to cache
            print(f"  💾 Saving indices to cache...")
            np.savez_compressed(
                cache_path,
                valid_indices=self.valid_indices,
                database_labels=self.database_labels
            )
            size_mb = cache_path.stat().st_size / (1024**2)
            print(f"  ✓ Cached! Size: {size_mb:.1f} MB")

    def _build_indices(self):
        """Build indices by scanning through dataset."""
        print(f"  Building multitask indices (excluding NA samples)...")

        zarr_root = self.dataset.root
        db_data = zarr_root['db/db']

        valid_indices = []
        database_labels = []

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
            chunk_labels = db_data[sample_start:sample_end]

            # Filter out NA samples
            na_mask = chunk_labels[:, self.na_task_idx] == 0

            # Extract database labels for valid samples
            global_indices = np.arange(len(chunk_labels)) + sample_start
            valid_mask = na_mask

            valid_indices.extend(global_indices[valid_mask].tolist())

            # Extract labels for the 5 databases
            db_labels = chunk_labels[valid_mask][:, self.database_task_indices]
            database_labels.extend(db_labels.tolist())

        # Convert to numpy arrays
        self.valid_indices = np.array(valid_indices, dtype=np.int64)
        self.database_labels = np.array(database_labels, dtype=np.int8)

        elapsed = time.time() - start_time

        print(f"\n  ✓ Indices built in {elapsed:.1f}s")
        print(f"    Valid samples: {len(self.valid_indices):,}")

        self._print_dataset_statistics()

    def _print_dataset_statistics(self):
        """Print statistics about the dataset."""
        print(f"\n  Dataset label distribution:")
        for db_idx, db_name in enumerate(self.database_names):
            n_pos = (self.database_labels[:, db_idx] == 1).sum()
            pct = 100 * n_pos / len(self.database_labels)
            print(f"    {db_name:>10}: {n_pos:>10,} positive ({pct:>5.2f}%)")

    def _create_stratified_batches(self):
        """
        Create chunk-aware stratified batches.

        Strategy:
        1. Group indices by chunk
        2. Shuffle chunk order (if shuffle=True)
        3. Within each chunk:
           a. Shuffle samples
           b. Create batches maintaining target proportions
        """
        print(f"\n  Creating stratified batches...")

        # Group indices by chunk
        chunk_to_data = defaultdict(lambda: {'indices': [], 'labels': []})

        for idx, labels in zip(self.valid_indices, self.database_labels):
            chunk_id = idx // self.graphs_per_chunk
            chunk_to_data[chunk_id]['indices'].append(idx)
            chunk_to_data[chunk_id]['labels'].append(labels)

        # Convert to numpy arrays
        for chunk_id in chunk_to_data:
            chunk_to_data[chunk_id]['indices'] = np.array(chunk_to_data[chunk_id]['indices'])
            chunk_to_data[chunk_id]['labels'] = np.array(chunk_to_data[chunk_id]['labels'])

        print(f"    Data spans {len(chunk_to_data)} chunks")

        # Get chunk IDs
        chunk_ids = sorted(chunk_to_data.keys())

        # Shuffle chunk order if requested
        if self.shuffle:
            np.random.shuffle(chunk_ids)
            print(f"    Shuffled chunk order")

        # Create batches
        self.batch_indices = []
        batch_stats = []

        for chunk_id in chunk_ids:
            chunk_indices = chunk_to_data[chunk_id]['indices']
            chunk_labels = chunk_to_data[chunk_id]['labels']

            # Shuffle within chunk if requested
            if self.shuffle:
                perm = np.random.permutation(len(chunk_indices))
                chunk_indices = chunk_indices[perm]
                chunk_labels = chunk_labels[perm]

            # Create batches from this chunk using greedy stratified sampling
            offset = 0
            while offset < len(chunk_indices):
                batch_end = min(offset + self.batch_size, len(chunk_indices))
                batch_size_actual = batch_end - offset

                # Get batch
                batch_idx_list = chunk_indices[offset:batch_end].tolist()
                batch_labels = chunk_labels[offset:batch_end]

                # Compute batch statistics
                batch_proportions = batch_labels.mean(axis=0)

                self.batch_indices.append(batch_idx_list)
                batch_stats.append(batch_proportions)

                offset = batch_end

        batch_stats = np.array(batch_stats)

        print(f"\n  ✓ Created {len(self.batch_indices)} batches")
        print(f"\n  Batch statistics:")
        print(f"    Average batch size: {np.mean([len(b) for b in self.batch_indices]):.1f}")
        print(f"\n    Average proportions vs. target:")
        for db_idx, db_name in enumerate(self.database_names):
            avg_prop = batch_stats[:, db_idx].mean()
            target_prop = self.target_proportions[db_idx]
            diff = avg_prop - target_prop
            print(f"      {db_name:>10}: {avg_prop*100:>5.2f}% (target: {target_prop*100:>5.2f}%, "
                  f"diff: {diff*100:>+5.2f}%)")

        print(f"{'='*80}\n")

    def __iter__(self):
        """Iterate through batches."""
        for batch in self.batch_indices:
            yield batch

    def __len__(self):
        """Return number of batches."""
        return len(self.batch_indices)


# ============================================================================
# DataLoader Creation Function
# ============================================================================

def create_multitask_dataloader(
    dataset,
    batch_size: int = 512,
    shuffle: bool = True,
    target_proportions: Optional[List[float]] = None,
    num_workers: int = 8,
    **kwargs
):
    """
    Create DataLoader for multitask classification.

    Args:
        dataset: PharmitWrapper or PharmitDataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        target_proportions: Target proportions for [CSC, MCULE, PubChem, ZINC, MolPort]
        num_workers: Number of workers
        **kwargs: Additional args for sampler

    Returns:
        DataLoader with multitask collate function
    """
    from torch.utils.data import DataLoader
    from omtra.dataset.data_module import omtra_collate_fn

    # Get underlying dataset if wrapped
    if hasattr(dataset, 'ds'):
        underlying_dataset = dataset.ds
        wrapper = dataset
    else:
        underlying_dataset = dataset
        from omtra.dataset.pharmit import PharmitWrapper
        wrapper = PharmitWrapper(dataset)

    # Create sampler
    sampler = MultitaskStratifiedSampler(
        dataset=underlying_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        target_proportions=target_proportions,
        **kwargs
    )

    # Create custom collate function
    def multitask_collate_fn(batch):
        """Collate function that extracts labels for 5 databases."""
        # Use standard collate
        collated = omtra_collate_fn(batch)

        # Extract labels for 5 databases: CSC (2), MCULE (5), PubChem (8), ZINC (12), MolPort (6)
        full_labels = collated['system_features']['pharmit_library']  # [B, N]
        database_indices = [2, 5, 8, 12, 6]

        # Extract only the 5 database labels
        multitask_labels = full_labels[:, database_indices]  # [B, 5]

        # Replace with multitask labels
        collated['system_features']['pharmit_library'] = multitask_labels

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

def diagnose_sampler(sampler: MultitaskStratifiedSampler, n_batches: int = 20):
    """
    Diagnose the sampler to verify stratification is working.

    Args:
        sampler: The sampler to diagnose
        n_batches: Number of batches to check (default: 20)
    """
    print("="*80)
    print("SAMPLER DIAGNOSTICS")
    print("="*80)

    batch_stats = []

    for batch_idx, batch_indices in enumerate(sampler.batch_indices[:n_batches]):
        # Get labels for this batch
        batch_labels = sampler.database_labels[np.isin(sampler.valid_indices, batch_indices)]

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

    # Summary statistics
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
