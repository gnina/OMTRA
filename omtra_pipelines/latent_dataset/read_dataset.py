"""
Example script demonstrating how to load and use the LatentDataset for confidence model training.

The LatentDataset combines pre-computed latent features with original molecular features,
providing a complete graph suitable for training confidence models.
"""

from omtra.dataset.latent import LatentDataset, omtra_collate_fn
from torch.utils.data import DataLoader

# Configuration
DATA_DIR = "./" 
SPLIT_NAME = "confidence_dataset" 
BATCH_SIZE = 32
SHUFFLE = True
NUM_WORKERS = 0

def inspect_batch(batch):
    """Print detailed information about a batch for inspection."""
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Task: {batch['task_name']}")
    print(f"Dataset: {batch['dataset_name']}")
    
    graph = batch['graph']
    print(f"\nGraph info:")
    print(f"  - Batch size: {graph.batch_size}")
    print(f"  - Total nodes: {graph.num_nodes()}")
    print(f"  - Total edges: {graph.num_edges()}")
    print(f"  - Node types: {graph.ntypes}")
    print(f"  - Edge types: {len(graph.etypes)} edge types")
    
    print(f"\nLigand features:")
    for name, tensor in graph.nodes['lig'].data.items():
        print(f"  - {name}: {tensor.shape} {tensor.dtype}")
    
    print(f"\nSystem features (metrics):")
    for name, tensor in batch['system_features'].items():
        print(f"  - {name}: {tensor.shape} {tensor.dtype}")
        print(f"    Range: [{tensor.min():.3f}, {tensor.max():.3f}]")

def main():
    # Load dataset
    print("Loading LatentDataset...")
    dataset = LatentDataset(
        split=SPLIT_NAME, 
        processed_data_dir=DATA_DIR
    )
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        collate_fn=omtra_collate_fn,
        num_workers=NUM_WORKERS
    )
    
    # Process first batch as example
    print(f"\nProcessing first batch...")
    for batch in dataloader:
        inspect_batch(batch)
        break
    
    print(f"\nDataset ready for confidence model training!")
    print(f"Available features:")
    print(f"  - Original molecular features (x_1_true, a_1_true, c_1_true, etc.)")
    print(f"  - Latent features (scalar_latents, vec_latents, pos_latents)")
    print(f"  - Coordinates (coords_gt, coords_pred)")
    print(f"  - Quality metrics (rdkit_rmsd, kabsch_rmsd)")

if __name__ == "__main__":
    main()