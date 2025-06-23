from omtra.dataset.latent import LatentDataset, omtra_collate_fn
from torch.utils.data import DataLoader

DATA_DIR = "./" 
SPLIT_NAME = "confidence_dataset" 
BATCH_SIZE = 32

confidence_dataset = LatentDataset(split=SPLIT_NAME, processed_data_dir=DATA_DIR)
data_loader = DataLoader(confidence_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=omtra_collate_fn)

for batch in data_loader:
    print(f"Keys in batch: {list(batch.keys())}")
    print(f"Task Name: {batch['task_name']}")
    print(f"Dataset Name: {batch['dataset_name']}")
    print("____________________________________________")
    
    task = batch['task_name']
    dataset = batch['dataset_name']
    graph = batch['graph']
    system_features = batch['system_features']

    # Inspect the graph
    print(f"Number of graphs in batch: {graph.batch_size}")
    print(f"Total nodes in batch: {graph.num_nodes()}")
    print(f"Total edges in batch: {graph.num_edges()}")

    print("\nNode features for 'lig' type:")
    for name, tensor in graph.nodes['lig'].data.items():
        print(f"  - '{name}': tensor of shape {tensor.shape} and type {tensor.dtype}")
        print(f"    - Example values: {tensor[0][:3]}...")
    
    # Inspect the system features
    print("\nSystem Features (Metrics):")
    for name, tensor in system_features.items():
        print(f"  - '{name}': tensor of shape {tensor.shape} and type {tensor.dtype}")
        print(f"    - Example values: {tensor[0]}")

    break