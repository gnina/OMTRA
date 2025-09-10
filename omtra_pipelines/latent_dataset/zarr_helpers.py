import zarr
import numpy as np
from pathlib import Path

def init_zarr_store_latents(store_path: Path, totals: dict, n_chunks: int, node_types: list):
    store = zarr.storage.LocalStore(store_path)
    root = zarr.group(store=store)

    # Create main groups
    latents_group = root.create_group('latents')
    coords_group  = root.create_group('coordinates') 
    metadata_group = root.create_group('metadata') 
    metrics_group = root.create_group('metrics')

    # Extract dimensions from totals
    n_graphs = totals['n_mols']
    n_atoms = totals['n_atoms'] 
    scalar_dim = totals['scalar_dim'] # 256
    vector_dim = totals['vector_dim'] # 16
    
    # Calculate chunk sizes
    graphs_per_chunk = n_graphs // n_chunks
    atoms_per_chunk = n_atoms // n_chunks
    
    # latents
    for ntype in node_types:
        latents_group.create_array(f'{ntype}_scalar_features', shape=(n_atoms, scalar_dim), chunks=(atoms_per_chunk, scalar_dim), dtype=np.float32)
        latents_group.create_array(f'{ntype}_vec_features', shape=(n_atoms, vector_dim, 3), chunks=(atoms_per_chunk, vector_dim, 3), dtype=np.float32)
        latents_group.create_array(f'{ntype}_positions', shape=(n_atoms, 3), chunks=(atoms_per_chunk, 3), dtype=np.float32)
    
    # coords
    coords_group.create_array('coords_gt', shape=(n_atoms, 3), chunks=(atoms_per_chunk, 3), dtype=np.float32)
    coords_group.create_array('coords_pred', shape=(n_atoms, 3), chunks=(atoms_per_chunk, 3), dtype=np.float32)
    
    # metadata
    metadata_group.create_array('graph_lookup', shape=(n_graphs, 2), chunks=(graphs_per_chunk, 2), dtype=np.int64)

    # metrics
    metrics_group.create_array('rdkit_rmsd', shape=(n_graphs,), chunks=(graphs_per_chunk,), dtype=np.float32)
    metrics_group.create_array('kabsch_rmsd', shape=(n_graphs,), chunks=(graphs_per_chunk,), dtype=np.float32)

    print("Latent dataset Zarr structure:")
    print(root.tree())
    
    return store, root