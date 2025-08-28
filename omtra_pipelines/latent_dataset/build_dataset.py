import zarr
from tqdm import tqdm
from pathlib import Path

import torch
import numpy as np
import json
from omegaconf import OmegaConf

from rdkit.Chem import rdMolAlign
from omtra.load.quick import datamodule_from_config
from omtra.utils.graph import build_lookup_table
import omtra.load.quick as quick_load

from zarr_helpers import init_zarr_store_latents 
from utils import get_gt_as_rdkit_ligand, find_rigid_alignment

# Configuration; 
# includes things such as :
# 1) how we run inference (eg. sampling steps)
# 2) how and where to save the latents (eg. # of chunks, output path)
# 3) which model & dataset was used

cfg = {
    'source_split': 'val',
    'task_name': 'ligand_conformer',  # model will run inference on this task
    'dataset_name': 'pharmit',
    'generation': {
        'n_timesteps': 200,
        'n_replicates': 1,
        'unconditional_n_atoms_dist': 'pharmit'
    },
    'batch_size': 64,
    'num_total_samples': "full",  # Use "full" for entire dataset 
    'ckpt_path': '../../models/batch_295000.ckpt',
    'output_zarr_path': Path("confidence_dataset.zarr"),
    'n_zarr_chunks': 2
}

hydra_cfg = quick_load.load_cfg(overrides=['task_group=no_protein']) # you can provide overrides to the default config via command-line syntax here
datamodule = datamodule_from_config(hydra_cfg)
train_dataset = datamodule.load_dataset(cfg['source_split'])
pharmit_dataset = train_dataset.datasets[cfg['dataset_name']]

model = quick_load.omtra_from_checkpoint(cfg['ckpt_path']).cuda().eval()

# populate "full" dataset option
if cfg['num_total_samples'] == "full":
    cfg['num_total_samples'] = len(pharmit_dataset)
    print(f"Using full dataset: {cfg['num_total_samples']} samples")

# Get the total number of atoms in the dataset
graph_lookup = pharmit_dataset.slice_array('lig/node/graph_lookup', 0, cfg['num_total_samples'])
atom_counts_per_sample = graph_lookup[:, 1] - graph_lookup[:, 0]
total_atoms = int(graph_lookup[-1, 1])  # idx of the last atom of the last sample

dataset_spec = {
    'n_mols' : cfg['num_total_samples'],
    'n_atoms': total_atoms,
    'scalar_dim': model.vector_field.n_hidden_scalars,
    'vector_dim': model.vector_field.n_vec_channels
}

graph_lookup_table = build_lookup_table(atom_counts_per_sample)

_, root = init_zarr_store_latents(
    store_path=cfg['output_zarr_path'], 
    totals=dataset_spec,
    n_chunks=cfg['n_zarr_chunks']
)

# Pre-populate the lookup table, as it's fully computed now
root['metadata/graph_lookup'][:] = graph_lookup_table

# resolve the configs of hydra and save a snapshot of it to read later
resolved_datamodule_config = OmegaConf.to_container(hydra_cfg.task_group.datamodule, resolve=True)

essential_config = {
    'task_name': cfg['task_name'],
    'dataset_name': cfg['dataset_name'],

    'datamodule_config': resolved_datamodule_config,
    'fake_atom_p': float(hydra_cfg.fake_atom_p),
    'source_split': cfg['source_split'],
    
    'model_checkpoint': str(cfg['ckpt_path']),
    'generation_config': cfg['generation']
}

# Save config inside Zarr store as root attributes
root.attrs['config'] = essential_config

records = []
delta = [] # evaluating how different kabsch_rmsd is to rdkit_rmsd

for i in tqdm(range(0, cfg['num_total_samples'], cfg['batch_size']), desc="Processing Batches"): 
    current_batch_idx = range(i, min(i + cfg['batch_size'], cfg['num_total_samples']))
    
    g_list = []  
    for dataset_idx in current_batch_idx:
        g_list.append(pharmit_dataset[(cfg['task_name'], dataset_idx)])
    
    sampled_systems = model.sample(
        task_name=cfg['task_name'],
        g_list=g_list,
        device="cuda",
        n_replicates=cfg['generation']['n_replicates'],
        unconditional_n_atoms_dist=cfg['generation']['unconditional_n_atoms_dist'],
        n_timesteps=cfg['generation']['n_timesteps'],
        visualize=True,
        extract_latents_for_confidence=True 
    )

    for j, sample_idx in enumerate(current_batch_idx):
        # Pos of atom in Zarr arrays
        atom_start, atom_end = graph_lookup_table[sample_idx]

        pred_coords = sampled_systems[j].g.nodes["lig"].data["x_1"].cpu()
        gt_coords   = g_list[j].nodes['lig'].data['x_1_true'].cpu()
    
        pred_coords_rdkit = sampled_systems[j].get_rdkit_ligand()
        gt_coords_rdkit   = get_gt_as_rdkit_ligand(g_list[j])

        pred_aligned = find_rigid_alignment(pred_coords, gt_coords)
        kabsch_rmsd = torch.sqrt(((pred_aligned - gt_coords)**2).sum(axis=1).mean()).item()
    
        rdkit_rmsd = rdMolAlign.GetBestRMS(pred_coords_rdkit, gt_coords_rdkit)
    
        # print(kabsch_rmsd, rdkit_rmsd)
        delta.append(kabsch_rmsd - rdkit_rmsd)
        
        latents = {
            "node_scalar_features" : sampled_systems[j].g.nodes["lig"].data["node_scalar_features"].cpu().numpy(),
            "node_vec_features" : sampled_systems[j].g.nodes["lig"].data["node_vec_features"].cpu().numpy(),
            "node_positions" : sampled_systems[j].g.nodes["lig"].data["node_positions"].cpu().numpy(),
        }

        # write to zarr
        root['latents/node_scalar_features'][atom_start:atom_end] = latents["node_scalar_features"]
        root['latents/node_vec_features'][atom_start:atom_end] = latents["node_vec_features"]
        root['latents/node_positions'][atom_start:atom_end] = latents["node_positions"]
        
        root['coordinates/coords_gt'][atom_start:atom_end] = gt_coords.numpy()
        root['coordinates/coords_pred'][atom_start:atom_end] = pred_coords.numpy()
        
        root['metrics/rdkit_rmsd'][sample_idx] = rdkit_rmsd
        root['metrics/kabsch_rmsd'][sample_idx] = kabsch_rmsd

        # # print shapes
        # print("example", i + j)
        # print("latents shape (node_scalar_features) ",latents['node_scalar_features'].shape)
        # print("latents shape (node_vec_features) ",latents['node_vec_features'].shape)
        # print("latents shape (node_positions) ",latents['node_positions'].shape)
        # print("gt_coords shape ",gt_coords.shape)
        # print("pred_coords ", pred_coords.shape)

# print stats
delta = np.array(delta)
print(f"\nMean delta (Kabsch - GetBestRMS): {delta.mean()}")