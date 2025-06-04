import torch
import numpy as np
from rdkit.Chem import rdMolAlign
from omtra.load.quick import datamodule_from_config
import omtra.load.quick as quick_load

from utils import get_gt_as_rdkit_ligand, find_rigid_alignment

cfg = quick_load.load_cfg(overrides=['task_group=no_protein']) # you can provide overrides to the default config via command-line syntax here
datamodule = datamodule_from_config(cfg)
train_dataset = datamodule.load_dataset("val")
pharmit_dataset = train_dataset.datasets['pharmit']

batch_size = 128
num_total_samples = 256 # len(pharmit_dataset) changed for testing
ckpt_path = '/home/ruh/Research/OMTRA/models/batch_295000.ckpt'


model = quick_load.omtra_from_checkpoint(ckpt_path).cuda().eval()

records = []
delta = [] # evaluating how different kabsch_rmsd is to rdkit_rmsd

for i in range(0, num_total_samples, batch_size):    
    current_batch_idx = range(i, min(i + batch_size, num_total_samples))
    
    g_list = [] 
    for dataset_idx in current_batch_idx:
        g_list.append(pharmit_dataset[('ligand_conformer', dataset_idx)])
    
    sampled_systems = model.sample(
        task_name='ligand_conformer',
        g_list=g_list,
        device="cuda",
        n_replicates=1,
        unconditional_n_atoms_dist='pharmit',
        n_timesteps=200,
        visualize=True,
        extract_latents_for_confidence = True 
    )

    print(".", end="")

    for _ in range(0, batch_size):
        pred_coords = sampled_systems[_].g.nodes["lig"].data["x_1"].cpu()
        gt_coords   = g_list[_].nodes['lig'].data['x_1_true'].cpu()
    
        pred_coords_rdkit = sampled_systems[_].get_rdkit_ligand()
        gt_coords_rdkit   = get_gt_as_rdkit_ligand(g_list[_])

        
        R, t = find_rigid_alignment(pred_coords, gt_coords)
        pred_aligned = (R.mm(pred_coords.T)).T + t
        kabsch_rmsd = torch.sqrt(((pred_aligned - gt_coords)**2).sum(axis=1).mean())
    
        rdkit_rmsd = rdMolAlign.GetBestRMS(pred_coords_rdkit, gt_coords_rdkit)
    
        # print(kabsch_rmsd, rdkit_rmsd)
        delta.append(kabsch_rmsd - rdkit_rmsd)
        
        latents = {
            "node_scalar_features" : sampled_systems[_].g.nodes["lig"].data["node_scalar_features"],
            "node_vec_features" : sampled_systems[_].g.nodes["lig"].data["node_vec_features"],
            "node_positions" : sampled_systems[_].g.nodes["lig"].data["node_positions"],
        }
        
        records.append({
            "latents"    : latents,
            "coords_gt"  : gt_coords,
            "coords_pred": pred_coords,
            "GetBestRMS" : rdkit_rmsd,
            "Kabsch_RMSD" : kabsch_rmsd
        })

# print stats
delta = np.array(delta)
print(f"\nMean delta (Kabsch - GetBestRMS): {delta.mean()}")    

out_file = "confidence_dataset.pt"
torch.save(records, out_file)
print(f"✓ Saved {len(records)} examples to {out_file}")