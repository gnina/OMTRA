# predict.py
import os
from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate
import pytorch_lightning as pl
import omtra.load.quick as quick_load
import torch

from omtra.tasks.tasks import Task
from omtra.tasks.register import task_name_to_class
from omtra.eval.register import get_eval
import json
from biotite.structure.io.pdbx import CIFFile
import biotite.structure as struc

from omtra.utils import omtra_root
from pathlib import Path
OmegaConf.register_new_resolver("omtra_root", omtra_root, replace=True)

default_config_path = Path(omtra_root()) / 'configs'
default_config_path = str(default_config_path)

from rdkit import Chem
import argparse

def parse_args():
    p = argparse.ArgumentParser(
        description="Run the OMTRA sampler (originally via Hydra)"
    )
    # Core sampling args
    p.add_argument(
        "checkpoint",
        type=Path,
        help="Path to the model checkpoint (required)"
    )
    p.add_argument(
        "--task",
        type=str,
        # default="denovo_ligand",
        help="Task to sample for (e.g. denovo_ligand)",
        required=True
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="pharmit",
        help="Dataset to sample from (e.g. pharmit)"
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples to draw"
    )
    p.add_argument(
        "--n_replicates",
        type=int,
        default=1,
        help=(
            "For conditional sampling: number of replicates per input sample"
        )
    )
    p.add_argument(
        "--dataset_start_idx",
        type=int,
        default=0,
        help="Index in the dataset to start sampling from"
    )
    p.add_argument(
        "--n_timesteps",
        type=int,
        default=250,
        help="Number of integration steps to take when sampling"
    )
    p.add_argument(
        "--visualize",
        action="store_true",
        help="If set, visualize the sampling process"
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to write outputs to"
    )
    p.add_argument(
        "--pharmit_path",
        type=str,
        default=None,
        help="Path to the Pharmit dataset (optional)"
    )
    p.add_argument(
        "--plinder_path",
        type=str,
        default=None,
        help="Path to the Plinder dataset (optional)"
    )
    p.add_argument("--metrics", action="store_true", help="If set, compute metrics for the samples")

    return p.parse_args()


def write_mols_to_sdf(mols, filename):
    """
    Write a list of RDKit molecules to an SDF file.
    """
    writer = Chem.SDWriter(str(filename))
    writer.SetKekulize(False)
    for mol in mols:
        if mol is not None:
            writer.write(mol)
    writer.close()

def write_arrays_to_cif(arrays, filename):
    cif_file = CIFFile()
    arr_stack = struc.stack(arrays)
    struc.io.pdbx.set_structure(cif_file, arr_stack)
    cif_file.write(filename)

def main(args):
    # 1) resolve checkpoint path
    ckpt_path = args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found")
    
    # 2) load the exact train‐time config
    train_cfg_path = ckpt_path.parent.parent / '.hydra' / 'config.yaml'
    train_cfg = quick_load.load_trained_model_cfg(train_cfg_path)

    # apply some changes to the config to enable sampling
    train_cfg.num_workers = 0
    if args.pharmit_path:
        train_cfg.pharmit_path = args.pharmit_path
    if args.plinder_path:
        train_cfg.plinder_path = args.plinder_path

    # get device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 4) instantiate datamodule & model
    dm  = quick_load.datamodule_from_config(train_cfg)
    multitask_dataset = dm.load_dataset('val')
    model = quick_load.omtra_from_checkpoint(ckpt_path).to(device).eval()
    
    # get task we are sampling for
    task_name: str = args.task
    task: Task = task_name_to_class(task_name)

    # get raw dataset object
    if args.dataset == 'plinder':
        plinder_link_version = task.plinder_link_version
        dataset = multitask_dataset.datasets['plinder'][plinder_link_version]
    elif args.dataset == 'pharmit':
        dataset = multitask_dataset.datasets['pharmit']
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    # get g_list
    if task.unconditional:
        g_list = None
        n_replicates = args.n_samples
    else:
        dataset_idxs = range(args.dataset_start_idx, args.dataset_start_idx + args.n_samples)
        g_list = [ dataset[(task_name, i)].to(device) for i in dataset_idxs ]
        n_replicates = args.n_replicates

    sampled_systems = model.sample(
        g_list=g_list,
        n_replicates=n_replicates,
        task_name=task_name,
        unconditional_n_atoms_dist=args.dataset,
        device=device,
        n_timesteps=args.n_timesteps,
        visualize=args.visualize,
    )

    if args.output_dir is None:
        output_dir = ckpt_path.parent.parent / 'samples'
    else:
        output_dir = args.output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving samples to {output_dir}")

    if args.visualize:
        for i, sys in enumerate(sampled_systems):
            prot = 'protein_identity' in task.groups_present
            xt_traj_mols = sys.build_traj(ep_traj=False, lig=True, prot=prot)
            xhat_traj_mols = sys.build_traj(ep_traj=True, lig=True, prot=prot)
            xt_file = output_dir / f"{task_name}_xt_traj_{i}.sdf"
            xhat_file = output_dir / f"{task_name}_xhat_traj_{i}.sdf"
            write_mols_to_sdf(xt_traj_mols['lig'], xt_file)
            write_mols_to_sdf(xhat_traj_mols['lig'], xhat_file)
            
            if 'protein_identity' in task.groups_present or 'protein_structure' in task.groups_present:
                write_arrays_to_cif(xt_traj_mols['prot'], xt_file.with_suffix('.cif'))
                write_arrays_to_cif(xhat_traj_mols['prot'], xhat_file.with_suffix('.cif'))
    else:
        rdkit_mols = [ s.get_rdkit_ligand() for s in sampled_systems ]
        output_file = output_dir / f"{task_name}_samples.sdf"
        write_mols_to_sdf(rdkit_mols, output_file)

    if args.metrics:
        eval_fn = get_eval(task_name)
        metrics = eval_fn(sampled_systems)
        metrics_file = output_dir / f"{task_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
        for k, v in metrics.items():
            print(f"{k}: {v:.3f}")
        
        

if __name__ == "__main__":
    args = parse_args()
    main(args)