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

from omtra.utils import omtra_root
from pathlib import Path
OmegaConf.register_new_resolver("omtra_root", omtra_root, replace=True)

default_config_path = Path(omtra_root()) / 'configs'
default_config_path = str(default_config_path)


from rdkit import Chem

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

@hydra.main(config_path=default_config_path, config_name="sample")
def main(cfg):
    # 1) resolve checkpoint path
    ckpt_path = Path(cfg.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found")
    
    # 2) load the exact train‐time config
    train_cfg_path = ckpt_path.parent.parent / '.hydra' / 'config.yaml'
    train_cfg = quick_load.load_trained_model_cfg(train_cfg_path)
    
    # override anything in the training config file with anything passed in (sample.yaml by default)
    # or anything passed in via the command line, i.e., if you need to override the pharmit or plinder paths
    merged_cfg = OmegaConf.merge(train_cfg, cfg)

    # get device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 4) instantiate datamodule & model
    dm  = quick_load.datamodule_from_config(merged_cfg)
    multitask_dataset = dm.load_dataset('val')
    model = quick_load.omtra_from_checkpoint(ckpt_path).to(device)
    
    # get task we are sampling for
    task_name: str = cfg.task
    task: Task = task_name_to_class(task_name)

    # get raw dataset object
    if cfg.dataset == 'plinder':
        plinder_link_version = task.plinder_link_version
        dataset = multitask_dataset.datasets['plinder'][plinder_link_version]
    elif cfg.dataset == 'pharmit':
        dataset = multitask_dataset.datasets['pharmit']
    else:
        raise ValueError(f"Unknown dataset {cfg.dataset}")

    # get g_list
    if task.unconditional:
        g_list = None
        n_replicates = cfg.n_samples
    else:
        dataset_idxs = range(cfg.dataset_start_idx, cfg.dataset_start_idx + cfg.n_samples)
        g_list = [ dataset[(task_name, i)].to(device) for i in dataset_idxs ]
        n_replicates = cfg.n_replicates

    sampled_systems = model.sample(
        g_list=g_list,
        n_replicates=n_replicates,
        task_name=task_name,
        unconditional_n_atoms_dist=cfg.dataset,
        device=device,
        n_timesteps=cfg.n_timesteps,
        visualize=cfg.visualize,
    )

    if cfg.output_dir is None:
        output_dir = ckpt_path.parent.parent / 'samples'
    else:
        output_dir = Path(cfg.output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving samples to {output_dir}")

    if cfg.visualize:
        for i, sys in enumerate(sampled_systems):
            xt_traj_mols = sys.build_traj(ep_traj=False, lig=True)
            xhat_traj_mols = sys.build_traj(ep_traj=True, lig=True)
            xt_file = output_dir / f"{task_name}_xt_traj_{i}.sdf"
            xhat_file = output_dir / f"{task_name}_xhat_traj_{i}.sdf"
            write_mols_to_sdf(xt_traj_mols['lig'], xt_file)
            write_mols_to_sdf(xhat_traj_mols['lig'], xhat_file)
    else:
        rdkit_mols = [ s.get_rdkit_ligand() for s in sampled_systems ]
        output_file = output_dir / f"{task_name}_samples.sdf"
        write_mols_to_sdf(rdkit_mols, output_file)
        

if __name__ == "__main__":
    main()