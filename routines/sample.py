import os
from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate
import pytorch_lightning as pl
import omtra.load.quick as quick_load
import torch
from typing import List, Any, Iterator

from omtra.tasks.tasks import Task
from omtra.tasks.register import task_name_to_class
from omtra.eval.register import get_eval
from omtra.eval.system import write_mols_to_sdf, write_arrays_to_cif
import json

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

def generate_sample_names(n_systems: int, n_replicates: int) -> List[str]:
    """
    Generate names of the form 'sys_{system_idx}_rep_{rep_idx}'.

    Args:
        n_systems: Number of unique systems.
        n_replicates: Number of replicate samples per system.

    Returns:
        A list of strings like ['sys_0_rep_0', 'sys_0_rep_1', ..., 'sys_{n_systems-1}_rep_{n_replicates-1}'].
    """
    return [
        f"sys_{i}_rep_{j}"
        for i in range(n_systems)
        for j in range(n_replicates)
    ]


def group_samples_by_system(
    sample_names: List[str],
    sample_objects: List[Any],
    n_systems: int,
    n_replicates: int
) -> Iterator[List[Any]]:
    """
    Given parallel lists of names and objects, yield lists of objects
    grouped by system ID.

    Args:
        sample_names: List of names like 'sys_{i}_rep_{j}'.
        sample_objects: List of sample objects, same length & order as sample_names.
        n_systems: Total number of systems (must match max sys ID + 1).
        n_replicates: Number of replicates per system.

    Yields:
        A list of length `n_replicates` of sample_objects for each system in order 0..n_systems-1.

    Raises:
        ValueError: if lengths don’t match or names don’t parse correctly.
    """
    total = n_systems * n_replicates
    if len(sample_names) != total or len(sample_objects) != total:
        raise ValueError(
            f"Expected {total} samples (got {len(sample_names)}/{len(sample_objects)})"
        )

    # Prepare empty slots for each system
    grouped = {
        sys_id: [None] * n_replicates
        for sys_id in range(n_systems)
    }

    # Parse names and place objects into the right slot
    for name, obj in zip(sample_names, sample_objects):
        parts = name.split('_')
        try:
            sys_id = int(parts[1])
            rep_id = int(parts[3])
        except (IndexError, ValueError):
            raise ValueError(f"Sample name not in 'sys_X_rep_Y' format: '{name}'")

        if not (0 <= sys_id < n_systems) or not (0 <= rep_id < n_replicates):
            raise ValueError(f"Parsed out-of-range ids sys={sys_id}, rep={rep_id}")

        grouped[sys_id][rep_id] = obj

    # Yield each system's list in order
    for sys_id in range(n_systems):
        yield grouped[sys_id]

def write_ground_truth(
        n_systems: int,
        n_replicates: int,
        task: Task,
        output_dir: Path, 
        sampled_systems,
        g_list,
        prot_cif: bool = True
    ):
    for cond_idx in range(n_systems):

        # get an example system containing the ground truth information of interest
        sys_idx = cond_idx*n_replicates
        sys = sampled_systems[sys_idx]

        # directory for writing ground truth
        sys_gt_dir = output_dir / f"sys_{cond_idx}_gt"
        sys_gt_dir.mkdir(parents=True, exist_ok=True)

        # write the ground truth ligand
        gt_lig_file = sys_gt_dir / "ligand.sdf"
        sys.write_ligand(
            gt_lig_file, 
            ground_truth=True, 
            g=g_list[cond_idx].to('cpu'),
        )

        # write the ground truth protein if present
        if 'protein_identity' in task.groups_present:
            if prot_cif:
                gt_prot_file = sys_gt_dir / "protein.cif"
                sys.write_protein(gt_prot_file, ground_truth=True)
            else:
                sys.write_protein_pdb(sys_gt_dir, filename='protein', ground_truth=True)
            
        # write the ground truth pharmacophore
        if 'pharmacophore' in task.groups_present:
            gt_pharm_file = sys_gt_dir / "pharmacophore.xyz"
            sys.write_pharmacophore(
                gt_pharm_file, 
                ground_truth=True, 
                g=g_list[cond_idx].to('cpu')
                )

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

    # set coms if protein is present
    if 'protein_identity' in task.groups_present and (any(group in task.groups_present for group in ['ligand_identity', 'ligand_identity_condensed'])):
        coms = [ g.nodes['lig'].data['x_1_true'].mean(dim=0) for g in g_list ]
    else:
        coms = None

    sampled_systems = model.sample(
        g_list=g_list,
        n_replicates=n_replicates,
        task_name=task_name,
        unconditional_n_atoms_dist=args.dataset,
        device=device,
        n_timesteps=args.n_timesteps,
        visualize=args.visualize,
        coms=coms,
    )

    if args.output_dir is None:
        output_dir = ckpt_path.parent.parent / 'samples'
    else:
        output_dir = args.output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving samples to {output_dir}")

    if task.unconditional and not args.visualize:
        lig_samples = [ s.get_rdkit_ligand() for s in sampled_systems ]
        output_file = output_dir / f"{task_name}_lig.sdf"
        write_mols_to_sdf(lig_samples, output_file)
    elif task.unconditional and args.visualize:
        for i, sys in enumerate(sampled_systems):
            lig_xt_file = output_dir / f"sys{i}_xt.sdf"
            lig_xhat_file = output_dir / f"sys{i}_xhat.sdf"
            sys.write_ligand(lig_xt_file, trajectory=True, endpoint=False)
            sys.write_ligand(lig_xhat_file, trajectory=True, endpoint=True)
    elif not task.unconditional and not args.visualize:
        # write ground truth for everything in the system, once per system
        write_ground_truth(
            n_systems=len(g_list),
            n_replicates=n_replicates,
            task=task,
            output_dir=output_dir,
            sampled_systems=sampled_systems,
            g_list=g_list
            )

        # collect all the ligands for each system
        sample_names = generate_sample_names(
            n_systems=len(g_list), 
            n_replicates=n_replicates
        )
        for sys_id, replicates in enumerate(
            group_samples_by_system(
            sample_names=sample_names,
            sample_objects=sampled_systems,
            n_systems=len(g_list),
            n_replicates=n_replicates
            )
        ):
            sys_gt_dir = output_dir / f"sys_{sys_id}_gt"

            # write all ligands
            if 'ligand_structure' in task.groups_generated:
                ligands = [s.get_rdkit_ligand() for s in replicates]
                output_file = sys_gt_dir / f"gen_ligands.sdf"
                write_mols_to_sdf(ligands, output_file)

            if 'protein_structure' in task.groups_generated:
                # write all proteins
                proteins = [s.get_protein_array() for s in replicates]
                prot_file = sys_gt_dir / f"gen_prot.cif"
                write_arrays_to_cif(proteins, prot_file)

            if 'pharmacophore' in task.groups_generated:
                pharms = [s.get_pharmacaphore_from_graph(xyz=True) for s in replicates]
                pharm_file = sys_gt_dir / f"gen_pharm.cif"
                with open(pharm_file, 'w') as f:
                    f.write(sum(pharms))

    elif not task.unconditional and args.visualize:
        # write ground truth for everything in the system, once per system
        write_ground_truth(
            n_systems=len(g_list),
            n_replicates=n_replicates,
            task=task,
            output_dir=output_dir,
            sampled_systems=sampled_systems,
            g_list=g_list)
        
        # collect all the ligands for each system
        sample_names = generate_sample_names(
            n_systems=len(g_list), 
            n_replicates=n_replicates
        )
        for sample_idx, (sample_name, system) in enumerate(zip(
            sample_names, sampled_systems
        )):
            base_system_idx = sample_idx // n_replicates
            sys_gt_dir = output_dir / f"sys_{base_system_idx}_gt"
            

            if 'ligand_structure' in task.groups_generated:
                lig_xt_file = sys_gt_dir / f"{sample_name}_lig_xt.sdf"
                lig_xhat_file = sys_gt_dir / f"{sample_name}_lig_xhat.sdf"
                system.write_ligand(lig_xt_file, trajectory=True, endpoint=False)
                system.write_ligand(lig_xhat_file, trajectory=True, endpoint=True)

            if 'protein_structure' in task.groups_generated:
                prot_xt_file = sys_gt_dir / f"{sample_name}_prot_xt.cif"
                prot_xhat_file = sys_gt_dir / f"{sample_name}_prot_xhat.cif"
                system.write_protein(prot_xt_file, trajectory=True, endpoint=False)
                system.write_protein(prot_xhat_file, trajectory=True, endpoint=True)

            if 'pharmacophore' in task.groups_generated:
                pharm_xt_file = sys_gt_dir / f"{sample_name}_pharm_xt.cif"
                pharm_xhat_file = sys_gt_dir / f"{sample_name}_pharm_xhat.cif"
                system.write_pharmacophore(pharm_xt_file, trajectory=True, endpoint=False)
                system.write_pharmacophore(pharm_xhat_file, trajectory=True, endpoint=True)

    if args.metrics and model.eval_config is not None:
        metrics = {}
        for eval in model.eval_config.get(task_name, []):
            for eval_name, config  in eval.items():
                eval_fn = get_eval(eval_name)
                metrics.update(eval_fn(sampled_systems, config.get("params", {})))
                
        metrics_file = output_dir / f"{task_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
        for k, v in metrics.items():
            print(f"{k}: {v:.3f}")
        
        

if __name__ == "__main__":
    args = parse_args()
    main(args)