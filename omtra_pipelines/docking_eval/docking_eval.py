from typing import Dict, List
from pathlib import Path
import argparse
import os

import pandas as pd
import torch
import dgl
import gc
import sys
import time

from omtra.tasks.tasks import Task
from omtra.tasks.register import task_name_to_class
from omtra.load.quick import datamodule_from_config
import omtra.load.quick as quick_load
from omtra.eval.system import SampledSystem, write_arrays_to_pdb, write_mols_to_sdf
from omtra.eval.metrics.compute import (
    compute_metrics,
    system_pairs_from_path,
    determine_pb_mode,
)
from routines.sample import write_ground_truth, generate_sample_names, group_samples_by_system



def parse_args():
    p = argparse.ArgumentParser(description='Evaluate ligand poses')

    # option for early exit
    p.add_argument('--exit_if_done', action='store_true', help="Stop execution if the output metric file already exists.")

    # --- Mutually exclusive group inside IO ---
    io = p.add_argument_group("Input/Output Options")
    group = io.add_mutually_exclusive_group(required=True)
    group.add_argument("--ckpt_path", type=Path, default=None, help='Path to model checkpoint.')
    group.add_argument("--samples_dir", type=Path, default=None, help='Path to samples. Use existing samples, do not sample a model')
    io.add_argument("--sample_only", action="store_true", help='Only sample the model. Do not compute metrics.')  
    io.add_argument("--output_dir", type=Path, default=None, help='Output directory.')
    io.add_argument("--sys_info_file", type=str, default=None, help="Path to the system info file (optional).")

    # --- Sampling options ---
    sampling = p.add_argument_group("Sampling Options")

    sampling.add_argument("--task", type=str, help='Task to sample for (e.g. denovo_ligand).', required=True)
    sampling.add_argument("--dataset", type=str, default="plinder", help='Dataset.')
    sampling.add_argument("--split", type=str, default="test", help='Data split (i.e., train, val).')
    sampling.add_argument("--dataset_start_idx", type=int, default=0, help="Index in the dataset to start sampling from.")
    sampling.add_argument("--sample_start_idx", type=int, default=None, help="Index in the sample directory to start getting samples from.")
    sampling.add_argument("--sys_idx_file", type=str, default=None, help='Path to a file with pre-selected system indices.')
    sampling.add_argument("--plinder_path", type=str, default=None, help="Path to the Plinder dataset (optional).")
    sampling.add_argument("--crossdocked_path", type=str, default=None, help="Path to the Crossdocked dataset (optional).")

    sampling.add_argument("--n_samples", type=int, default=None, help='Number of samples to evaluate.')
    sampling.add_argument("--n_replicates", type=int, help="Number of replicates per input sample.", required=True)
    sampling.add_argument("--n_timesteps", type=int, default=250, help="Number of integration steps to take when sampling.")
    sampling.add_argument("--n_lig_atom_margin", type=float, default=0.075, help="Margin for number of ligand atoms for de novo design if using number of ground truth ligand atoms.")

    sampling.add_argument('--fixed_coord_max_std', type=float, default=None, help='Maximum sampled standard deviation of noise added to coordinates of fixed atoms. If not set, falls back to the value used during training.')
    sampling.add_argument('--fixed_coord_std', type=float, default=None, help='Optionally fix the standard deviation of the noise added to coordinates of fixed atoms')
    sampling.add_argument('--fixed_token_max_prob', type=float, default=None, help='Maximum sampled probability of replacing categorical tokens of fixed atoms. If not set, falls back to the value used during training.')
    sampling.add_argument('--fixed_token_prob', type=float, default=None, help='Optionally fix the probability of replacing categorical tokens of fixed atoms')

    sampling.add_argument("--stochastic_sampling", action="store_true", help="If set, perform stochastic sampling.")
    sampling.add_argument("--noise_scaler", type=float, default=1.0, help="Noise scaling param for stochastic sampling.")
    sampling.add_argument("--eps", type=float, default=0.01, help="g(t) param for stochastic sampling.")
    
    sampling.add_argument("--max_batch_size", type=int, default=500, help='Maximum number of systems to sample per batch.')
    sampling.add_argument("--bs_per_gbmem", type=float, default=None, help='Batch size per GB/EM on the GPU.')
    

    # --- Metrics computation options ---
    metrics = p.add_argument_group("Metrics Options")

    metrics.add_argument("--timeout", type=int, default=2700, help='Maximum running time in seconds for any eval metric.',)
    metrics.add_argument("--disable_pb_valid", action="store_true",  help='Disables PoseBusters validity check.', )    
    metrics.add_argument("--disable_gnina", action="store_true", help='Disables GNINA docking score calculation.')    
    metrics.add_argument("--disable_posecheck", action="store_true", help='Disables strain, clashes, and pocket-ligand interaction computation.')
    metrics.add_argument("--disable_rmsd", action="store_true", help='Disables RMSD computation between generated ligand and ground truth ligand.')
    metrics.add_argument("--disable_interaction_recovery", action="store_true", help='Disables analysis of interaction recovery by generated ligands.')
    metrics.add_argument("--disable_pharm_match", action="store_true", help='Disables computations of matching pharmacophores by generated ligands.')
    metrics.add_argument('--disable_strain', action='store_true', help='Disables strain energy calculation.')
    metrics.add_argument("--disable_ground_truth_metrics", action="store_true", help='Disables all relevant metrics on the truth ligand.')
    metrics.add_argument("--disable_fixed_frag_metrics", action="store_true", help='Disables computations of fixed fragment metrics.')

    args = p.parse_args()

    return args


def sample_system(ckpt_path: Path,
                  task: Task,
                  dataset_start_idx: int,
                  n_replicates: int,
                  n_timesteps: int,
                  dataset: str,
                  split: str,
                  max_batch_size: int,
                  dataset_name: str,
                  n_samples: int = None,
                  sys_idx_file: Path = None,
                  plinder_path: Path = None,
                  crossdocked_path: Path = None,
                  **kwargs
                  ):
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found")
    
    # 2) load the exact train‐time config
    train_cfg_path = ckpt_path.parent.parent / '.hydra' / 'config.yaml'
    train_cfg = quick_load.load_trained_model_cfg(train_cfg_path)

    # apply some changes to the config to enable sampling
    train_cfg.num_workers = 0
    if plinder_path:
        train_cfg.plinder_path = plinder_path
    if crossdocked_path:
        train_cfg.crossdocked_path = crossdocked_path

    # get device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 4) instantiate datamodule & model
    #dm  = quick_load.datamodule_from_config(train_cfg)
    #multitask_dataset = dm.load_dataset(split)
    model = quick_load.omtra_from_checkpoint(ckpt_path).to(device).eval()

    if sys_idx_file is None:
        dataset_idxs = range(dataset_start_idx, dataset_start_idx + n_samples) 
    else:
        # read in pre-determined index file
        with open(sys_idx_file, "r") as f:
            line = f.readline().strip()
            dataset_idxs = [int(i) for i in line.split(",")]

            if n_samples is not None:
                dataset_idxs = dataset_idxs[:n_samples]
            
    sys_info = None

    if dataset == 'plinder':
        plinder_link_version = task.plinder_link_version
        
        cfg = quick_load.load_cfg(overrides=['task_group=protein'], plinder_path=plinder_path)
        plinder_datamodule = datamodule_from_config(cfg)    
        dataset = plinder_datamodule.load_dataset(split).datasets['plinder'][plinder_link_version]
        
        #dataset = multitask_dataset.datasets['plinder'][plinder_link_version]
        dataset_name = 'plinder'

        # system info
        sys_info = dataset.system_lookup[dataset.system_lookup["system_idx"].isin(dataset_idxs)].loc[dataset_idxs].copy()

        sys_info.loc[:, 'sys_id'] = [f"sys_{idx}_gt" for idx in range(sys_info.shape[0])]
        sys_info['n_gt_lig_atoms'] =  sys_info['lig_atom_end'] - sys_info['lig_atom_start']
        sys_info = sys_info.loc[:, ['system_id', 'ligand_id', 'ccd', 'n_gt_lig_atoms', 'sys_id']]


    elif dataset == 'crossdocked':

        cfg = quick_load.load_cfg(overrides=['task_group=fixed_crossdocked'], crossdocked_path=crossdocked_path)
        crossdocked_datamodule = datamodule_from_config(cfg)    
        dataset = crossdocked_datamodule.load_dataset(split).datasets['crossdocked']
                               
        #dataset = multitask_dataset.datasets['crossdocked']
        dataset_name = 'crossdocked'

        # system info
        sys_info = dataset.system_lookup[dataset.system_lookup["system_idx"].isin(dataset_idxs)].copy()
        sys_info['n_gt_lig_atoms'] = sys_info['lig_atom_end'] - sys_info['lig_atom_start']
        sys_info = sys_info.loc[:, ['lig_sdf', 'rec_pdb', 'n_gt_lig_atoms']] 
        sys_info['lig_id'] = sys_info['lig_sdf'].apply(lambda x: Path(Path(x).stem).stem)

        # sort systems
        # sorted_idx = natsorted(sys_info.index, key=lambda i: sys_info.loc[i, "lig_id"])
        # sys_info = sys_info.iloc[sorted_idx].reset_index(drop=True)
        sys_info.loc[:, 'sys_id'] = [f"sys_{idx}_gt" for idx in range(sys_info.shape[0])]
        
        # sort dataset indices to match sys_info
        # dataset_idxs = list(dataset_idxs)
        # dataset_idxs = [dataset_idxs[i] for i in sorted_idx]

    elif dataset == 'pharmit':
        raise ValueError(f"Pharmit dataset does not include proteins!")
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    # get g_list
    g_list = [ dataset[(task.name, i)].to(device) for i in dataset_idxs ]

    # set coms if protein is present
    if 'protein_identity' in task.groups_present and (any(group in task.groups_present for group in ['ligand_identity', 'ligand_identity_condensed'])):
        coms = [ g.nodes['lig'].data['x_1_true'].mean(dim=0) for g in g_list ]
    else:
        coms = None
    
    # sample the model in batches
    sampled_systems = model.sample_in_batches(g_list=g_list,
                                              n_replicates=n_replicates,
                                              max_batch_size=max_batch_size,
                                              task_name=task.name,
                                              unconditional_n_atoms_dist=dataset_name,
                                              device=device,
                                              n_timesteps=n_timesteps,
                                              visualize=False,
                                              coms=coms,
                                              **kwargs,
                                              )
    

    return g_list, sampled_systems, sys_info



def write_system_pairs(g_list: List[dgl.DGLHeteroGraph],
                       sampled_systems: List[SampledSystem],
                       task: Task,
                       n_replicates: int,
                       output_dir: Path):

    write_ground_truth(
        n_systems=len(g_list),
        n_replicates=n_replicates,
        task=task,
        output_dir=output_dir,
        sampled_systems=sampled_systems,
        g_list=g_list,
        prot_cif=False
        )
    
    system_pairs = {}

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
        sys_pair = {} 

        sys_name = f"sys_{sys_id}_gt"
        sys_gt_dir = output_dir / sys_name

        gen_ligs = [s.get_rdkit_ligand() for s in replicates]
        num_fixed_atoms = [s.get_n_fixed_atoms() for s in replicates]

        for i, lig in enumerate(gen_ligs):
            lig.SetProp("_Name", f"gen_ligands_{i}")

        true_lig = replicates[0].get_gt_ligand(g=g_list[sys_id].to('cpu')) 
        true_lig.SetProp("_Name", "ground_truth")
        true_lig_file = sys_gt_dir / "ligand.sdf"

        true_prot_file = sys_gt_dir / "protein_0.pdb"
        true_prot_id = "protein_0"

        if 'pharmacophore' in task.groups_present:
            pharm = replicates[0].get_pharmacophore_from_graph(g=g_list[sys_id].to('cpu'), kind='gt')

        if 'protein_structure' in task.groups_generated:
            # pair each generated ligand to generated protein
            for i, lig in enumerate(gen_ligs):
                pair = {}

                # generated ligand 
                pair['gen_ligs'] = [lig]
                gen_lig_file = sys_gt_dir / f"gen_ligands_{i}.sdf"
                write_mols_to_sdf([lig], gen_lig_file)
                pair['gen_ligs_file'] = gen_lig_file
                pair['gen_ligs_ids'] = [f"gen_ligands_{i}"]

                # true ligand 
                pair['true_lig'] = true_lig
                pair['true_lig_file'] = true_lig_file
                pair['n_fixed_atoms'] = num_fixed_atoms[i]
                
                # generated protein
                pair["gen_prot_file"] = sys_gt_dir / f"gen_prot_{i}.pdb"
                pair["gen_prot_id"] = f"gen_prot_{i}"

                # true protein
                pair["true_prot_file"] = true_prot_file
                pair["true_prot_id"] = true_prot_id
                
                # true pharmacophores
                if 'pharmacophore' in task.groups_present:
                    pair["true_pharm"] = pharm

                sys_pair[f"pair_{i}"] = pair

            # write proteins to pdbs
            proteins = [s.get_protein_array() for s in replicates]
            write_arrays_to_pdb(proteins, sys_gt_dir, 'gen_prot')

        else:  
            pair = {}

            # pair all generated ligands to one reference protein
            pair['gen_ligs'] = gen_ligs
            gen_lig_file = sys_gt_dir / f"gen_ligands.sdf"
            write_mols_to_sdf(gen_ligs, gen_lig_file)
            pair['gen_ligs_file'] = gen_lig_file
            pair['gen_ligs_ids'] = [f"gen_ligands_{i}" for i in range(len(gen_ligs))]

            # true ligand 
            pair['true_lig'] = true_lig
            pair['true_lig_file'] = sys_gt_dir / f"ligand.sdf"
            pair['n_fixed_atoms'] = num_fixed_atoms[0]

            # set generated protein to reference protein
            pair['gen_prot_file'] = true_prot_file
            pair['gen_prot_id'] = true_prot_id

            # true protein
            pair['true_prot_file'] = true_prot_file
            pair['true_prot_id'] = true_prot_id

            # true pharmacophores
            if 'pharmacophore' in task.groups_present:
                pair["true_pharm"] = pharm
            
            sys_pair['pair_0'] = pair
        
        system_pairs[sys_name] = sys_pair

    return system_pairs


def get_device_properties_with_retry(dev_idx=0, retries=5, delay=5):
    """Get CUDA device properties with retries."""
    last_err = None
    for i in range(retries):
        try:
            return torch.cuda.get_device_properties(dev_idx)
        except Exception as e:
            last_err = e
            print(f"[WARN] CUDA init failed (attempt {i+1}/{retries}): {e}")
            time.sleep(delay)
    raise RuntimeError(f"Failed to init CUDA after {retries} retries") from last_err

def main(args):
    task_name: str = args.task
    task: Task = task_name_to_class(task_name)

    if args.samples_dir is None:

        if task.unconditional or ('protein_identity' not in task.groups_present):
            raise ValueError("Sampling mode requires a protein-conditioned task. "
                             "Use --samples_dir for metrics-only mode on protein-free tasks.")
        
        model_ckpt = Path(args.ckpt_path)
        if args.output_dir is None:
            output_dir = model_ckpt.parent.parent / f"samples_{task_name}_{args.dataset}"
        else:
            output_dir = args.output_dir 
        output_dir.mkdir(parents=True, exist_ok=True, mode=0o777)

        metrics_file = output_dir / "eval_metrics.csv"
        if metrics_file.exists() and args.exit_if_done:
            print('output file already exists, exiting')
            print(f'output file of interest: {metrics_file}')
            sys.exit()

        # Additional keyword arguments for special types of sampling
        kwargs = {'stochastic_sampling': args.stochastic_sampling,
                  'noise_scaler': args.noise_scaler,
                  'eps': args.eps,
                  'n_lig_atom_margin': args.n_lig_atom_margin,
                  'fixed_coord_max_std': args.fixed_coord_max_std,
                  'fixed_coord_std': args.fixed_coord_std,
                  'fixed_token_max_prob': args.fixed_token_max_prob,
                  'fixed_token_prob': args.fixed_token_prob}
        
        if args.bs_per_gbmem is not None:
            # gpu_mem_available = torch.cuda.get_device_properties(0).total_memory // (1024**3)  
            get_device_properties_with_retry(0)  # ensure CUDA is initialized
            free_mem, _ = torch.cuda.mem_get_info(0)
            gpu_mem_available = free_mem // (1024**3) # in GB (free memory only)
            max_batch_size = int(gpu_mem_available * args.bs_per_gbmem)
            max_batch_size = max(1, max_batch_size)  # ensure at least batch size of 1
            print(f"Setting max_batch_size to {max_batch_size} based on available GPU memory.")
        else:
            max_batch_size = args.max_batch_size

        # Get samples from checkpoint
        g_list, sampled_systems, sys_info = sample_system(ckpt_path=args.ckpt_path,
                                                          task=task,
                                                          dataset_start_idx=args.dataset_start_idx,
                                                          n_samples=args.n_samples,
                                                          sys_idx_file=args.sys_idx_file,
                                                          n_replicates=args.n_replicates,
                                                          n_timesteps=args.n_timesteps,
                                                          dataset=args.dataset,
                                                          split=args.split,
                                                          max_batch_size=max_batch_size,
                                                          dataset_name=args.dataset,
                                                          plinder_path=args.plinder_path,
                                                          crossdocked_path=args.crossdocked_path,
                                                          **kwargs)
        
        print("Finished sampling. Clearing torch GPU cache...\n")
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        if isinstance(sys_info, pd.DataFrame) and not sys_info.empty:
            sys_info.to_csv(f"{output_dir}/sys_info.csv", index=False)
        
        # write samples to output files and configure dictionary of system pairs
        system_pairs = write_system_pairs(g_list=g_list,
                                          sampled_systems=sampled_systems,
                                          task=task,
                                          n_replicates=args.n_replicates,
                                          output_dir=output_dir)
    else:
        samples_dir = args.samples_dir
        output_dir = args.output_dir or samples_dir
        output_dir.mkdir(parents=True, exist_ok=True, mode=0o777)

        metrics_file = output_dir / "eval_metrics.csv"
        if metrics_file.exists() and args.exit_if_done:
            print('output file already exists, exiting')
            print(f'output file of interest: {metrics_file}')
            sys.exit()

        if args.sys_info_file is None:
            sys_info_file =  f"{samples_dir}/sys_info.csv"
            print(f"Using default system info file: {sys_info_file}")
        else:
            sys_info_file = args.sys_info_file
        
        try:
            sys_info = pd.read_csv(sys_info_file)
        except Exception as e:  # case where we didn't generate a system info file
            print(f"Warning: Could not find system info csv at {sys_info_file}")
            sys_info = None

        system_pairs = system_pairs_from_path(samples_dir=samples_dir,
                                              task=task,
                                              n_samples=args.n_samples,
                                              sample_start_idx=args.sample_start_idx,
                                              n_replicates=args.n_replicates)
    
    if not args.sample_only:
        pb_mode = determine_pb_mode(task)

        is_partial_task = len(task.partial_modalities_fixed) > 0

        metrics_to_run = {'pb_valid': not args.disable_pb_valid,
                        'gnina': not args.disable_gnina,
                        'posecheck': not args.disable_posecheck,
                        'rmsd': not args.disable_rmsd and 'ligand_identity_condensed' not in task.groups_generated,
                        'interaction_recovery': not args.disable_interaction_recovery,
                        'pharm_match': (not args.disable_pharm_match) and ('pharmacophore' in task.groups_present),
                        'ground_truth': not args.disable_ground_truth_metrics,
                        'fixed_frag_metrics': (not args.disable_fixed_frag_metrics) and is_partial_task}

        # Auto-disable metrics that are inapplicable for this task
        from omtra.eval.metrics.compute import determine_applicable_metrics
        applicable = determine_applicable_metrics(task)
        for k in list(metrics_to_run):
            if k in applicable:
                metrics_to_run[k] = metrics_to_run[k] and applicable[k]

        protein_generated = "protein_structure" in task.groups_generated
        frag_systems = sampled_systems if args.samples_dir is None else None

        metrics = compute_metrics(system_pairs=system_pairs,
                                pb_mode=pb_mode,
                                metrics_to_run=metrics_to_run,
                                timeout=args.timeout,
                                disable_strain=args.disable_strain,
                                sampled_systems=frag_systems,
                                n_replicates=args.n_replicates,
                                protein_generated=protein_generated,
                                )

        metrics = metrics.reset_index()

        if isinstance(sys_info, pd.DataFrame) and not sys_info.empty:
            metrics = metrics.merge(sys_info, how='left', on='sys_id')  # Merge on 'sys_id'

        if args.sample_start_idx is None:
            metrics.to_csv(f"{output_dir}/eval_metrics.csv", index=False)
        else:
            metrics.to_csv(f"{output_dir}/eval_metrics_{args.sample_start_idx}.csv", index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    
