from typing import Dict, List
from pathlib import Path
import argparse

import pandas as pd
import torch
import subprocess
import os
import dgl
import gc
import multiprocessing as mp
import pickle

from rdkit import Chem
import posebusters as pb
from posebusters.modules.rmsd import check_rmsd

from omtra.utils import omtra_root
from omtra.tasks.tasks import Task
from omtra.tasks.register import task_name_to_class
import omtra.load.quick as quick_load
from omtra.eval.system import SampledSystem, write_arrays_to_pdb, write_mols_to_sdf
from routines.sample import write_ground_truth, generate_sample_names, group_samples_by_system 
from omtra.eval.evals import *



def parse_args():
    p = argparse.ArgumentParser(description='Evaluate ligand poses')

    p.add_argument(
        '--ckpt_path',
        type=Path,
        default=None,
        help='Path to model checkpoint.',
    )
    p.add_argument(
        '--samples_dir',
        type=Path,
        default=None,
        help='Path to samples. Use existing samples, do not sample a model',
    )
    p.add_argument(
        '--task',
        type=str,
        help='Task to sample for (e.g. denovo_ligand).',
        required=True
    )
    p.add_argument(
        '--n_samples',
        type=int,
        default=10,
        help='Number of samples to evaluate.',
    )
    p.add_argument(
        "--n_replicates",
        type=int,
        default=1,
        help="Number of replicates per input sample."
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
        '--output_dir', 
        type=Path, 
        default=None,
        help='Output directory.', 
    )
    p.add_argument(
        '--split',
        type=Path,
        default="val",
        help='Data split (i.e., train, val).',
    )
    p.add_argument(
        '--max_batch_size',
        type=int,
        default=600,
        help='Maximum number of systems to sample per batch.',
    )
    p.add_argument(
        '--timeout',
        type=int,
        default=600,
        help='Maximum running time for any eval metric.',
    )
    p.add_argument(
        "--stochastic_sampling",
        type=bool,
        default=None,
        help="Perform stochastic sampling?."
    )
    p.add_argument(
        "--noise_scaler",
        type=float,
        default=1.0,
        help="Noise scaling param for stochastic sampling."
    )
    p.add_argument(
        "--eps",
        type=float,
        default=0.01,
        help="g(t) param for stochastic sampling."
    )
    p.add_argument(
        "--pharmit_path",
        type=str,
        default=None,
        help="Path to the Pharmit dataset (optional)."
    )
    return p.parse_args()


def pb_valid(
    lig,
    pb_workers=0
) -> List[bool]:

    buster = pb.PoseBusters(config="mol", max_workers=pb_workers)
    df_pb = buster.bust(lig, None, None)
    df_pb.columns = [f"pb_{col}" for col in df_pb.columns]
    df_pb['pb_valid'] = df_pb[df_pb['pb_sanitization'] == True].values.astype(bool).all(axis=1)
    
    return df_pb


def rmsd(gen_lig, true_lig):
    res = check_rmsd(mol_pred=gen_lig, mol_true=true_lig)
    res = res.get("results", {})
    rmsd = res.get("rmsd", -1.0)
    return rmsd

def run_with_timeout(func, *args, timeout, **kwargs):
    def target(q, *a, **k):
        try:
            res = func(*a, **k)
            q.put(res)
        except Exception as e:
            q.put(e)

    q = mp.Queue()
    p = mp.Process(target=target, args=(q, *args), kwargs=kwargs)
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        print(f"[TIMEOUT] {func.__name__} killed after {timeout}s \n", flush=True)
        return None

    result = q.get() if not q.empty() else None
    if isinstance(result, Exception):
        print(f"[ERROR] {func.__name__} failed: {result} \n", flush=True)
        return None
    return result


def compute_metrics(sampled_systems, 
                    task_name, 
                    ):
    metrics = {}

    if 'denovo_ligand' in task_name:
        metrics.update(validity(sampled_systems, {}))
    elif 'ligand_conformer' in task_name:
        metrics.update(ligand_rmsd(sampled_systems, {'rmsd_threshold':2.0}))

    metrics.update(stability(sampled_systems, {}))
    metrics.update(check_reos_and_rings(sampled_systems, {}))
    metrics.update(geometry(sampled_systems, {'ignore_hydrogens':True}))
    metrics.update(flatness(sampled_systems, {'threshold_flatness':0.1}))
    metrics.update(pb_valid_unconditional(sampled_systems, {'max_workers':0}))

    return metrics


def sample_system(ckpt_path: Path,
                  task: Task,
                  dataset_start_idx: int,
                  n_samples: int,
                  n_replicates: int,
                  n_timesteps: int,
                  split: str,
                  max_batch_size: int,
                  stochastic_sampling,
                  noise_scaler,
                  eps,
                  pharmit_path: Path = None,
                  ):
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found")
    
    # 2) load the exact train‐time config
    train_cfg_path = ckpt_path.parent.parent / '.hydra' / 'config.yaml'
    train_cfg = quick_load.load_trained_model_cfg(train_cfg_path)

    # apply some changes to the config to enable sampling
    train_cfg.num_workers = 0
    if pharmit_path:
        train_cfg.pharmit_path = pharmit_path

    # get device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 4) instantiate datamodule & model
    dm  = quick_load.datamodule_from_config(train_cfg)
    multitask_dataset = dm.load_dataset(split)
    model = quick_load.omtra_from_checkpoint(ckpt_path).to(device).eval()

    # get raw dataset object
    dataset = multitask_dataset.datasets['pharmit']

    # get g_list
    if task.unconditional:
        g_list = None
        n_replicates = n_samples
    else:
        dataset_idxs = range(dataset_start_idx, dataset_start_idx + n_samples)
        g_list = [ dataset[(task.name, i)].to(device) for i in dataset_idxs ]

    
    # sample the model in batches
    sampled_systems = model.sample_in_batches(g_list=g_list,
                                              n_replicates=n_replicates,
                                              max_batch_size=max_batch_size,
                                              task_name=task.name,
                                              unconditional_n_atoms_dist='pharmit',
                                              device=device,
                                              n_timesteps=n_timesteps,
                                              visualize=False,
                                              coms=None,
                                              stochastic_sampling=stochastic_sampling,
                                              noise_scaler=noise_scaler,
                                              eps=eps,)

    return g_list, sampled_systems

# TODO: remove 
def write_systems(g_list: List[dgl.DGLHeteroGraph],
                 sampled_systems: List[SampledSystem],
                 task: Task,
                 n_replicates: int,
                 output_dir: Path):

    if 'ligand_conformer' in task.name:
        write_ground_truth(
            n_systems=len(g_list),
            n_replicates=n_replicates,
            task=task,
            output_dir=output_dir,
            sampled_systems=sampled_systems,
            g_list=g_list,
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

            sys_name = f"sys_{sys_id}_gt"
            sys_gt_dir = output_dir / sys_name

            gen_ligs = [s.get_rdkit_ligand() for s in replicates]

            for i, lig in enumerate(gen_ligs):
                lig.SetProp("_Name", f"gen_ligands_{i}")

            # generated ligand 
            gen_lig_file = sys_gt_dir / f"gen_ligands.sdf"
            write_mols_to_sdf(gen_ligs, gen_lig_file)

    else:  
        gen_ligs = [ s.get_rdkit_ligand() for s in sampled_systems ]
        gen_lig_file = output_dir / f"gen_ligands.sdf"
        write_mols_to_sdf(gen_ligs, gen_lig_file)


def main(args):
    task_name: str = args.task
    task: Task = task_name_to_class(task_name)

    if args.max_batch_size < args.n_samples:
        raise ValueError("Maximum number of systems to sample per batch must be greater than the number of graphs")
       
    output_dir = args.output_dir or Path(omtra_root()) / 'samples' / 'uncond_stoch_sampling' / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.samples_dir is None:
        # Get samples from checkpoint
        g_list, sampled_systems = sample_system(ckpt_path=args.ckpt_path,
                                                          task=task,
                                                          dataset_start_idx=args.dataset_start_idx,
                                                          n_samples=args.n_samples,
                                                          n_replicates=args.n_replicates,
                                                          n_timesteps=args.n_timesteps,
                                                          split=args.split,
                                                          max_batch_size=args.max_batch_size,
                                                          pharmit_path=args.pharmit_path,
                                                          stochastic_sampling=args.stochastic_sampling,
                                                          noise_scaler=args.noise_scaler,
                                                          eps=args.eps,
                                                          )
        
        print("Finished sampling. Clearing torch GPU cache...\n")
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        write_systems(g_list=g_list,
                       sampled_systems=sampled_systems,
                       task=task,
                       n_replicates=args.n_replicates,
                       output_dir=output_dir)
        
    else:
        raise NotImplemented('Evaluation from a directory of samples has not been implemented yet.')

    
    metrics = compute_metrics(sampled_systems=sampled_systems,
                              task_name=task_name)  

    print(metrics, flush=True)      

    with open(output_dir / f"{task_name}_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    
