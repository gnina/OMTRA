from typing import Dict, List
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import subprocess
import os
import copy 
import dgl
import math

from rdkit import Chem
from rdkit.Chem import AllChem
import posebusters as pb
from posebusters.modules.rmsd import check_rmsd
from posecheck import PoseCheck

from omtra.utils import omtra_root
from omtra.tasks.tasks import Task
from omtra.tasks.register import task_name_to_class
import omtra.load.quick as quick_load
from omtra.eval.system import SampledSystem, write_arrays_to_pdb, write_mols_to_sdf
from routines.sample import write_ground_truth, generate_sample_names, group_samples_by_system 



def parse_args():
    p = argparse.ArgumentParser(description='Evaluate ligand poses')

    p.add_argument(
        '--ckpt_path',
        type=Path,
        default=None,
        help='Path to model checkpoint.',
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
        '--pb_valid',
        type=bool, 
        default=True,
        help='Compute fraction of ligands that are PoseBusters valid.', 
    )    
    p.add_argument(
        '--gnina', 
        type=bool, 
        default=True,
        help='VINA and VINA min scores.'
    )    
    p.add_argument(
        '--posecheck', 
        type=bool, 
        default=True,
        help='Compute strain, clashes, and interactions using PoseCheck.'
    )
    p.add_argument(
        '--rmsd', 
        type=bool, 
        default=True,
        help='Compute RMSD between generated ligand and ground truth ligand.'
    )
    p.add_argument(
        '--output_dir', 
        type=str, 
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
        default=50,
        help='Maximum number of systems to sample per batch.',
    )
    p.add_argument(
        "--plinder_path",
        type=str,
        default=None,
        help="Path to the Plinder dataset (optional)."
    )
    return p.parse_args()


def pb_valid(
    gen_ligs,
    true_lig,
    prot,
    task: Task,
    pb_workers=0
) -> List[bool]:
    
    if 'ligand_identity' in task.groups_generated:
        config = "dock"
    else:
        config = 'redock'
    
    try:
        Chem.SanitizeMol(prot)

        if prot.GetNumAtoms() == 0:
            print("PoseBusters valid eval: Empty receptor.")
            return None
        if true_lig.GetNumAtoms() == 0:
            print("PoseBusters valid eval: Empty true ligand.")
            return None
    except Exception as e:
        print(f"PoseBusters valid eval: True ligand or protein failed to sanitize: {e}")
        return None

    if not gen_ligs:
        return None

    buster = pb.PoseBusters(config=config, max_workers=pb_workers)
    df_pb = buster.bust(gen_ligs, true_lig, prot)
    df_pb['pb_valid'] = df_pb[df_pb['sanitization'] == True].values.astype(bool).all(axis=1)

    # TODO: keep entire table
    
    return df_pb


def gnina(protein_file, lig_file, env):

    scores = {'Affinity': [],
              'CNNscore': [],
              'CNNaffinity': [],
              'CNNvariance': [],
              'Intramolecular energy': [],
              'vina_min': []}

    vina_cmd = ['./gnina.1.3.2',
                '-r', protein_file,
                '-l', lig_file,
                '--score_only',
                '--seed', '42'
                ]

    result = subprocess.run(vina_cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print("Error running GNINA:")
        print(result.stderr)
        return None

    for line in result.stdout.splitlines():
        for name, score in scores.items():
            if line.strip().startswith(f"{name}: "):
                try:
                    score.append(float(line.strip(f"{name}: ").split()[0]))
                except:
                    print(f"Failed to extract {name}.")
                    score.append(None)
        
    vina_min_cmd = [
        './gnina.1.3.2',
        '-r', protein_file,
        '-l', lig_file,
        '--minimize', 
        '--seed', '42'
        ]
    
    result = subprocess.run(vina_min_cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print("Error running GNINA (minimize):")
        print(result.stderr)
        return None

    for line in result.stdout.splitlines():
        if line.strip().startswith("Affinity:"):
            try:
                scores['vina_min'].append(float(line.strip().split()[1]))
            except:
                print("GNINA: Failed to extract VINA min score.")
    return scores

def posecheck(protein_file, ligs):
    
    # Initialize the PoseCheck object
    pc = PoseCheck()
    
    interaction_types = ['HBAcceptor', 'HBDonor', 'PiStacking', 'Hydrophobic']

    results = {}

    # load a protein from a PDB file (will run reduce in the background)
    pc.load_protein_from_pdb(protein_file)

    # load RDKit molecules directly
    pc.load_ligands_from_mols(ligs)

    results['clashes'] = pc.calculate_clashes()
    results['strain'] = pc.calculate_strain_energy()

    interactions = pc.calculate_interactions()

    n_lig_atoms = [lig.GetNumAtoms() for lig in ligs]

    for i_type in interaction_types:
        cols = [col for col in interactions.columns if col[2] == i_type]
        i_sum = interactions[cols].sum(axis=1) 
        results[i_type] = [n_interactions / n_atoms for (n_interactions, n_atoms) in zip(i_sum, n_lig_atoms)]  # number of interactions normalized by the number of ligand atoms

    return results

def rmsd(gen_lig, true_lig):
    res = check_rmsd(mol_pred=gen_lig, mol_true=true_lig) # TODO: align firs? Kabsch rmsd
    res = res.get("results", {})
    rmsd = res.get("rmsd", -1.0)
    return rmsd

def compute_metrics(system_pairs, task, metrics_to_run):
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = "/net/galaxy/home/koes/dkoes/local/miniconda/envs/cuda/lib/"

    # dataframe for metrics
    metrics = {'sys_id': [],
               'protein_id': [],
               'ligand_id': []}
    
    for sys_id, pairs in system_pairs.items():
        for _, data in pairs.items():
            for lig_id in data['gen_ligs_ids']:
                metrics['sys_id'].append(sys_id)
                metrics['protein_id'].append(data['protein_id'])
                metrics['ligand_id'].append(lig_id)
    
    metrics = pd.DataFrame(metrics)
    
    for sys_id, pairs in system_pairs.items():
        for pair_id, data in pairs.items():
            rows = (metrics['sys_id'] == sys_id) & (metrics['protein_id'] == data['protein_id']) & metrics['ligand_id'].isin(data['gen_ligs_ids'])
            
            # PoseBusters valid
            if metrics_to_run['pb_valid']:
                results = pb_valid(data['gen_ligs'], data['true_lig'], data['protein'], task)
                metrics.loc[rows, results.columns] = results.to_numpy()
            
            # GNINA
            if metrics_to_run['gnina']:
                # generated ligand
                results = gnina(data['protein_file'], data['gen_ligs_file'], env)

                for name, val in results.items():
                    metrics.loc[rows, name] = val

                # ground truth ligand
                results_true = gnina(data['protein_file'], data['true_lig_file'], env)

                for name, val in results_true.items():
                    metrics.loc[rows, f"{name}_true"] = val*len(data['gen_ligs'])

            # PoseCheck
            if metrics_to_run['posecheck']:
                # generated ligand
                results = posecheck(data['protein_file'], data['gen_ligs'])
                
                for name, val in results.items():
                    metrics.loc[rows, name] = val
                
                # ground truth ligand
                results_true = posecheck(data['protein_file'], [data['true_lig']])
                
                for name, val in results_true.items():
                    metrics.loc[rows, f"{name}_true"] = val*len(data['gen_ligs'])
            
            # RMSD
            if metrics_to_run['rmsd']:
                rmsds = []
                for gen_lig in data['gen_ligs']:
                    rmsds.append(rmsd(gen_lig, data['true_lig']))
                
                metrics.loc[rows, 'rmsd'] = rmsds

    return metrics


def sample_system(ckpt_path: Path,
                  task: Task,
                  dataset_start_idx: int,
                  n_samples: int,
                  n_replicates: int,
                  n_timesteps: int,
                  split: str,
                  max_batch_size: int,
                  plinder_path: Path = None):
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found")
    
    # 2) load the exact train‐time config
    train_cfg_path = ckpt_path.parent.parent / '.hydra' / 'config.yaml'
    train_cfg = quick_load.load_trained_model_cfg(train_cfg_path)

    # apply some changes to the config to enable sampling
    train_cfg.num_workers = 0
    if plinder_path:
        train_cfg.plinder_path = plinder_path

    # get device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 4) instantiate datamodule & model
    dm  = quick_load.datamodule_from_config(train_cfg)
    multitask_dataset = dm.load_dataset(split)
    model = quick_load.omtra_from_checkpoint(ckpt_path).to(device).eval()

    # get raw dataset object
    plinder_link_version = task.plinder_link_version
    dataset = multitask_dataset.datasets['plinder'][plinder_link_version]

    # get g_list
    dataset_idxs = range(dataset_start_idx, dataset_start_idx + n_samples)
    g_list = [ dataset[(task.name, i)].to(device) for i in dataset_idxs ]

    # system info
    sys_info = dataset.system_lookup[dataset.system_lookup["system_idx"].isin(dataset_idxs)].copy()
    sys_info.loc[:, 'sys_id'] = [f"sys_{idx}_gt" for idx in sys_info['system_idx']]

    # set coms if protein is present
    if 'protein_identity' in task.groups_present and (any(group in task.groups_present for group in ['ligand_identity', 'ligand_identity_condensed'])):
        coms = [ g.nodes['lig'].data['x_1_true'].mean(dim=0) for g in g_list ]
    else:
        coms = None
    
    # sample the model in batches
    reps_per_batch = min(max_batch_size // n_samples, n_replicates)
    n_full_batches = n_replicates // reps_per_batch
    last_batch_reps = n_replicates % reps_per_batch

    sampled_systems = []
    sample_names = []

    for i in range(n_full_batches):
        sample_names += [f"sys_{sys_idx}_rep_{(i*reps_per_batch)+rep_idx}" for sys_idx in range(n_samples) for rep_idx in range(reps_per_batch)]

        sampled_systems += model.sample(g_list=g_list,
                                    n_replicates=reps_per_batch,
                                    task_name=task.name,
                                    unconditional_n_atoms_dist=dataset,
                                    device=device,
                                    n_timesteps=n_timesteps,
                                    visualize=False,
                                    coms=coms,
                                    )
        
    # last batch
    if last_batch_reps > 0:
        sample_names += [f"sys_{sys_idx}_rep_{(n_full_batches*reps_per_batch)+rep_idx}" for sys_idx in range(n_samples) for rep_idx in range(last_batch_reps)]

        sampled_systems += model.sample(g_list=g_list,
                                        n_replicates=last_batch_reps,
                                        task_name=task.name,
                                        unconditional_n_atoms_dist=dataset,
                                        device=device,
                                        n_timesteps=n_timesteps,
                                        visualize=False,
                                        coms=coms,
                                        )
     
    return g_list, sampled_systems, sys_info, sample_names



def write_system_pairs(g_list: List[dgl.DGLHeteroGraph],
                       sampled_systems: List[SampledSystem],
                       sample_names: List[str],
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

        all_gen_ligs = [s.get_rdkit_ligand() for s in replicates]
        gen_ligs = []

        # sanitize generated ligands and check that they have atoms
        for lig in all_gen_ligs:
            try:
                Chem.SanitizeMol(lig)
                
                if lig.GetNumAtoms() > 0:
                        gen_ligs.append(lig)
                else:
                    print("Empty generated ligand.")
            except Exception:
                print("Generated ligand failed to sanitize.")
        

        # sanitize true ligand
        true_lig = replicates[0].get_rdkit_ref_ligand()
        try:
            Chem.SanitizeMol(true_lig)
        except Exception:
            print("True ligand failed to sanitize.")


        if 'protein_structure' in task.groups_generated:
            # pair each generated ligand to generated protein
            
            for i, lig in enumerate(gen_ligs):
                pair = {}

                # generated ligand stuff
                pair["gen_ligs"] = [lig]

                gen_lig_file = sys_gt_dir / f"gen_ligands_{i}.sdf"
                write_mols_to_sdf([lig], gen_lig_file)
                pair["gen_ligs_file"] = gen_lig_file
                pair["gen_ligs_ids"] = [f"gen_ligands_{i}"]
                
                # protein stuff
                pair["protein"] = replicates[i].get_rdkit_protein()

                prot_file = sys_gt_dir / f"gen_prot_{i}.pdb"
                pair["protein_file"] = prot_file
                pair["protein_id"] = f"gen_prot_{i}"

                # true ligand stuff
                pair["true_lig"] = true_lig
                pair["true_lig_file"] = sys_gt_dir / f"ligand.sdf"

                sys_pair[f"pair_{i}"] = pair

            # write proteins to pdb
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

            # protein stuff
            pair['protein'] = replicates[0].get_rdkit_protein()
            pair['protein_file'] = sys_gt_dir / f"protein_0.pdb"
            pair['protein_id'] = "protein_0"
            
            # true ligand stuff
            pair['true_lig'] = true_lig
            pair['true_lig_file'] = sys_gt_dir / f"ligand.sdf"

            sys_pair['pair_0'] = pair
        
        system_pairs[sys_name] = sys_pair

    return system_pairs

def main(args):
    task_name: str = args.task
    task: Task = task_name_to_class(task_name)

    if task.unconditional:
        raise ValueError("docking_evals.py is for docking evaluation, a conditional task.")

    if args.max_batch_size < args.n_samples:
        raise ValueError("Maximum number of systems to sample per batch must be greater than the number of graphs")
       
    if args.output_dir is None:
        output_dir =  Path(omtra_root()) / 'omtra_pipelines' / 'docking_eval' / 'outputs'
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    samples_dir = output_dir / 'samples' / task_name

    # Get samples from checkpoint
    g_list, sampled_systems, sys_info, sample_names = sample_system(ckpt_path=args.ckpt_path,
                                                                    task=task,
                                                                    dataset_start_idx=args.dataset_start_idx,
                                                                    n_samples=args.n_samples,
                                                                    n_replicates=args.n_replicates,
                                                                    n_timesteps=args.n_timesteps,
                                                                    split=args.split,
                                                                    max_batch_size=args.max_batch_size,
                                                                    plinder_path=args.plinder_path)
    
    # write samples to output files and configure dictionary of system pairs
    system_pairs = write_system_pairs(g_list=g_list,
                                    sampled_systems=sampled_systems,
                                    sample_names=sample_names,
                                    task=task,
                                    n_replicates=args.n_replicates,
                                    output_dir=samples_dir)
    
    metrics_to_run = {'pb_valid': args.pb_valid,
                      'gnina': args.gnina,
                      'posecheck': args.posecheck,
                      'rmsd': args.rmsd}
    
    metrics = compute_metrics(system_pairs=system_pairs,
                              task=task,
                              metrics_to_run=metrics_to_run)
    
    sys_info.to_csv(f"{output_dir}/{task_name}_sys_info.csv", index=False)
    metrics.to_csv(f"{output_dir}/{task_name}_metrics.csv", index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    
