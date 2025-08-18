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

from rdkit import Chem
from rdkit.Chem import AllChem
import posebusters as pb
from biotite.interface import rdkit as bt_rdkit

#from genbench3d.metrics.strain_energy import strain_energy


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
        '--vina', 
        type=bool, 
        default=True,
        help='VINA and VINA min scores.'
    )    
    p.add_argument(
        '--strain', 
        type=bool, 
        default=True,
        help='Compute complex strain.'
    )
    p.add_argument(
        '--output_dir', 
        type=str, 
        default=None,
        help='Output directory.', 
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="plinder",
        help="Dataset to sample from (e.g. plinder)"
    )
    p.add_argument(
        '--pharmit_path', 
        type=str, 
        default=None,
        help='Path to the Pharmit dataset (optional).', 
    )
    p.add_argument(
        "--plinder_path",
        type=str,
        default=None,
        help="Path to the Plinder dataset (optional)"
    )
    return p.parse_args()


def pb_valid(
    system_pairs: Dict[str, Dict[str, Dict[str, SampledSystem]]],
    task: Task,
    metrics: pd.DataFrame,
    pb_workers=0
) -> float:


    if 'ligand_identity' in task.groups_generated:
        config = "dock"
    else:
        config = 'redock'

    for sys_id, pairs in system_pairs.items():

        for pair_id, data in pairs.items():

            gen_ligs = data['gen_ligs']
            true_lig = data['true_lig']
            prot = data['protein']

            valid_ligs = []
            valid_lig_ids = []

            try:
                Chem.SanitizeMol(prot)
                Chem.SanitizeMol(true_lig)

                if prot.GetNumAtoms() == 0:
                    print("PoseBusters valid eval: Empty receptor.")
                    continue
                if true_lig.GetNumAtoms() == 0:
                    print("PoseBusters valid eval: Empty true ligand.")
                    continue
            except Exception as e:
                print(f"PoseBusters valid eval: True ligand or protein failed to sanitize: {e}")
                continue
            
            for i, lig in enumerate(gen_ligs):
                try:
                    Chem.SanitizeMol(lig)

                    if lig.GetNumAtoms() > 0:
                        valid_ligs.append(lig)
                        valid_lig_ids.append(data['gen_ligs_ids'][i])
                    else:
                        print("PoseBusters valid eval: Empty ligand.")
                except Exception:
                    print("PoseBusters valid eval: Ligand failed to sanitize.")

            if not valid_ligs:
                    continue

            buster = pb.PoseBusters(config=config, max_workers=pb_workers)
            df_pb = buster.bust(valid_ligs, true_lig, prot)
            pb_results = df_pb[df_pb['sanitization'] == True].values.astype(bool).all(axis=1)
            
            for i, result in enumerate(pb_results):
                metrics.loc[(metrics['sys_id'] == sys_id) & (metrics['protein_id'] == data['protein_id']) & (metrics['ligand_id'] == valid_lig_ids[i]), 'pb_valid'] = result
        
    return metrics



def vina_scores(system_pairs, metrics) -> Dict[str, float]:
    
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/net/galaxy/home/koes/dkoes/local/miniconda/envs/cuda/lib/"

    for sys_id, pairs in system_pairs.items():
        for pair_id, data in pairs.items():
            vina_scores = []
            vina_min_scores = []

            vina_cmd = [
                "./gnina.1.3.2",
                "-r", data['protein_file'],
                "-l", data['gen_ligs_file'],
                "--score_only",
                "--cnn_scoring", "none",
                "--seed", "42"
                ]

            result = subprocess.run(vina_cmd, capture_output=True, text=True, env=env)

            # TODO: add additional scores from output

            if result.returncode != 0:
                print("Error running GNINA:")
                print(result.stderr)
                return None

            for line in result.stdout.splitlines():
                if line.strip().startswith("Affinity:"):
                    try:
                        vina_scores.append(float(line.strip().split()[1]))
                    except:
                        print("Failed to extract VINA score.")

            vina_min_cmd = [
                "./gnina.1.3.2",
                "-r", data['protein_file'],
                "-l", data['gen_ligs_file'],
                "--minimize", 
                "--cnn_scoring", "none",
                "--seed", "42"
                ]
            
            result = subprocess.run(vina_min_cmd, capture_output=True, text=True, env=env)

            if result.returncode != 0:
                print("Error running GNINA (minimize):")
                print(result.stderr)
                return None

            for line in result.stdout.splitlines():
                if line.strip().startswith("Affinity:"):
                    try:
                        vina_min_scores.append(float(line.strip().split()[1]))
                    except:
                        print("Failed to extract VINA min score.")

            for i, score in enumerate(vina_scores):
                metrics.loc[(metrics['sys_id'] == sys_id) & (metrics['protein_id'] == data['protein_id']) & (metrics['ligand_id'] == data['gen_ligs_ids'][i]), 'vina_score'] = score
                metrics.loc[(metrics['sys_id'] == sys_id) & (metrics['protein_id'] == data['protein_id']) & (metrics['ligand_id'] == data['gen_ligs_ids'][i]), 'vina_score_min'] = vina_min_scores[i]

    return metrics


def strain(system_pairs, metrics):
    for sys_id, pairs in system_pairs.items():
        for pair_id, data in pairs.items():
            for i, gen_lig in enumerate(data['gen_ligs']):
                gen_lig = Chem.AddHs(gen_lig)

                # Relax generated ligand using UFF/MMFF optimization
                relaxed_lig = copy.deepcopy(gen_lig)
                try:
                    AllChem.EmbedMolecule(relaxed_lig)
                    AllChem.MMFFOptimizeMolecule(relaxed_lig)
                except Exception as e:
                    print(f"Error relaxing ligand {data['gen_ligs_ids'][i]} for system {sys_id}: {e}")

                strain = strain_energy(gen_lig, relaxed_lig)
                metrics.loc[(metrics['sys_id'] == sys_id) & (metrics['protein_id'] == data['protein_id']) & (metrics['ligand_id'] == data['gen_ligs_ids'][i]), 'strain'] = strain

    return metrics

def sample_system(ckpt_path: Path,
                  task: str,
                  dataset_start_idx: int,
                  n_samples: int,
                  n_replicates: int,
                  n_timesteps: int,
                  dataset: str,
                  pharmit_path: Path = None,
                  plinder_path: Path = None):
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found")
    
    # 2) load the exact train‐time config
    train_cfg_path = ckpt_path.parent.parent / '.hydra' / 'config.yaml'
    train_cfg = quick_load.load_trained_model_cfg(train_cfg_path)

    # apply some changes to the config to enable sampling
    train_cfg.num_workers = 0
    if pharmit_path:
        train_cfg.pharmit_path = pharmit_path
    if plinder_path:
        train_cfg.plinder_path = plinder_path

    # get device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 4) instantiate datamodule & model
    dm  = quick_load.datamodule_from_config(train_cfg)
    multitask_dataset = dm.load_dataset('val')
    model = quick_load.omtra_from_checkpoint(ckpt_path).to(device).eval()


    # get raw dataset object
    if dataset == 'plinder':
        plinder_link_version = task.plinder_link_version
        dataset = multitask_dataset.datasets['plinder'][plinder_link_version]
    elif dataset == 'pharmit':
        dataset = multitask_dataset.datasets['pharmit']
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    # get g_list
    dataset_idxs = range(dataset_start_idx, dataset_start_idx + n_samples)
    g_list = [ dataset[(task.name, i)].to(device) for i in dataset_idxs ]

    # set coms if protein is present
    if 'protein_identity' in task.groups_present and (any(group in task.groups_present for group in ['ligand_identity', 'ligand_identity_condensed'])):
        coms = [ g.nodes['lig'].data['x_1_true'].mean(dim=0) for g in g_list ]
    else:
        coms = None

    sampled_systems = model.sample(
        g_list=g_list,
        n_replicates=n_replicates,
        task_name=task.name,
        unconditional_n_atoms_dist=dataset,
        device=device,
        n_timesteps=n_timesteps,
        visualize=False,
        coms=coms,
    )
    return g_list, sampled_systems



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
                pair["true_lig"] = replicates[0].get_rdkit_ref_ligand()
                pair["true_lig_file"] = sys_gt_dir / f"ligand.sdf"

                sys_pair[f"pair_{i}"] = pair

            # write proteins to pdb
            proteins = [s.get_protein_array() for s in replicates]
            write_arrays_to_pdb(proteins, sys_gt_dir, 'gen_prot')

        else:  
            pair = {}

            # pair all generated ligands to one reference protein
            pair["gen_ligs"] = gen_ligs

            gen_lig_file = sys_gt_dir / f"gen_ligands.sdf"
            write_mols_to_sdf(gen_ligs, gen_lig_file)
            pair["gen_ligs_file"] = gen_lig_file

            pair["gen_ligs_ids"] = [f"gen_ligands_{i}" for i in range(len(gen_ligs))]

            # protein stuff
            pair["protein"] = replicates[0].get_rdkit_protein()
            pair["protein_file"] = sys_gt_dir / f"protein_0.pdb"
            pair["protein_id"] = "protein_0"
            
            # true ligand stuff
            pair["true_lig"] = replicates[0].get_rdkit_ref_ligand()
            pair["true_lig_file"] = sys_gt_dir / f"ligand.sdf"

            sys_pair['pair_0'] = pair
        
        system_pairs[sys_name] = sys_pair

    return system_pairs



def main(args):
    # TODO: handle multiple replicates

    task_name: str = args.task
    task: Task = task_name_to_class(task_name)

    if task.unconditional:
        raise ValueError("Pose_evals.py is for docking evaluation, a conditional task.")
       
    if args.output_dir is None:
        output_dir = Path('.')
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    samples_dir = output_dir / 'samples' / task_name

    # Get samples from checkpoint
    g_list, sampled_systems = sample_system(ckpt_path=args.ckpt_path,
                                    task=task,
                                    dataset_start_idx=args.dataset_start_idx,
                                    n_samples=args.n_samples,
                                    n_replicates=args.n_replicates,
                                    n_timesteps=args.n_timesteps,
                                    dataset=args.dataset,
                                    pharmit_path=args.pharmit_path,
                                    plinder_path=args.plinder_path)
    
    system_pairs = write_system_pairs(g_list=g_list,
                                    sampled_systems=sampled_systems,
                                    task=task,
                                    n_replicates=args.n_replicates,
                                    output_dir=samples_dir)
            
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

    if args.pb_valid:
        metrics = pb_valid(system_pairs, task, metrics)

    if args.vina:
        metrics = vina_scores(system_pairs, metrics)

    # if args.strain:
    #     strain_energies = strain(gen_ligs)
    #     metrics['strain'] = strain_energies
    #     metrics['avg_strain'] = np.mean(strain_energies)

    metrics.to_csv()

if __name__ == "__main__":
    args = parse_args()
    main(args)
    
