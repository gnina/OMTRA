from typing import Dict, List
from pathlib import Path
import argparse

import pandas as pd
import torch
import numpy as np
import subprocess
import os
import dgl
import gc
import multiprocessing as mp
import tempfile
from collections import defaultdict
from scipy.spatial.distance import cdist
from natsort import natsorted

from rdkit import Chem
import posebusters as pb
from posebusters.modules.rmsd import check_rmsd
from posecheck import PoseCheck
from posecheck.utils.chem import remove_radicals


from omtra.utils import omtra_root
from omtra.tasks.tasks import Task
from omtra.tasks.register import task_name_to_class
from omtra.load.quick import datamodule_from_config
import omtra.load.quick as quick_load
from omtra.eval.system import SampledSystem, write_arrays_to_pdb, write_mols_to_sdf
from routines.sample import write_ground_truth, generate_sample_names, group_samples_by_system 
from omtra.data.pharmacophores import get_pharmacophores
from omtra.constants import ph_idx_to_elem



def parse_args():
    p = argparse.ArgumentParser(description='Evaluate ligand poses')

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
   
    args = p.parse_args()

    return args


def pb_valid(
    gen_ligs,
    true_lig,
    prot_file,
    task: Task,
    pb_workers=0
) -> List[bool]:
    
    if ('ligand_identity' in task.groups_generated) or ('ligand_identity_condensed' in task.groups_generated):  # de novo design
        config = 'dock'
        true_lig = None
    elif true_lig is None:  # ground truth ligand case
        config = 'dock'
    else:   # ligand conformer
        config = 'redock'

    if not gen_ligs:
        return None

    buster = pb.PoseBusters(config=config, max_workers=pb_workers)
    df_pb = buster.bust(gen_ligs, true_lig, prot_file)
    df_pb.columns = [f"pb_{col}" for col in df_pb.columns]
    df_pb['pb_valid'] = df_pb[df_pb['pb_sanitization'] == True].values.astype(bool).all(axis=1)
    
    return df_pb


def _run_gnina(lig_file, prot_file, env, minimization):

    gnina_binary = Path(omtra_root()) / "gnina.1.3.2"

    # Check if it exists and is a file
    if not gnina_binary.is_file():
        print(f"Could not find GNINA pre-built binary under path {omtra_root()}.")
        print("Please download file at https://github.com/gnina/gnina/releases/tag/v1.3.2 and try again.")

    
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp:
        output_sdf = Path(tmp.name)
    
    try:
        if not minimization:
            scores = {'minimizedAffinity': {},  # VINA score
                      'CNNscore': {},
                      'CNNaffinity': {},
                      'CNNaffinity_variance': {}}
            
            vina_cmd = ['./gnina.1.3.2',
                        '-r', prot_file,
                        '-l', lig_file,
                        '--score_only',
                        '-o', output_sdf,
                        '--seed', '42']
        else:
            scores = {'minimizedAffinity': {}}

            vina_cmd = ['./gnina.1.3.2',
                        '-r', prot_file,
                        '-l', lig_file,
                        '-o', output_sdf,
                        '--minimize', 
                        '--seed', '42']

        cmd_result = subprocess.run(vina_cmd, capture_output=True, text=True, env=env)

        if cmd_result.returncode != 0:
            print("Error running GNINA:", flush=True)
            print(cmd_result.stderr)
            return None

        supplier = Chem.SDMolSupplier(output_sdf, sanitize=False, removeHs=False)

        for lig in supplier:
            if lig is None:
                continue
            lig_id = lig.GetProp('_Name')

            for name, vals in scores.items():
                vals[lig_id] = float(lig.GetProp(name))
    
    finally:
        output_sdf.unlink(missing_ok=True)

    return scores


def gnina(lig_file, prot_file, env):
    results = {}

    vina_results = _run_gnina(lig_file, prot_file, env, minimization=False)

    if vina_results is not None:
        results.update(vina_results)

    vina_min_results =  _run_gnina(lig_file, prot_file, env, minimization=True)
    if vina_min_results is not None:
        results['vina_min'] = vina_min_results['minimizedAffinity']
    
    return results


def posecheck(ligs, prot_file, true_lig=None, true_prot_file=None, interaction_recovery=False, disable_strain=False):
    
    # initialize the PoseCheck object
    pc = PoseCheck()
    
    interaction_types = ['HBAcceptor', 'HBDonor', 'Hydrophobic', 'PiStacking']

    results = {}

    # load a protein from a PDB file (will run reduce in the background)
    pc.load_protein_from_pdb(prot_file)

    # load RDKit molecules directly
    pc.load_ligands_from_mols(ligs)

    results['clashes'] = pc.calculate_clashes()
    if not disable_strain:
        results['strain'] = pc.calculate_strain_energy()

    interactions = pc.calculate_interactions()

    n_lig_atoms = [lig.GetNumAtoms() for lig in ligs]

    for i_type in interaction_types:
        cols = [col for col in interactions.columns if col[2] == i_type]
        i_sum = interactions[cols].sum(axis=1) 
        results[i_type] = [n_interactions / n_atoms for (n_interactions, n_atoms) in zip(i_sum, n_lig_atoms)]  # number of interactions normalized by the number of ligand atoms

    if interaction_recovery:
        fingerprint_interaction_types = ['HBAcceptor', 'HBDonor', 'PiStacking', 'XBDonor', 'CationPi', 'PiCation', 'Cationic', 'Anionic']

        pc = PoseCheck()
        pc.load_protein_from_pdb(true_prot_file)
        pc.load_ligands_from_mols([true_lig])
        true_interactions = pc.calculate_interactions()
        cols = [col for col in true_interactions.columns if col[2] in fingerprint_interaction_types]
        true_interactions_filtered = true_interactions[cols]

        cols = [col for col in interactions.columns if col[2] in fingerprint_interaction_types]
        interactions_filtered = interactions[cols]

        recovery = pd.DataFrame(False,  index=interactions_filtered.index, columns=true_interactions_filtered.columns)
        common_cols = interactions_filtered.columns.intersection(true_interactions_filtered.columns)
        recovery[common_cols] = interactions_filtered[common_cols]

        results['interaction_recovery'] = (recovery.sum(axis=1) / recovery.shape[1]).to_list()

    return results


def rmsd(gen_lig, true_lig):
    res = check_rmsd(mol_pred=gen_lig, mol_true=true_lig)
    res = res.get("results", {})
    rmsd = res.get("rmsd", -1.0)
    return rmsd


def compute_pharmacophore_match(gen_ligs, true_pharm, threshold=1.0):
    results = {"perfect_pharm_match": [],
               "frac_true_pharms_matched": []}
    
    for gen_lig in gen_ligs:
        try:
            gen_coords, gen_types, _, _ = get_pharmacophores(gen_lig)
            gen_coords = np.array(gen_coords) # has shape (n_gen_pharms, 3)
            gen_types = np.array(gen_types) # has shape (n_gen_pharms)

        except Exception as e:
            print(f"Failed to get pharmacophores for generated ligand: {e}")
            results['perfect_pharm_match'].append(None)
            results['frac_true_pharms_matched'].append(None)
            continue

        true_coords = true_pharm['coords']
        true_types = true_pharm['types_idx']

        # convert to numpy arrays
        true_coords = np.array(true_coords) # has shape (n_true_pharms, 3)
        true_types = np.array(true_types) # has shape (n_true_pharms)

        d = cdist(true_coords, gen_coords)
        same_type_mask = true_types[:, None] == gen_types[None, :]

        matching_pharms = (d < threshold) & same_type_mask

        n_true_pharms = true_coords.shape[0]
        all_true_matched = matching_pharms.any(axis=1).all()

        if n_true_pharms == 0:
            n_true_pharms = 1

        results['perfect_pharm_match'].append(all_true_matched)
        results['frac_true_pharms_matched'].append(matching_pharms.any(axis=1).sum() / n_true_pharms)

    return results


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
        print(f"\n[TIMEOUT] {func.__name__} killed after {timeout}s \n", flush=True)
        return None

    result = q.get() if not q.empty() else None
    if isinstance(result, Exception):
        print(f"\n[ERROR] {func.__name__} failed: {result} \n", flush=True)
        return None
    return result


def repair_and_sanitize(lig, i):
    try:
        Chem.SanitizeMol(lig)
    except Exception as e:
        print(f"An error occurred during sanitization of ligand {i}: {e}")
        return None
    if any(atom.GetNumRadicalElectrons() > 0 for atom in lig.GetAtoms()):
        print(f"Ligand {i} has a radical")
        return None
    return lig

def compute_metrics(system_pairs: List[SampledSystem], 
                    task: Task, 
                    metrics_to_run: Dict[str, bool], 
                    timeout: int,
                    disable_strain: bool = False,
                    ):
    
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = "/net/galaxy/home/koes/dkoes/local/miniconda/envs/cuda/lib/"

    # dataframe for metrics
    rows = []
    for sys_id, pairs in system_pairs.items():
        for _, data in pairs.items():
            for lig_id in data['gen_ligs_ids']:
                rows.append({
                    'sys_id': sys_id,
                    'protein_id': data['gen_prot_id'],
                    'gen_ligand_id': lig_id
                })
    
    metrics = pd.DataFrame(rows)
    metrics.set_index(['sys_id', 'protein_id', 'gen_ligand_id'], inplace=True)

    
    for sys_id, pairs in system_pairs.items():
        for pair_id, data in pairs.items():

            print(f"––––––––––––––––––––––––––––––––", flush=True)
            print(f"{sys_id}, {data['gen_ligs_ids']}, {data['gen_prot_id']}", flush=True)
            print(f"––––––––––––––––––––––––––––––––", flush=True)

            # Sanitize generated ligands and keep track of the ones that sanitize
            valid_gen_ligs = []
            valid_gen_lig_ids = []

            for i, lig in enumerate(data['gen_ligs']):
                lig = repair_and_sanitize(lig, i)
                if lig is not None:
                    valid_gen_ligs.append(lig)
                    valid_gen_lig_ids.append(data['gen_ligs_ids'][i])
            

            # Sanitize ground truth ligand
            true_lig = data['true_lig']
            try:
                Chem.SanitizeMol(true_lig)
            except Exception as e:
                metrics_to_run['ground_truth'] = False
                print(f"An error encountered during sanitization of true ligand for system {sys_id}: {e}")
                print(f"Automatically disabling ground truth metric computation. \n")

            all_indices = pd.MultiIndex.from_product([[sys_id], [data['gen_prot_id']], data['gen_ligs_ids']], names=['sys_id', 'protein_id', 'gen_ligand_id'])
            valid_lig_indices = pd.MultiIndex.from_product([[sys_id], [data['gen_prot_id']], valid_gen_lig_ids], names=['sys_id', 'protein_id', 'gen_ligand_id'])
            
            metrics.loc[all_indices, 'RDKit_valid'] = False
            metrics.loc[valid_lig_indices, 'RDKit_valid'] = True

            # PoseBusters valid
            if metrics_to_run['pb_valid']:

                pb_results = run_with_timeout(pb_valid, 
                                              timeout=timeout, 
                                              gen_ligs=valid_gen_ligs,
                                              true_lig=true_lig, 
                                              prot_file=data['gen_prot_file'], 
                                              task=task)
                
                if pb_results is not None:
                    pb_results.index = valid_lig_indices
                    metrics.loc[valid_lig_indices, pb_results.columns] = pb_results
                else:
                    for i, gen_lig in enumerate(valid_gen_ligs):
                        pb_result_single = run_with_timeout(pb_valid, 
                                                            timeout=120, 
                                                            gen_ligs=gen_lig,
                                                            true_lig=true_lig, 
                                                            prot_file=data['gen_prot_file'], 
                                                            task=task)
                        
                        if pb_result_single is not None:
                            metrics.loc[(sys_id, data['gen_prot_id'], valid_gen_lig_ids[i]), pb_result_single.columns] = pb_result_single.iloc[0].values
                        else:
                            print(f"Could not resolve PoseBusters eval on single generated ligand {valid_gen_lig_ids[i]}\n")
                
                if metrics_to_run['ground_truth']:
                    pb_true_results = run_with_timeout(pb_valid,
                                                       timeout=120,
                                                       gen_ligs=true_lig,
                                                       true_lig=None,
                                                       prot_file=data['true_prot_file'],
                                                       task=task)
                    
                    if pb_true_results is not None:
                        pb_true_results = pd.DataFrame([pb_true_results.iloc[0].values] * len(all_indices), columns=pb_true_results.columns, index=all_indices)
                        pb_true_results.columns = [f"{col}_true" for col in pb_true_results.columns]
                        metrics.loc[all_indices, pb_true_results.columns] = pb_true_results

            # PoseCheck
            if metrics_to_run['posecheck']:

                if sys_id == 'sys_227_gt':  # TODO: remove this after posebusters processing
                    continue

                # generated ligand
                posechk_results = run_with_timeout(posecheck, 
                                                timeout=timeout,
                                                ligs=valid_gen_ligs,
                                                prot_file=data['gen_prot_file'],
                                                true_lig=true_lig,
                                                true_prot_file=data['true_prot_file'],
                                                interaction_recovery=metrics_to_run['interaction_recovery'],
                                                disable_strain=disable_strain)
                
                if posechk_results is not None:
                    posechk_results = pd.DataFrame(posechk_results, index=valid_lig_indices)
                    metrics.loc[valid_lig_indices, posechk_results.columns] = posechk_results
                else:
                    for i, gen_lig in enumerate(valid_gen_ligs):
                        posechk_result_single = run_with_timeout(posecheck, 
                                                            timeout=120,
                                                            ligs=[gen_lig],
                                                            prot_file=data['gen_prot_file'],
                                                            true_lig=true_lig,
                                                            true_prot_file=data['true_prot_file'],
                                                            interaction_recovery=metrics_to_run['interaction_recovery'],
                                                            disable_strain=disable_strain
                                                                )

                        if posechk_result_single is not None:
                            metrics.loc[(sys_id, data['gen_prot_id'], valid_gen_lig_ids[i]), list(posechk_result_single.keys())] = pd.Series({k: v[0] for k, v in posechk_result_single.items()})
                        else:
                            print(f"Could not resolve PoseCheck eval on single generated ligand {valid_gen_lig_ids[i]}\n")  

                # ground truth ligand
                if metrics_to_run['ground_truth']:
                    posechk_true_results = run_with_timeout(posecheck,
                                                            timeout=120,
                                                            ligs=[true_lig],
                                                            prot_file=data['true_prot_file'],
                                                            disable_strain=disable_strain)
                    
                    if posechk_true_results is not None:
                        flat_row = {k: v[0] for k, v in posechk_true_results.items()}
                        posechk_true_results = pd.DataFrame([flat_row]*len(all_indices), index=all_indices)
                        posechk_true_results.columns =  [f"{col}_true" for col in posechk_true_results.keys()]
                        metrics.loc[all_indices, posechk_true_results.columns] = posechk_true_results
        
            # GNINA
            if metrics_to_run['gnina']:
                # generated ligand
                gnina_results = run_with_timeout(gnina,
                                                 timeout=timeout,
                                                 lig_file=data['gen_ligs_file'],
                                                 prot_file=data['gen_prot_file'], 
                                                 env=env)

                if gnina_results is not None:
                    gnina_results = pd.DataFrame(gnina_results)
                    gnina_results.index = pd.MultiIndex.from_product([[sys_id], [data['gen_prot_id']], gnina_results.index], names=['sys_id', 'protein_id', 'gen_ligand_id'])
                    metrics.loc[gnina_results.index, gnina_results.columns] = gnina_results
                
                #  ground truth ligand
                if metrics_to_run['ground_truth']:
                    gnina_true_results = run_with_timeout(gnina,
                                                          timeout=timeout,
                                                          lig_file=data['true_lig_file'],
                                                          prot_file=data['true_prot_file'], 
                                                          env=env)
                    
                    if gnina_true_results is not None:
                        flat_row = {k: v[true_lig.GetProp('_Name')] for k, v in gnina_true_results.items()}
                        gnina_true_results = pd.DataFrame([flat_row]*len(all_indices), index=all_indices)
                        gnina_true_results.columns = [f"{col}_true" for col in gnina_true_results.columns]
                        metrics.loc[all_indices, gnina_true_results.columns] = gnina_true_results
            
            # RMSD
            if metrics_to_run['rmsd']:
                for i, gen_lig in enumerate(data['gen_ligs']):
                    rmsd_results = None
                    rmsd_results = run_with_timeout(rmsd,
                                                    timeout=120,
                                                    gen_lig=gen_lig,
                                                    true_lig=true_lig)
                
                    if rmsd_results is not None: 
                        metrics.loc[(sys_id, data['gen_prot_id'], data['gen_ligs_ids'][i]), 'rmsd'] = rmsd_results
            
            # pharmacophroe matching
            if metrics_to_run['pharm_match']:
                pharm_results = run_with_timeout(compute_pharmacophore_match,
                                                 timeout=timeout,
                                                 gen_ligs=valid_gen_ligs,   # TODO: will this work if we pass all ligands not just the RDKit valid ones?
                                                 true_pharm=data['true_pharm'])
                
                pharm_results = pd.DataFrame(pharm_results, index=valid_lig_indices)
                metrics.loc[valid_lig_indices, pharm_results.columns] = pharm_results
            
    return metrics


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
        sorted_idx = natsorted(sys_info.index, key=lambda i: sys_info.loc[i, "lig_id"])
        sys_info = sys_info.iloc[sorted_idx].reset_index(drop=True)
        sys_info.loc[:, 'sys_id'] = [f"sys_{idx}_gt" for idx in range(sys_info.shape[0])]
        
        # sort dataset indices to match sys_info
        dataset_idxs = list(dataset_idxs)
        dataset_idxs = [dataset_idxs[i] for i in sorted_idx]

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


def system_pairs_from_path(samples_dir: Path,
                           task: Task,
                           n_samples: int,
                           sample_start_idx: int,
                           n_replicates: int):
    system_pairs = {}

    if sample_start_idx is None:
        sample_start_idx = 0

    for sys_idx in range(sample_start_idx, sample_start_idx+n_samples):
        
        sys_name = f"sys_{sys_idx}_gt"
        sys_dir = samples_dir / sys_name   

        if not os.path.isdir(sys_dir):
            print(f"WARNING: Missing directory for system {sys_idx}. Skipping this system.")
            continue

        sys_pair = {}

        true_lig_file = sys_dir / "ligand.sdf"

        if not os.path.exists(true_lig_file):
            print(f"WARNING: Missing ground truth ligand file for system {sys_idx}. Depending on downstream metrics this may cause pipeline failures.")

        true_lig = Chem.SDMolSupplier(str(true_lig_file), sanitize=False, removeHs=False)[0]

        true_prot_file = sys_dir / "protein_0.pdb"
        true_prot_id = 'protein_0'

        if 'pharmacophore' in task.groups_present:
            pharm = {}
            true_pharm_file = sys_dir / "pharmacophore.xyz"

            if not os.path.exists(true_pharm_file):
                print(f"WARNING: Missing pharmacophore file for system {sys_idx}. Depending on downstream metrics this may cause pipeline failures.")
                pharm_missing = True

            else:
                pharm_data = np.loadtxt(true_pharm_file, skiprows=1, dtype=str)

                if pharm_data.ndim == 1:
                    pharm_data = pharm_data.reshape(1, -1)

                pharm['types_idx'] = [ph_idx_to_elem.index(p) for p in pharm_data[:, 0].tolist()]
                pharm['coords'] = pharm_data[:, 1:].astype(float)


        if 'protein_structure' in task.groups_generated:   # Flexible protein tasks
            
            for rep_idx in range(n_replicates):
                pair = {}
                gen_lig_file = sys_dir / f"gen_ligands_{rep_idx}.sdf"
                
                if os.path.exists(gen_lig_file):

                    # generated ligand 
                    gen_ligs = [mol for mol in Chem.SDMolSupplier(str(gen_lig_file), sanitize=False, removeHs=False) if mol is not None]
                    pair['gen_ligs'] = gen_ligs
                    pair['gen_ligs_file'] = gen_lig_file
                    pair['gen_ligs_ids'] = [mol.GetProp("_Name") for mol in gen_ligs]

                    # true ligand 
                    pair['true_lig'] = true_lig
                    pair['true_lig_file'] = true_lig_file
                    
                    # generated protein
                    pair["gen_prot_file"] = sys_dir / f"gen_prot_{rep_idx}.pdb"
                    pair["gen_prot_id"] = f"gen_prot_{rep_idx}"

                    # true protein
                    pair["true_prot_file"] = true_prot_file
                    pair["true_prot_id"] = true_prot_id

                    if not os.path.exists(true_prot_file):
                        print(f"WARNING: Missing true protein file for system {sys_idx}. Depending on downstream metrics this may cause pipeline failures.")

                    # true pharmacophores
                    if 'pharmacophore' in task.groups_present and not pharm_missing:
                        pair["true_pharm"] = pharm

                    sys_pair[f"pair_{rep_idx}"] = pair
                else:
                    print(f"WARNING: Missing file for generated ligand {rep_idx} for system {sys_idx}.")

        else:   # rigid protein tasks
            pair = {}

            # generated ligand 
            gen_lig_file = sys_dir / f"gen_ligands.sdf"

            if not os.path.exists(gen_lig_file):
                print(f"WARNING: Missing generated ligands file for system {sys_idx}. Skipping this system.")
                continue

            gen_ligs = [mol for mol in Chem.SDMolSupplier(str(gen_lig_file), sanitize=False, removeHs=False) if mol is not None]
            pair['gen_ligs'] = gen_ligs
            pair['gen_ligs_file'] = gen_lig_file
            pair['gen_ligs_ids'] = [mol.GetProp("_Name") for mol in gen_ligs]

            # true ligand 
            pair['true_lig'] = true_lig
            pair['true_lig_file'] = true_lig_file


            if not os.path.exists(true_prot_file):
                print(f"WARNING: Missing true protein file for system {sys_idx}. Skipping this system.")
                continue
            
            # generated protein
            pair["gen_prot_file"] = true_prot_file
            pair["gen_prot_id"] = true_prot_id

            # true protein
            # TODO: change back for non-reference analysis
            pair["true_prot_file"] = true_prot_file
            pair["true_prot_id"] =  true_prot_id


            # true pharmacophores
            if 'pharmacophore' in task.groups_present and not pharm_missing:
                pair["true_pharm"] = pharm

            sys_pair['pair_0'] = pair
        
        system_pairs[sys_name] = sys_pair

    return system_pairs

def main(args):
    task_name: str = args.task
    task: Task = task_name_to_class(task_name)

    if task.unconditional or ('protein_identity' not in task.groups_present):
        raise ValueError("This script is for evaluating models on protein-conditioned tasks.")

    if args.samples_dir is None:
        
        model_ckpt = Path(args.ckpt_path)
        if args.output_dir is None:
            output_dir = model_ckpt.parent.parent / f"samples_{task_name}_{args.dataset}"
        else:
            output_dir = args.output_dir 
        output_dir.mkdir(parents=True, exist_ok=True, mode=0o777)

        # Additional keyword arguments for special types of sampling
        kwargs = {'stochastic_sampling': args.stochastic_sampling,
                  'noise_scaler': args.noise_scaler,
                  'eps': args.eps,
                  'n_lig_atom_margin': args.n_lig_atom_margin}
        
        if args.bs_per_gbmem is not None:
            gpu_mem_available = torch.cuda.get_device_properties(0).total_memory // (1024**3)  # in GB
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
        metrics_to_run = {'pb_valid': not args.disable_pb_valid,
                        'gnina': not args.disable_gnina,
                        'posecheck': not args.disable_posecheck,
                        'rmsd': not args.disable_rmsd and 'ligand_identity_condensed' not in task.groups_generated,
                        'interaction_recovery': not args.disable_interaction_recovery,
                        'pharm_match': (not args.disable_pharm_match) and ('pharmacophore' in task.groups_present),
                        'ground_truth': not args.disable_ground_truth_metrics}
        
        metrics = compute_metrics(system_pairs=system_pairs,
                                task=task,
                                metrics_to_run=metrics_to_run,
                                timeout=args.timeout,
                                disable_strain=args.disable_strain
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
    
