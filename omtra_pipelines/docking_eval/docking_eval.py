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

from rdkit import Chem
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
        default=500,
        help='Maximum number of systems to sample per batch.',
    )
    p.add_argument(
        '--timeout',
        type=int,
        default=600,
        help='Maximum running time for any eval metric.',
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
    prot_file,
    task: Task,
    pb_workers=0
) -> List[bool]:
    
    if ('ligand_identity' in task.groups_generated) or ('ligand_identity_condensed' in task.groups_generated):
        config = 'dock'
        true_lig = None
    elif true_lig is None:  # ground truth ligand case
        config = 'dock'
    else:
        config = 'redock'

    if not gen_ligs:
        return None

    buster = pb.PoseBusters(config=config, max_workers=pb_workers)
    df_pb = buster.bust(gen_ligs, true_lig, prot_file)
    df_pb.columns = [f"pb_{col}" for col in df_pb.columns]
    df_pb['pb_valid'] = df_pb[df_pb['pb_sanitization'] == True].values.astype(bool).all(axis=1)
    
    return df_pb


def gnina(prot_file, lig_file, output_sdf, env):
    scores = {'minimizedAffinity': {},
              'CNNscore': {},
              'CNNaffinity': {},
              'CNNaffinity_variance': {}}

    vina_cmd = ['./gnina.1.3.2',
                '-r', prot_file,
                '-l', lig_file,
                '--score_only',
                '-o', output_sdf,
                '--seed', '42'
                ]

    result = subprocess.run(vina_cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print("Error running GNINA:", flush=True)
        print(result.stderr)
        return None

    supplier = Chem.SDMolSupplier(output_sdf, sanitize=False, removeHs=False)

    for lig in supplier:
        if lig is None:
            continue
        lig_id = lig.GetProp('_Name')

        for name, vals in scores.items():
            vals[lig_id] = float(lig.GetProp(name))
    
    # remove temporary file
    os.remove(output_sdf)


    # minimize
    scores['vina_min'] = {}

    vina_min_cmd = [
        './gnina.1.3.2',
        '-r', prot_file,
        '-l', lig_file,
        '-o', output_sdf,
        '--minimize', 
        '--seed', '42'
        ]
    
    result = subprocess.run(vina_min_cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print("Error running GNINA (minimize):", flush=True)
        print(result.stderr)
        return None

    supplier = Chem.SDMolSupplier(output_sdf, sanitize=False, removeHs=False)

    for lig in supplier:
        if lig is None:
            continue
        lig_id = lig.GetProp('_Name')

        scores['vina_min'][lig_id] = float(lig.GetProp('minimizedAffinity'))
    
    # remove temporary file
    os.remove(output_sdf)

    return scores


def posecheck(prot_file, ligs):
    
    # initialize the PoseCheck object
    pc = PoseCheck()
    
    interaction_types = ['HBAcceptor', 'HBDonor', 'Hydrophobic', 'PiStacking']

    results = {}

    # load a protein from a PDB file (will run reduce in the background)
    pc.load_protein_from_pdb(prot_file)

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


def compute_metrics(system_pairs, 
                    task, 
                    metrics_to_run, 
                    output_dir,
                    timeout):
    
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = "/net/galaxy/home/koes/dkoes/local/miniconda/envs/cuda/lib/"

    # dataframe for metrics
    metrics = {'sys_id': [],
               'protein_id': [],
               'gen_ligand_id': []}
    
    for sys_id, pairs in system_pairs.items():
        for _, data in pairs.items():
            for lig_id in data['gen_ligs_ids']:
                metrics['sys_id'].append(sys_id)
                metrics['protein_id'].append(data['gen_prot_id'])
                metrics['gen_ligand_id'].append(lig_id)
    
    metrics = pd.DataFrame(metrics)
    
    for sys_id, pairs in system_pairs.items():
        for pair_id, data in pairs.items():

            print(f"––––––––––––––––––––––––––––––––", flush=True)
            print(f"{sys_id}, {data['gen_ligs_ids']}, {data['gen_prot_id']}", flush=True)
            print(f"––––––––––––––––––––––––––––––––", flush=True)

            valid_gen_ligs = []
            valid_gen_lig_ids = []
            
            for i, lig in enumerate(data['gen_ligs']):
                try:
                    Chem.SanitizeMol(lig)
                    valid_gen_ligs.append(lig)
                    valid_gen_lig_ids.append(data['gen_ligs_ids'][i])
                except Chem.rdchem.KekulizeException:
                    print(f"Invalid: Kekulization failed for generated ligand {i}")
                except Chem.rdchem.MolSanitizeException:
                    print(f"Invalid: General sanitization failed for generated ligand {i}")
                except Exception as e:
                    print(f"Invalid: Another error encountered during sanitization of generated ligand {i}: {e}")

            true_lig = data['true_lig']
            try:
                Chem.SanitizeMol(true_lig)
                
            except Chem.rdchem.KekulizeException:
                print(f"Invalid: Kekulization failed for true ligand")
            except Chem.rdchem.MolSanitizeException:
                print(f"Invalid: General sanitization failed for true ligand")
            except Exception as e:
                print(f"Invalid: Another error encountered during sanitization of true ligand for system {sys_id}: {e}")

            all_rows = (metrics['sys_id'] == sys_id) & (metrics['protein_id'] == data['gen_prot_id']) & metrics['gen_ligand_id'].isin(data['gen_ligs_ids'])
            valid_lig_rows = (metrics['sys_id'] == sys_id) & (metrics['protein_id'] == data['gen_prot_id']) & metrics['gen_ligand_id'].isin(valid_gen_lig_ids)

            metrics.loc[all_rows, 'RDKit_valid'] = False
            metrics.loc[valid_lig_rows, 'RDKit_valid'] = True
            
            # PoseBusters valid
            if metrics_to_run['pb_valid']:
                results = run_with_timeout(pb_valid,
                                           timeout=timeout,
                                           gen_ligs=valid_gen_ligs, 
                                           true_lig=true_lig, 
                                           prot_file=data['gen_prot_file'], 
                                           task=task)

                if results is not None:
                    metrics.loc[valid_lig_rows, results.columns] = results.to_numpy(dtype=bool)
                    metrics.loc[all_rows & ~valid_lig_rows, results.columns] = False
                
                true_results = run_with_timeout(pb_valid,
                                                timeout=timeout,
                                                gen_ligs=true_lig, 
                                                true_lig=None, 
                                                prot_file=data['gen_prot_file'], 
                                                task=task)

                if true_results is not None:
                    cols = [f"{c}_true" for c in true_results.columns]
                    metrics.loc[all_rows, cols] = true_results.to_numpy(dtype=bool)
            
            # PoseCheck
            if metrics_to_run['posecheck']:
                # generated ligand
                results = run_with_timeout(posecheck, 
                                           timeout=timeout,
                                           prot_file=data['gen_prot_file'], 
                                           ligs=valid_gen_ligs)
                
                if results is not None:
                    for name, val in results.items():
                        metrics.loc[valid_lig_rows, name] = val
                
                # ground truth ligand
                results_true = run_with_timeout(posecheck, 
                                                timeout=timeout,
                                                prot_file=data['true_prot_file'], 
                                                ligs=[true_lig])
                
                if results_true is not None:
                    for name, val in results_true.items():
                        metrics.loc[all_rows, f"{name}_true"] = val[0]

            # GNINA
            if metrics_to_run['gnina']:
                output_sdf = f"{output_dir}/gnina_output.sdf" 
                
                # generated ligand
                results = run_with_timeout(gnina, 
                                           timeout=timeout,
                                           prot_file=data['gen_prot_file'], 
                                           lig_file=data['gen_ligs_file'], 
                                           output_sdf=output_sdf,
                                           env=env)

                if results is not None:
                    for metric_name, vals in results.items():
                        for lig_id, score in vals.items():
                            row = (metrics['sys_id'] == sys_id) & (metrics['protein_id'] == data['gen_prot_id']) & (metrics['gen_ligand_id'] == lig_id)
                            metrics.loc[row, metric_name] = score

                # ground truth ligand
                results_true = run_with_timeout(gnina, 
                                                timeout=timeout,
                                                prot_file=data['true_prot_file'], 
                                                lig_file=data['true_lig_file'], 
                                                output_sdf=output_sdf, 
                                                env=env)
                
                if results_true is not None:
                    for metric_name, vals in results_true.items():
                        metrics.loc[all_rows, f"{metric_name}_true"] = vals['']
            
            # RMSD
            if metrics_to_run['rmsd']:
                for i, gen_lig in enumerate(data['gen_ligs']):
                    results = run_with_timeout(rmsd, 
                                               timeout=timeout,
                                               gen_lig=gen_lig, 
                                               true_lig=true_lig)
                    
                    if results is not None: 
                        lig_id = data['gen_ligs_ids'][i]
                        row = (metrics['sys_id'] == sys_id) & (metrics['protein_id'] == data['gen_prot_id']) & (metrics['gen_ligand_id'] == lig_id) 
                        metrics.loc[row, 'rmsd'] = results

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
    sys_info = sys_info.loc[:, ['system_id', 'ligand_id', 'ccd', 'sys_id']]

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
                                              unconditional_n_atoms_dist=dataset,
                                              device=device,
                                              n_timesteps=n_timesteps,
                                              visualize=False,
                                              coms=coms)

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
                pair['true_lig_file'] = sys_gt_dir / f"ligand.sdf"
                
                # generated protein
                pair["gen_prot_file"] = sys_gt_dir / f"gen_prot_{i}.pdb"
                pair["gen_prot_id"] = f"gen_prot_{i}"

                # true protein
                pair["true_prot_file"] = sys_gt_dir / f"protein_0.pdb"
                pair["true_prot_id"] = "protein_0"

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
            pair['gen_prot_file'] = sys_gt_dir / f"protein_0.pdb"
            pair['gen_prot_id'] = "protein_0"

            # true protein
            pair['true_prot_file'] = sys_gt_dir / f"protein_0.pdb"
            pair['true_prot_id'] = "protein_0"
            
            sys_pair['pair_0'] = pair
        
        system_pairs[sys_name] = sys_pair

    return system_pairs


def system_pairs_from_path(samples_dir: Path,
                           task: Task,
                           n_samples: int,
                           n_replicates: int):
    system_pairs = {}

    for sys_idx in range(n_samples):
        
        sys_name = f"sys_{sys_idx}_gt"
        sys_dir = samples_dir / sys_name   

        sys_pair = {}

        if 'protein_structure' in task.groups_generated:
            
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
                    true_lig_file = sys_dir / f"ligand.sdf"
                    pair['true_lig'] = Chem.SDMolSupplier(str(true_lig_file), sanitize=False, removeHs=False)[0]
                    pair['true_lig_file'] = true_lig_file
                    
                    # generated protein
                    pair["gen_prot_file"] = sys_dir / f"gen_prot_{rep_idx}.pdb"
                    pair["gen_prot_id"] = f"gen_prot_{rep_idx}"

                    # true protein
                    pair["true_prot_file"] = sys_dir / "protein_0.pdb"
                    pair["true_prot_id"] = "protein_0"

                    sys_pair[f"pair_{rep_idx}"] = pair

        else:  
            pair = {}

            # generated ligand 
            gen_lig_file = sys_dir / f"gen_ligands.sdf"
            gen_ligs = [mol for mol in Chem.SDMolSupplier(str(gen_lig_file), sanitize=False, removeHs=False) if mol is not None]
            pair['gen_ligs'] = gen_ligs
            pair['gen_ligs_file'] = gen_lig_file
            pair['gen_ligs_ids'] = [mol.GetProp("_Name") for mol in gen_ligs]

            # true ligand 
            true_lig_file = sys_dir / f"ligand.sdf"
            pair['true_lig'] = Chem.SDMolSupplier(str(true_lig_file), sanitize=False, removeHs=False)[0]
            pair['true_lig_file'] = true_lig_file
            
            # generated protein
            pair["gen_prot_file"] = sys_dir / "protein_0.pdb"
            pair["gen_prot_id"] = f"protein_0"

            # true protein
            pair["true_prot_file"] = sys_dir / "protein_0.pdb"
            pair["true_prot_id"] = "protein_0"
        
            sys_pair['pair_0'] = pair
        
        system_pairs[sys_name] = sys_pair

    return system_pairs


def main(args):
    task_name: str = args.task
    task: Task = task_name_to_class(task_name)

    if task.unconditional:
        raise ValueError("This script is for docking evaluation, a conditional task.")

    if args.max_batch_size < args.n_samples:
        raise ValueError("Maximum number of systems to sample per batch must be greater than the number of graphs")
       
    output_dir = args.output_dir or Path(omtra_root()) / 'omtra_pipelines' / 'docking_eval' / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.samples_dir is None:
        samples_dir = output_dir / 'samples' / task_name
        samples_dir.mkdir(parents=True, exist_ok=True)

        # Get samples from checkpoint
        g_list, sampled_systems, sys_info = sample_system(ckpt_path=args.ckpt_path,
                                                          task=task,
                                                          dataset_start_idx=args.dataset_start_idx,
                                                          n_samples=args.n_samples,
                                                          n_replicates=args.n_replicates,
                                                          n_timesteps=args.n_timesteps,
                                                          split=args.split,
                                                          max_batch_size=args.max_batch_size,
                                                          plinder_path=args.plinder_path)
        
        print("Finished sampling. Clearing torch GPU cache...\n")
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        sys_info.to_csv(f"{samples_dir}/{task_name}_sys_info.csv", index=False)
        
        # write samples to output files and configure dictionary of system pairs
        system_pairs = write_system_pairs(g_list=g_list,
                                          sampled_systems=sampled_systems,
                                          task=task,
                                          n_replicates=args.n_replicates,
                                          output_dir=samples_dir)
    else:
        samples_dir = args.samples_dir
        sys_info =pd.read_csv(f"{samples_dir}/{task_name}_sys_info.csv")
        system_pairs = system_pairs_from_path(samples_dir=samples_dir,
                                              task=task,
                                              n_samples=args.n_samples,
                                              n_replicates=args.n_replicates)
    
    metrics_to_run = {'pb_valid': args.pb_valid,
                      'gnina': args.gnina,
                      'posecheck': args.posecheck,
                      'rmsd': args.rmsd}
    
    metrics = compute_metrics(system_pairs=system_pairs,
                              task=task,
                              metrics_to_run=metrics_to_run,
                              output_dir=samples_dir,
                              timeout=args.timeout)        

    metrics = pd.merge(metrics, sys_info, on='sys_id', how='left')
    metrics.to_csv(f"{output_dir}/{task_name}_metrics.csv", index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    
