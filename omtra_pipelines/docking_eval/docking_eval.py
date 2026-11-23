from typing import Dict, List
from pathlib import Path
import argparse
import json
import hashlib
from rdkit import Chem
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
    sampling.add_argument("--pharmit_path", type=str, default=None, help="Path to a Pharmit data dir (the datamodule needs its stats files even for protein tasks). Default: <repo>/data/pharmit.")
    sampling.add_argument("--crossdocked_path", type=str, default=None, help="Path to the Crossdocked dataset (optional).")

    sampling.add_argument('--seed', type=int, default=None, help='Seed conditioning selection, crop, perturbations, and sampling for reproducible evaluation.')
    sampling.add_argument("--n_samples", type=int, default=None, help='Number of samples to evaluate.')
    sampling.add_argument("--n_replicates", type=int, help="Number of replicates per input sample.", required=True)
    sampling.add_argument("--n_timesteps", type=int, default=250, help="Number of integration steps to take when sampling.")
    sampling.add_argument("--n_lig_atom_margin", type=float, default=0.075, help="Margin for number of ligand atoms for de novo design if using number of ground truth ligand atoms.")
    sampling.add_argument("--com_offset_magnitude", type=float, default=0.0, help="Magnitude of random offset applied to the ground-truth ligand center of mass used to define the pocket, before sampling. Set to 0 to disable.")
    sampling.add_argument("--additional_prot_crop", type=float, default=None, help="Eval-time-only fixed pocket crop distance (Angstroms) from the ligand, applied uniformly regardless of the checkpoint's training-time crop settings. Allows a matched-pocket-size comparison across dynamic-crop and non-dynamic-crop models. Leave unset to keep the existing behavior (crop settings inherited from the checkpoint's own training config).")

    sampling.add_argument("--stochastic_sampling", action="store_true", help="If set, perform stochastic sampling.")
    sampling.add_argument('--pharm_input_noise_std', type=float, default=0.0, help='Simulate pharmacophore input errors with per-point sigma sampled from [0, value]; preserve clean evaluation references.')
    sampling.add_argument('--fixed_coord_max_std', type=float, default=0.0, help='Simulate errors in supplied partial ligand coordinates.')
    sampling.add_argument('--fixed_token_max_prob', type=float, default=0.0, help='Simulate corruption of supplied partial ligand chemical tokens.')
    sampling.add_argument('--site_ligand_json', type=Path, default=None, help="CrossDocked docking diagnostic: JSON {lig_sdf basename: {native_xyz, bench_com}} giving the receptor's native ligand; --site_apply picks whether it defines the pocket crop, the centre, or both (default off).")
    sampling.add_argument('--site_apply', choices=['pocket', 'centre', 'both'], default='both')
    sampling.add_argument('--center_source', choices=['ligand', 'pocket'], default='ligand', help='ligand uses the reference ligand COM (oracle); pocket uses the supplied pocket centroid.')
    sampling.add_argument('--sample_ligand_size', action='store_true', help='Infer de novo ligand size from the dataset prior instead of the reference ligand size.')
    sampling.add_argument('--oracle_ligand_size', dest='sample_ligand_size',action='store_false',help='Explicit reference-size diagnostic (overrides an earlier --sample_ligand_size).')
    sampling.add_argument('--partial_mode', choices=['whole_fragments', 'fragment_bernoulli_atom', 'bernoulli_atoms', 'mixed'], default=None, help="Eval-time override of how the fixed (revealed) ligand atoms are chosen for partial tasks. Default: use the checkpoint's training partial_mode. 'bernoulli_atoms' with --partial_bernoulli_p reveals each atom with that probability.")
    sampling.add_argument('--partial_bernoulli_p', type=float, default=None, help="Eval-time override of the partial-reveal Bernoulli probability. Default: the checkpoint's training value (None = random per example).")
    sampling.add_argument("--noise_scaler", type=float, default=1.0, help="Noise scaling param for stochastic sampling.")
    sampling.add_argument("--eps", type=float, default=0.01, help="g(t) param for stochastic sampling.")
    
    sampling.add_argument("--max_batch_size", type=int, default=500, help='Maximum number of systems to sample per batch.')
    sampling.add_argument("--bs_per_gbmem", type=float, default=None, help='Batch size per GB/EM on the GPU.')
    

    sampling.add_argument('--prot_input_noise_max_std',type=float,default=0.)
    sampling.add_argument('--prot_input_noise_mode',choices=['atom','residue'],default='atom')
    sampling.add_argument('--size_prior_path',type=Path,default=None)
    sampling.add_argument('--allow_oracle_stereo',action='store_true',help='Explicit legacy benchmark adapter: derive requested identity from bound reference geometry.')
    sampling.add_argument('--stereo_unknown',action='store_true',help='Stereo-unknown mode: supply ligand chemistry only (from the graph tokens, no coordinates), so every stereocentre is "unspecified".')
    sampling.add_argument('--zero_correction',action='store_true',help='Diagnostic: zero the stereo-correction projection tables of the loaded checkpoint (same weights otherwise) to test inference dependence.')
    sampling.add_argument('--stereo_guidance_w',type=float,default=None,help='Classifier-free stereo guidance weight (diagnostic): x1 = x1_unknown + w*(x1_given - x1_unknown); w=1 equals ordinary given mode. Requires requested stereo.')
    sampling.add_argument('--stereo_identity_json',type=Path,default=None,help='Per-system supplied SMILES and optional graph_to_identity atom mapping.')
    sampling.add_argument('--stereo_hide_fraction',type=float,default=None,help='Diagnostic partial-label mode: after attaching the supplied identity, hide each specified centre/E-Z bond independently with this probability (training-time hide_stereo mechanism). Hidden sets are written to stereo_hidden_centres.json in the output directory.')
    sampling.add_argument('--stereo_hide_seed',type=int,default=0)
    sampling.add_argument('--observation_cache_dir',type=Path,default=None,help='Shared immutable conditioning cache used by every model in a paired comparison.')

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
    metrics.add_argument("--disable_lddt", action="store_true", help='Disables lDDT computation.')


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
                  pharmit_path: Path = None,
                  com_offset_magnitude: float = 0.0,
                  center_source: str = "ligand",
                  seed: int = None,
                  stereo_identity_json: Path = None,
                  allow_oracle_stereo: bool = False,
                  stereo_unknown: bool = False,
                  stereo_guidance_w: float = None,
                  zero_correction: bool = False,
                  stereo_hide_fraction: float = None,
                  stereo_hide_seed: int = 0,
                  stereo_hidden_report: Path = None,
                  observation_cache_dir: Path = None,
                  additional_prot_crop: float = None,
                  site_ligand_json: Path = None,
                  site_apply: str = 'both',
                  partial_mode: str = None,
                  partial_bernoulli_p: float = None,
                  **kwargs
                  ):
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found")
    
    # 2) load the exact train‐time config
    train_cfg_path = ckpt_path.parent.parent / 'training_config.yaml'
    if not train_cfg_path.exists():
        train_cfg_path = ckpt_path.parent.parent / '.hydra' / 'config.yaml'
    train_cfg = quick_load.load_trained_model_cfg(train_cfg_path)

    # apply some changes to the config to enable sampling
    train_cfg.num_workers = 0
    if plinder_path:
        train_cfg.plinder_path = plinder_path
    if crossdocked_path:
        train_cfg.crossdocked_path = crossdocked_path

    # Propagate the trained model's actual pocket-definition / dynamic-crop settings into
    # the eval dataset config below. Without this, quick_load.load_cfg() builds a fresh
    # config from configs/config.yaml's defaults (alpha=null, beta=null -> no cropping
    # applied), so a model trained with dynamic cropping would silently be evaluated on
    # the full, uncropped pocket it never saw during training.
    def _crop_override_value(v):
        return 'null' if v is None else v
    crop_overrides = [
        f'{field}={_crop_override_value(train_cfg.get(field, None))}'
        for field in ['crop_min_distance', 'crop_max_distance', 'alpha', 'beta']
    ]
    # additional_prot_crop is eval-only (val/test in the dataset classes, never
    # set during training), so it must come from this CLI arg, not the checkpoint's training
    # config -- inheriting it from train_cfg (as the other crop fields above do) would always
    # resolve to null, since no checkpoint's training config ever sets it. Passing this flag
    # forces the same fixed pocket crop across every model regardless of its own dynamic-crop
    # training settings; leaving it unset preserves prior behavior exactly (always null).
    crop_overrides.append(f'additional_prot_crop={_crop_override_value(additional_prot_crop)}')
    if additional_prot_crop is not None:
        # A requested fixed crop must be the ONLY crop. Otherwise a dynamic-crop checkpoint's
        # inherited random 8-12A crop runs first and the fixed crop only shrinks it further,
        # so e.g. a "10A" eval would give M4-M6 random pockets <=10A while M1-M3 get exactly 10A.
        crop_overrides = [o for o in crop_overrides
                          if not o.split('=')[0] in ('crop_min_distance', 'crop_max_distance', 'alpha', 'beta')]
        crop_overrides += ['crop_min_distance=null', 'crop_max_distance=null', 'alpha=null', 'beta=null']

    # Match training's partial masking recipe; preserve compatibility with older configs.
    # (CLI --partial_mode / --partial_bernoulli_p override the checkpoint's recipe for eval-time reveal sweeps.)
    eff_partial_mode = partial_mode if partial_mode is not None else train_cfg.get('partial_mode', 'whole_fragments')
    eff_partial_p = partial_bernoulli_p if partial_bernoulli_p is not None else train_cfg.get('partial_bernoulli_p', None)
    crop_overrides += [f"partial_mode={eff_partial_mode}",
                       f"partial_bernoulli_p={_crop_override_value(eff_partial_p)}"]

    # get device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 4) instantiate datamodule & model
    #dm  = quick_load.datamodule_from_config(train_cfg)
    #multitask_dataset = dm.load_dataset(split)
    model = quick_load.omtra_from_checkpoint(ckpt_path).to(device).eval()
    if zero_correction:
        if getattr(model.vector_field, 'stereo_correction', None) is None:
            raise ValueError('--zero_correction needs a checkpoint trained with stereo_correction')
        with torch.no_grad():
            model.vector_field.stereo_correction.w_current.zero_()
            model.vector_field.stereo_correction.w_pred.zero_()

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
        
        cfg = quick_load.load_cfg(overrides=['task_group=flexible_protein_cond_a'] + crop_overrides, plinder_path=plinder_path, pharmit_path=pharmit_path)
        plinder_datamodule = datamodule_from_config(cfg)    
        dataset = plinder_datamodule.load_dataset(split).datasets['plinder'][plinder_link_version]
        
        #dataset = multitask_dataset.datasets['plinder'][plinder_link_version]
        dataset_name = 'plinder'

        # system info
        sys_info = dataset.system_lookup.iloc[list(dataset_idxs)].copy()

        sys_info.loc[:, 'sys_id'] = [f"sys_{idx}_gt" for idx in range(sys_info.shape[0])]
        sys_info['n_gt_lig_atoms'] =  sys_info['lig_atom_end'] - sys_info['lig_atom_start']
        sys_info = sys_info.loc[:, ['system_id', 'ligand_id', 'ccd', 'n_gt_lig_atoms', 'sys_id']]


    elif dataset == 'crossdocked':

        cfg = quick_load.load_cfg(overrides=['task_group=fixed_crossdocked'] + crop_overrides, crossdocked_path=crossdocked_path, pharmit_path=pharmit_path)
        crossdocked_datamodule = datamodule_from_config(cfg)    
        dataset = crossdocked_datamodule.load_dataset(split).datasets['crossdocked']
                               
        #dataset = multitask_dataset.datasets['crossdocked']
        dataset_name = 'crossdocked'

        # system info
        sys_info = dataset.system_lookup.iloc[list(dataset_idxs)].copy()
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
        cfg = quick_load.load_cfg(overrides=['task_group=no_protein_cond_a'] + crop_overrides,pharmit_path=pharmit_path)
        dataset = datamodule_from_config(cfg).load_dataset(split).datasets['pharmit']
        dataset_name = 'pharmit'
        sys_info = pd.DataFrame({'system_id':['pharmit']*len(dataset_idxs),
                                 'ligand_id':[str(i) for i in dataset_idxs],
                                 'sys_id':[f'sys_{i}_gt' for i in range(len(dataset_idxs))]})
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    site = None
    if site_ligand_json is not None:
        if dataset_name != 'crossdocked':
            raise ValueError('--site_ligand_json is only defined for the CrossDocked dataset')
        import numpy as np
        site = json.loads(Path(site_ligand_json).read_text())
        if site_apply in ('pocket', 'both'):
            dataset.site_ref_coords = {k: np.asarray(v['native_xyz'], dtype=np.float32) for k, v in site.items()}

    # get g_list
    g_list = [ dataset[(task.name, i)].to(device) for i in dataset_idxs ]

    from omtra.data.requested_stereo import eligible
    from omtra.data.inference_stereo import attach_inference_stereo
    if model.vector_field.stereo_embedder is not None and model.vector_field.stereo_enabled and eligible(task):
        identities = json.loads(Path(stereo_identity_json).read_text()) if stereo_identity_json else None
        attach_inference_stereo(g_list, task, sys_info, identities=identities,
                                allow_oracle=allow_oracle_stereo, unknown=stereo_unknown)
        if stereo_hide_fraction is not None:
            if stereo_unknown or not 0.0 < stereo_hide_fraction <= 1.0:
                raise ValueError('--stereo_hide_fraction needs given mode and a value in (0, 1]')
            from omtra.data.requested_stereo import hide_stereo
            from omtra.data.inference_stereo import system_identity_key
            report = {}
            for i, g_i in enumerate(g_list):
                generator = torch.Generator().manual_seed(int(stereo_hide_seed) * 1000003 + i)
                hide_stereo(g_i, p_all=0.0, p_partial=1.0, partial_rate=float(stereo_hide_fraction), generator=generator)
                nd_i = g_i.nodes['lig'].data
                report[str(system_identity_key(sys_info.iloc[i]))] = {
                    'hidden_atoms': nd_i['stereo_hidden'].nonzero().flatten().tolist(),
                    'visible_atoms': ((nd_i['stereo_atom'] >= 2) & (nd_i['stereo_atom'] <= 5)).nonzero().flatten().tolist(),
                    'true_centres': nd_i['stereo_centre'].nonzero().flatten().tolist()}
            if stereo_hidden_report is not None:
                Path(stereo_hidden_report).write_text(json.dumps(report))
        if stereo_guidance_w is not None:
            if stereo_unknown:
                raise ValueError('Stereo guidance needs requested (given/oracle) stereo, not --stereo_unknown')
            from omtra.data.inference_stereo import attach_guidance_alternative
            attach_guidance_alternative(g_list, task)
    model.vector_field.stereo_guidance_w = stereo_guidance_w

    # set coms if protein is present
    if 'protein_identity' in task.groups_present and (any(group in task.groups_present for group in ['ligand_identity', 'ligand_identity_condensed'])):
        # offsetting the COMs, each system in a random direction by the same magnitude
        coms = []
        for gi, g in enumerate(g_list):
            center_type = 'lig' if center_source == 'ligand' else 'prot_atom'
            com = g.nodes[center_type].data['x_1_true'].mean(dim=0)
            if site is not None:
                e = site[Path(str(sys_info.iloc[gi]['lig_sdf'])).name]
                bench = g.nodes['lig'].data['x_1_true'].mean(dim=0)
                # the JSON must live in the graph's coordinate frame and ligand-atom convention
                assert float((bench - torch.as_tensor(e['bench_com'], dtype=bench.dtype, device=bench.device)).norm()) < 0.5, \
                    f"site JSON benchmark COM does not match the graph for {sys_info.iloc[gi]['lig_sdf']}"
                if site_apply in ('centre', 'both'):
                    com = torch.as_tensor(np.asarray(e['native_xyz']), dtype=com.dtype, device=com.device).mean(dim=0)
            if com_offset_magnitude > 0:
                direction = torch.randn(3)
                direction = direction / direction.norm()
                offset = (direction * com_offset_magnitude).to(com.device)
                coms.append(com + offset)
            else:
                coms.append(com)
    else:
        coms = None

    if observation_cache_dir is not None:
        from omtra.eval.observations import prepare_shared_observation
        contract={k:kwargs.get(k,default) for k,default in (
            ('prot_input_noise_max_std',0.),('prot_input_noise_mode','atom'),
            ('pharm_input_noise_std',0.),('fixed_coord_max_std',0.),('fixed_token_max_prob',0.))}
        contract.update(center_source=center_source,com_offset_magnitude=com_offset_magnitude,
                        partial_mode=eff_partial_mode,partial_bernoulli_p=eff_partial_p,seed=seed)
        for i,(graph,idx) in enumerate(zip(g_list,dataset_idxs)):
            center,_=prepare_shared_observation(graph,task,None if coms is None else coms[i],
                observation_cache_dir,f'{dataset_name}|{split}|{idx}',contract)
            if coms is not None:coms[i]=center
        for key in ('prot_input_noise_max_std','pharm_input_noise_std','fixed_coord_max_std','fixed_token_max_prob'):
            kwargs[key]=0.
    
    # sample the model in batches
    sampled_systems = model.sample_in_batches(g_list=g_list,
                                              n_replicates=n_replicates,
                                              max_batch_size=max_batch_size,
                                              task_name=task.name,
                                              unconditional_n_atoms_dist=dataset_name,
                                              device=device,
                                              n_timesteps=n_timesteps,
                                              allow_oracle_stereo=allow_oracle_stereo,
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
            if lig is None: continue
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
                pair['initial_prot_file']=sys_gt_dir/f'initial_prot_{i}.pdb'

                # true protein
                pair["true_prot_file"] = true_prot_file
                pair["true_prot_id"] = true_prot_id
                
                # true pharmacophores
                if 'pharmacophore' in task.groups_present:
                    pair["true_pharm"] = pharm

                sys_pair[f"pair_{i}"] = pair

            # write proteins to pdbs
            proteins = [s.get_receptor_array() for s in replicates]
            write_arrays_to_pdb(proteins, sys_gt_dir, 'gen_prot')
            write_arrays_to_pdb([s.get_initial_receptor_array() for s in replicates],sys_gt_dir,'initial_prot')

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
            if 'protein_identity' in task.groups_present:
                write_arrays_to_pdb([replicates[0].get_receptor_array(observation=True)],sys_gt_dir,'input_protein')
                pair['gen_prot_file'] = sys_gt_dir / 'input_protein_0.pdb'
            else:
                pair['gen_prot_file'] = None
            pair['gen_prot_id'] = true_prot_id

            # true protein
            pair['true_prot_file'] = true_prot_file
            pair['true_prot_id'] = true_prot_id

            # true pharmacophores
            if 'pharmacophore' in task.groups_present:
                pair["true_pharm"] = pharm
            
            sys_pair['pair_0'] = pair
        
        if 'pharmacophore' in task.groups_present:
            input_pharm = replicates[0].get_pharmacophore_from_graph(kind='input')
            with (sys_gt_dir/'input_pharmacophore.json').open('x') as stream:
                json.dump({k:v.tolist() if hasattr(v,'tolist') else v for k,v in input_pharm.items()},stream)
            for pair in sys_pair.values():pair['input_pharm']=input_pharm
        if 'protein_identity' not in task.groups_present:
            for pair in sys_pair.values():
                pair['gen_prot_file']=pair['true_prot_file']=None
        observations={}
        for nt in ('prot_atom','pharm','lig','npnde'):
            for key,value in replicates[0].g.nodes[nt].data.items():
                if key.endswith('_cond') or key.endswith('_cond_std') or key.startswith('stereo_') or key in ('entity_idx','entity_atom_idx','ccd_code'):
                    observations[nt+'__'+key]=value.cpu().numpy()
        import numpy as np
        np.savez_compressed(sys_gt_dir/'conditioning.npz',**observations)
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
    import json
    _T = {'t_start': time.time()}
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

        if metrics_file.exists() or any(output_dir.glob('sys_*_gt')):
            raise FileExistsError(f'Refusing to overwrite existing evaluation artifacts: {output_dir}')
        with (output_dir/'evaluation_protocol.json').open('x') as stream:
            json.dump({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},stream,indent=2)
        # Additional keyword arguments for special types of sampling
        kwargs = {'stochastic_sampling': args.stochastic_sampling,
                  'noise_scaler': args.noise_scaler,
                  'eps': args.eps,
                  'n_lig_atom_margin': None if args.sample_ligand_size else args.n_lig_atom_margin,
                  'pharm_input_noise_std': args.pharm_input_noise_std,
                  'prot_input_noise_max_std': args.prot_input_noise_max_std,
                  'prot_input_noise_mode': args.prot_input_noise_mode,
                  'size_prior_path': args.size_prior_path,
                  'allow_oracle_stereo': args.allow_oracle_stereo,
                  'stereo_unknown': args.stereo_unknown,
                  'stereo_guidance_w': args.stereo_guidance_w,
                  'zero_correction': args.zero_correction,
                  'stereo_hide_fraction': args.stereo_hide_fraction,
                  'stereo_hide_seed': args.stereo_hide_seed,
                  'stereo_hidden_report': (output_dir / 'stereo_hidden_centres.json') if args.stereo_hide_fraction is not None else None,
                  'stereo_identity_json': args.stereo_identity_json,
                  'observation_cache_dir': args.observation_cache_dir,
                  'fixed_coord_max_std': args.fixed_coord_max_std,
                  'fixed_token_max_prob': args.fixed_token_max_prob,
                  'center_source': args.center_source,
                  'seed': args.seed}
        
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
                                                          pharmit_path=args.pharmit_path,
                                                          com_offset_magnitude=args.com_offset_magnitude,
                                                          additional_prot_crop=args.additional_prot_crop,
                                                          site_ligand_json=args.site_ligand_json,
                                                          site_apply=args.site_apply,
                                                          partial_mode=args.partial_mode,
                                                          partial_bernoulli_p=args.partial_bernoulli_p,
                                                          **kwargs)
        
        _T['t_sampled'] = time.time()
        print("Finished sampling. Clearing torch GPU cache...\n")
        if torch.cuda.is_available(): torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available(): torch.cuda.ipc_collect()

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
    
    _T['t_written'] = time.time()
    if not args.sample_only:
        pb_mode = determine_pb_mode(task)

        is_partial_task = len(task.partial_modalities_fixed) > 0
        lig_identity_generated = 'ligand_identity_condensed' in task.groups_generated
        protein_generated = "protein_structure" in task.groups_generated

        metrics_to_run = {'pb_valid': not args.disable_pb_valid,
                        'gnina': not args.disable_gnina,
                        'posecheck': not args.disable_posecheck,
                        'rmsd': not args.disable_rmsd and (not lig_identity_generated),
                        'interaction_recovery': not args.disable_interaction_recovery,
                        'pharm_match': (not args.disable_pharm_match) and ('pharmacophore' in task.groups_present),
                        'ground_truth': not args.disable_ground_truth_metrics,
                        'fixed_frag_metrics': (not args.disable_fixed_frag_metrics) and is_partial_task,
                        'lddt_lig': (not args.disable_lddt) and (not protein_generated) and (not lig_identity_generated),
                        'lddt': (not args.disable_lddt) and protein_generated and (not lig_identity_generated)}

        # Auto-disable metrics that are inapplicable for this task
        from omtra.eval.metrics.compute import determine_applicable_metrics
        applicable = determine_applicable_metrics(task)
        for k in list(metrics_to_run):
            if k in applicable:
                metrics_to_run[k] = metrics_to_run[k] and applicable[k]

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
        from omtra.eval.metrics.compute import METRIC_TIMES
        _T['t_end'] = time.time()
        t0 = _T['t_start']
        timing = {'sampling_s': _T.get('t_sampled', t0) - t0, 'write_s': _T['t_written'] - _T.get('t_sampled', t0),
                  'metrics_s': _T['t_end'] - _T['t_written'], 'total_s': _T['t_end'] - t0,
                  'per_metric_s': {k: round(v, 1) for k, v in METRIC_TIMES.items()},
                  'n_systems': len(system_pairs), 'n_poses': int(len(metrics))}
        (Path(output_dir) / 'timing.json').write_text(json.dumps(timing, indent=1))

if __name__ == "__main__":
    args = parse_args()
    main(args)
    
