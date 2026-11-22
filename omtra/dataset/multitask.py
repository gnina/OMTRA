import torch
import numpy as np
import dgl
from typing import List, Dict
from copy import deepcopy
from pathlib import Path
from omegaconf import DictConfig

from omtra.dataset.register import dataset_name_to_class
from omtra.tasks.register import task_name_to_class
from omtra.tasks.tasks import Task
from omtra.load.conf import TaskDatasetCoupling

class MultitaskDataSet(torch.utils.data.Dataset):

    """A dataset capable of serving up samples from multiple zarr datasets."""

    def __init__(self, split: str, 
                 td_coupling: TaskDatasetCoupling,
                 single_dataset_configs: Dict[str, dict], 
                 graph_config: DictConfig,
                 prior_config: DictConfig,
                 fake_atom_p: float = 0.0,
                 stereo_features: bool = False,
                 stereo_source: str = 'training_geometry',
                 identity_catalog_path: str = None,
                 matched_mask_path: str = None,
                 quarantine_path: str = None,
                 stereo_hide_p_all: float = 0.0,
                 stereo_hide_p_partial: float = 0.0,
                 stereo_hide_partial_rate: float = 0.5,
        ):
        """
        Describing the nature of the inputs, for now:
        
        single_dataset_configs is a dictionary.
            the purpose of this argument is to specify what datasets the model will be trained on, and to provide the necessary kwargs to construct the single-task dataset objects.
            - each key is a string that is a valid dataset name according to the dataset register (see omtra.dataset.register)
            - each value is a dictionary containing kwargs that will be unpacked upon construction of the single-task dataset objects

        """
        self.split = split
        self.graph_config = graph_config
        self.prior_config = prior_config
        self.single_dataset_configs = single_dataset_configs
        self.td_coupling = td_coupling
        self.fake_atom_p = fake_atom_p
        self.stereo_features = stereo_features
        # Training-only random hiding of requested stereo (see requested_stereo.hide_stereo).
        self.stereo_hide = (stereo_hide_p_all, stereo_hide_p_partial, stereo_hide_partial_rate) if split == 'train' else (0.0, 0.0, 0.5)
        if stereo_source not in ('training_geometry','verified_identity','matched_geometry'):
            raise ValueError(f'Unknown training stereo source: {stereo_source}')
        self.identity_catalog = None
        if stereo_features and stereo_source == 'verified_identity':
            if not identity_catalog_path:
                raise ValueError('Verified training stereo requires an immutable identity_catalog_path')
            from omtra.data.identity_catalog import IdentityCatalog
            self.identity_catalog = IdentityCatalog(identity_catalog_path)
        self.matched_masks = None
        if stereo_features and stereo_source == 'matched_geometry':
            if not matched_mask_path:
                raise ValueError('matched_geometry training stereo requires a matched_mask_path (an immutable matched-mask sidecar)')
            from omtra.data.matched_stereo import MatchedMasks
            self.matched_masks = MatchedMasks(matched_mask_path)

        self.task_space = td_coupling.task_space
        self.dataset_space = td_coupling.dataset_space

        # loop over tasks that use plinder, infer from the task, which version of plinder is needed
        plinder_link_versions = set()
        try:
            plinder_coupling_idx = self.dataset_space.index('plinder')
        except ValueError:
            plinder_coupling_idx = None
        if plinder_coupling_idx is not None:
            # get tasks that use plinder
            p_task_plinder = td_coupling.p_dataset_task[:, :, plinder_coupling_idx]
            p_task_plinder = p_task_plinder.sum(dim=0) # tensor of shape (len(task_space))
            tasks_using_plinder = torch.where(p_task_plinder > 0)[0].tolist()
            for task_idx in tasks_using_plinder:
                task_name = self.task_space[task_idx]
                task_class = task_name_to_class(task_name)
                plinder_link_versions.add(task_class.plinder_link_version)


        # initialize dataset classes
        self.datasets = {}
        dataset_classes = [dataset_name_to_class[dataset_name] for dataset_name in self.dataset_space]
        for dataset_name, dataset_class in zip(self.dataset_space, dataset_classes):
            single_dataset_config = deepcopy(self.single_dataset_configs[dataset_name])

            if dataset_name == 'plinder':
                plinder_exclude_ids = None
                if plinder_exclude_file := single_dataset_config.pop("plinder_exclude_file", None):
                    plinder_exclude_ids = set(
                        l for l in Path(plinder_exclude_file).read_text().splitlines() if l.strip()
                    )
                plinder_dataset_objects = {}
                for plinder_link_name in sorted(list(plinder_link_versions)):
                    if plinder_link_name:
                        plinder_dataset_objects[plinder_link_name] = dataset_class(
                            link_version=plinder_link_name,
                            split=self.split,
                            graph_config=self.graph_config,
                            prior_config=self.prior_config,
                            fake_atom_p=fake_atom_p,
                            exclude_system_ids=plinder_exclude_ids,
                            **single_dataset_config)
                self.datasets[dataset_name] = plinder_dataset_objects
            elif dataset_name == 'crossdocked':
                crossdocked_exclude_codes = None
                if crossdocked_exclude_file := single_dataset_config.pop("crossdocked_exclude_file", None):
                    crossdocked_exclude_codes = set(
                        l.strip().upper() for l in Path(crossdocked_exclude_file).read_text().splitlines() if l.strip()
                    )
                self.datasets[dataset_name] = dataset_class(
                    split=self.split,
                    graph_config=self.graph_config,
                    prior_config=self.prior_config,
                    fake_atom_p=fake_atom_p,
                    exclude_pdb_codes=crossdocked_exclude_codes,
                    **single_dataset_config)
            else:
                self.datasets[dataset_name] = dataset_class(
                    split=self.split,
                    graph_config=self.graph_config,
                    prior_config=self.prior_config,
                    fake_atom_p=fake_atom_p,
                    **single_dataset_config)


        if self.identity_catalog is not None:
            for name, objects in self.datasets.items():
                variants = objects.items() if name == 'plinder' else [('default',objects)]
                for variant, dataset in variants:
                    view = f'{name}/{variant}/{self.split}'
                    dataset.identity_policy = self.identity_catalog.bind(view,dataset)

        if self.matched_masks is not None:
            for name, objects in self.datasets.items():
                variants = objects.items() if name == 'plinder' else [('default',objects)]
                for variant, dataset in variants:
                    view = f'{name}/{variant}/{self.split}'
                    if view in self.matched_masks.manifest['views']:
                        dataset.identity_policy = self.matched_masks.bind(view, dataset)
                    elif name in ('plinder', 'crossdocked'):
                        raise ValueError(f'No matched masks for training view {view}')
                    # Pharmit has no independent identity source: its identity-fixed tasks get no labels (handled below).

        # Exclusions apply equally to controls and labelled arms, without requiring
        # complete source identities. Validation/test membership is not changed.
        if quarantine_path and self.split == 'train':
            from omtra.data.quarantine import MembershipQuarantine
            quarantine = MembershipQuarantine(quarantine_path)
            bound_views = set()
            for name, objects in self.datasets.items():
                variants = objects.items() if name == 'plinder' else [('default', objects)]
                for variant, dataset in variants:
                    view = f'{name}/{variant}/{self.split}'
                    policy = quarantine.bind(view, dataset)
                    if policy is not None:
                        dataset.membership_quarantine = policy
                        bound_views.add(view)
            missing = set(quarantine.manifest['views']) - bound_views
            if missing:
                raise ValueError(f'Quarantine views were not bound: {sorted(missing)}')

    def __len__(self):
        # TODO: no idea if this is whats supposed to happen, check with ian 
        pass
    
        # total_length = 0
        # for name in self.datasets: 
        #     if name == 'plinder':
        #         for plinder_dataset in self.datasets[name].values():
        #             total_length += len(plinder_dataset)
        #     else:
        #         total_length += len(self.datasets[name])
        # return total_length

    def __getitem__(self, index):
        task_idx, dataset_idx, local_idx = index
        task_name = self.task_space[task_idx]

        task = task_name_to_class(task_name)

        dataset_name = self.dataset_space[dataset_idx]
        dataset_obj = self.datasets[dataset_name]
        if dataset_name == 'plinder':
            # get the plinder link version that this task uses
            plinder_link_version = task.plinder_link_version
            dataset_obj = dataset_obj[plinder_link_version]
            
        quarantine = getattr(dataset_obj, 'membership_quarantine', None)
        if quarantine is not None:
            quarantine.check(task, local_idx)
        g: dgl.DGLHeteroGraph = dataset_obj[(task_name, local_idx)]

        if getattr(self, 'stereo_features', False):
            from omtra.data.requested_stereo import attach_geometry, clear_stereo
            policy = getattr(dataset_obj,'identity_policy',None)
            try:
                if policy is not None:
                    policy.attach(g,task,local_idx)
                elif getattr(self, 'matched_masks', None) is not None:
                    # matched_geometry: a dataset without an independent identity source (Pharmit) has nothing that could contradict its poses, and whenever there is a
                    # stereo label the model must use it: label every centre from the pose (phosphorus is never a centre, see attach_identity). Only eligible tasks get labels.
                    attach_geometry(g, task)
                else:
                    attach_geometry(g, task)
            except (ValueError, RuntimeError) as err:   # RuntimeError: e.g. RDKit CIP digraph overflow on huge ligands
                if 'quarantine' in str(err):
                    raise
                # A rare unlabelable training ligand must not kill a multi-day run: train it
                # with all stereo labels unspecified. Every occurrence is logged (grep STEREO_FALLBACK).
                import sys
                clear_stereo(g)
                print(f'STEREO_FALLBACK dataset={dataset_name} task={task_name} idx={local_idx} reason={err}', file=sys.stderr, flush=True)
            hide = getattr(self, 'stereo_hide', (0.0, 0.0, 0.5))
            if hide[0] > 0 or hide[1] > 0:
                from omtra.data.requested_stereo import hide_stereo
                hide_stereo(g, *hide)
        # Authoritative dataset indices survive batching/copying for distributed audits.
        g.nodes['lig'].data['sample_id'] = torch.tensor(
            [dataset_idx, local_idx], dtype=torch.long).expand(g.num_nodes('lig'), 2).clone()
        import os
        if os.environ.get('OMTRA_DP_AUDIT') == '1':
            import random
            import numpy as np
            worker=torch.utils.data.get_worker_info()
            metadata=torch.tensor([worker.id if worker else -1,torch.initial_seed() % (2**63)],dtype=torch.long)
            probes=torch.tensor([torch.rand(()).item(),float(np.random.random()),random.random()])
            g.nodes['lig'].data['worker_rng_metadata']=metadata.expand(g.num_nodes('lig'),2).clone()
            g.nodes['lig'].data['worker_rng_probes']=probes.expand(g.num_nodes('lig'),3).clone()

        # TODO: task specific transforms
        # TODO: do individual datasets need access to the task name? figure out once implementing __getitem__ for individual datasets

        dataset_name = self.dataset_space[dataset_idx]

        return g, task_name, dataset_name

