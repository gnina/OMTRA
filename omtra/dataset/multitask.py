import torch
import numpy as np
import dgl
from typing import List, Dict
from copy import deepcopy
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
                plinder_dataset_objects = {}
                for plinder_link_name in sorted(list(plinder_link_versions)):
                    if plinder_link_name:
                        plinder_dataset_objects[plinder_link_name] = dataset_class(
                            link_version=plinder_link_name,
                            split=self.split, 
                            graph_config=self.graph_config,
                            prior_config=self.prior_config,
                            **single_dataset_config)
                self.datasets[dataset_name] = plinder_dataset_objects
            else:
                self.datasets[dataset_name] = dataset_class(
                    split=self.split, 
                    graph_config=self.graph_config,
                    prior_config=self.prior_config,
                    **single_dataset_config)


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
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(f"[RANK {rank}] Inside dataloader / etc.")
        task_idx, dataset_idx, local_idx = index
        task_name = self.task_space[task_idx]

        task = task_name_to_class(task_name)

        dataset_obj = self.datasets[self.dataset_space[dataset_idx]]
        if task.plinder_link_version is not None:
            # get the plinder link version that this task uses
            plinder_link_version = task.plinder_link_version
            dataset_obj = dataset_obj[plinder_link_version]
            
        g: dgl.DGLHeteroGraph = dataset_obj[(task_name, local_idx)]

        # TODO: task specific transforms
        # TODO: do individual datasets need access to the task name? figure out once implementing __getitem__ for individual datasets

        dataset_name = self.dataset_space[dataset_idx]

        return g, task_name, dataset_name

