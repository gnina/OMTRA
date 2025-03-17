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

        # initialize dataset classes
        self.datasets = {}
        dataset_classes = [dataset_name_to_class[dataset_name] for dataset_name in self.dataset_space]
        for dataset_name, dataset_class in zip(self.dataset_space, dataset_classes):
            single_dataset_config = deepcopy(self.single_dataset_configs[dataset_name])

            self.datasets[dataset_name] = dataset_class(
                split=self.split, 
                graph_config=self.graph_config,
                prior_config=self.prior_config,
                **single_dataset_config)


    def __len__(self):
        pass

    def __getitem__(self, index):
        task_idx, dataset_idx, local_idx = index
        task_name = self.task_space[task_idx]

        task = self.task_space[task_idx]
        g: dgl.DGLHeteroGraph = self.datasets[self.dataset_space[dataset_idx]][(task_name, local_idx)]

        # TODO: task specific transforms
        # TODO: do individual datasets need access to the task name? figure out once implementing __getitem__ for individual datasets

        dataset_name = self.dataset_space[dataset_idx]

        return g, task_name, dataset_name

