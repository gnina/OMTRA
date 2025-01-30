import torch
import numpy as np
import dgl
from typing import List, Dict
from copy import deepcopy

from omtra.dataset.register import dataset_name_to_class
from omtra.tasks.register import task_name_to_class
from omtra.tasks.tasks import Task

class MultitaskDataSet(torch.utils.data.Dataset):

    """A dataset capable of serving up samples from multiple zarr datasets."""

    def __init__(self, split: str, task_inputs: List[dict], single_dataset_configs: Dict[str, dict], dataset_task_coupling: dict):
        """
        Describing the nature of the inputs, for now:


        tasks is a list of dictionaries. 
            the purpose of this argument is to specify what tasks the model will be trained on, and the margial probability of the model being trained on each task, p(task)
            - each dictionary has two keys: 'name' and 'proability'
                - the value under 'name' must be a string that is a valid task name according to the task register (see omtra.tasks.register)
                - the value under 'probability' must be a positive float indicating the probability of training on this task
                    - if the sum of all probabilities is not 1, they will be normalized by their sum
        
        single_dataset_configs is a dictionary.
            the purpose of this argument is to specify what datasets the model will be trained on, and to provide the necessary kwargs to construct the single-task dataset objects.
            - each key is a string that is a valid dataset name according to the dataset register (see omtra.dataset.register)
            - each value is a dictionary containing kwargs that will be unpacked upon construction of the single-task dataset objects

        dataset_task_coupling is a dictionary.
            the purpose of this argument is to specify the probability of using specific datasets for a given task, p(dataset|task)
            - the keys are task names
            - these task names must be valid according to the task register (see omtra.tasks.register) AND appear in the 'tasks' list
            - the values must be a list of tuples
                - each tuple is of length 2. the first element is a dataset name, the second element is a float
                - dataset names must correspond to those in single_dataset_configs
                - if a dataset is specified in single_dataset_configs but not included in p(dataset|task) here, then we will assume p(dataset|task) = 0

        """

        # get the names of the datasets we'll be using
        self.dataset_names = list(single_dataset_configs.keys())

        # retrieve the tasks we need and their marginal probabilities p(task)
        self.task_names = []
        p_task = []
        for task_dict in task_inputs:
            self.task_names.append(task_dict['name'])
            p_task.append(task_dict['probability'])
        
        p_task = torch.tensor(p_task)
        p_task = p_task / p_task.sum()
        self.tasks: List[Task] = [task_name_to_class[task_name] for task_name in self.task_names]

        assert set(dataset_task_coupling.keys()) == set(self.task_names), "The keys of dataset_task_coupling must be the same as the task names in tasks"

        # construct p(dataset, task)
        p_dataset_task = torch.zeros(len(self.task_names), len(self.dataset_names), dtype=torch.float32)
        for task_idx, task_name in enumerate(self.task_names):
            for dataset_name, p in dataset_task_coupling[task_name]:
                dataset_idx = self.dataset_names.index(dataset_name)
                p_dataset_task[task_idx, dataset_idx] = p

        # normalize by row sum
        p_dataset_task = p_dataset_task / p_dataset_task.sum(dim=1, keepdim=True)

        # multiply by p(task) to get p(dataset, task)
        p_dataset_task = p_dataset_task * p_task.unsqueeze(1)
        p_dataset_task = p_dataset_task / p_dataset_task.sum() # just make sure it sums to 1
        self.p_dataset_task = p_dataset_task


        # initialize dataset classes
        dataset_classes = [dataset_name_to_class[dataset_name] for dataset_name in self.dataset_names]
        for dataset_name, dataset_class in zip(self.dataset_names, dataset_classes):
            single_dataset_config = deepcopy(single_dataset_configs[dataset_name])

            # this is super-duper clunky but we have to do it for now
            # if we are using the plinder dataset and we are doing a mixture of tasks that use and dont use the apo state
            # then we're going to need two separate chunk trackers in the sampler class, and as a result we need to double the cache size

            # get the tasks associated with this dataset
            if dataset_name == 'plinder':
                task_idxs_for_this_dataset = p_dataset_task[:, self.dataset_names.index(dataset_name)].nonzero(as_tuple=True)[0]
                tasks_for_this_dataset_ = [ self.tasks[task_idx] for task_idx in task_idxs_for_this_dataset ]
                task_uses_apo = [task.uses_apo for task in tasks_for_this_dataset_] 
                has_tasks_using_apo = any(task_uses_apo)
                has_tasks_not_using_apo = not all(task_uses_apo)
                if has_tasks_using_apo and has_tasks_not_using_apo:
                    single_dataset_config['n_chunks_cache'] = 7

        self.datasets = {
            dataset_name: dataset_class(
                split=split, **single_dataset_configs[dataset_name]) 
                for dataset_name, dataset_class in zip(self.dataset_names, dataset_classes)
        }


    def __len__(self):
        pass

    def __getitem__(self, index):
        
        task_idx, dataset_idx, local_idx = index
        task_name = self.task_names[task_idx]

        task = self.task_names[task_idx]
        g: dgl.DGLHeteroGraph = self.datasets[self.dataset_names[dataset_idx]][(task_idx, local_idx)]

        # TODO: task specific transforms
        # TODO: do individual datasets need access to the task name? figure out once implementing __getitem__ for individual datasets

        return g, task_name

