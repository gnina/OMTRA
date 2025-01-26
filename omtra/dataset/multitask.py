import torch
import numpy as np
import dgl
from typing import List, Dict

from omtra.dataset.register import dataset_name_to_class
from omtra.tasks.register import task_name_to_class

class MultitaskDataSet(torch.utils.data.Dataset):

    """A dataset capable of serving up samples from multiple zarr datasets."""

    def __init__(self, split: str, tasks: List[dict], single_dataset_configs: Dict[str, dict], dataset_task_coupling: dict):
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
        dataset_names = list(single_dataset_configs.keys())
        dataset_classes = [dataset_name_to_class[dataset_name] for dataset_name in dataset_names]

        task_names = []
        task_probs = []
        for task_dict in tasks:
            task_names.append(task_dict['name'])
            task_probs.append(task_dict['probability'])
        task_classes = [task_name_to_class[task_name] for task_name in task_names]


    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


