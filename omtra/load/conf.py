from omegaconf import DictConfig, OmegaConf, open_dict
from pathlib import Path
from omtra.utils import omtra_root
from typing import List
import hydra
import torch

from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
from dataclasses import dataclass

def merge_task_spec(cfg: DictConfig) -> DictConfig:
    """
    Load the single dataset configurations for the single datasets that are specified in the task_group configuration.

    Also build the time-dependent task coupling tensor, which specifies the probability of training on each task at each phase of training.
    """
    
    # Folder where single dataset YAML files are stored.
    single_datasets_dir = Path(omtra_root()) / 'configs' / 'task_group' / 'datamodule' / 'single_datasets'
    
    # infer the datasets specified in the task group
    dataset_task_coupling = cfg.task_group.dataset_task_coupling
    datasets = set()
    for task in dataset_task_coupling:
        task_datasets = [ c[0] for c in dataset_task_coupling[task] ]
        datasets.update(task_datasets)


    with open_dict(cfg.task_group.datamodule):
        for ds_name in datasets:
            ds_path = single_datasets_dir / f"{ds_name}.yaml"
            if not ds_path.exists():
                raise ValueError(f"Invalid dataset specified: {ds_path}")
            ds_cfg = OmegaConf.load(ds_path)
            cfg.task_group.datamodule.dataset_config.single_dataset_configs[ds_name] = ds_cfg

    return cfg

def instantiate_callbacks(callbacks_cfg: DictConfig, override_dir=None) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []
    if rank_zero_only.rank != 0:
        return callbacks

    if not callbacks_cfg:
        print("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            print(f"Instantiating callback <{cb_conf._target_}>")

            # override checkpoint dir if specified
            if override_dir is not None and 'CheckpointCallback' in cb_conf._target_:
                cb_conf.dirpath = override_dir

            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


@dataclass
class TaskDatasetCoupling:
    p_dataset_task: torch.Tensor # has shape (n_phases, n_tasks, n_datasets)
    task_space: List[str]
    dataset_space: List[str]
    phase_durations: torch.Tensor

    @property
    def n_phases(self):
        return len(self.phase_durations)
    
    @property
    def support(self):
        nonzero_idxs = self.p_dataset_task.sum(dim=0).nonzero()
        return nonzero_idxs


def build_td_coupling(task_phases, dataset_task_coupling) -> TaskDatasetCoupling:
    """
    

    task_phases is a list of dictionaries. 
        the purpose of this argument is to specify what tasks the model will be trained on, and the margial probability of the model being trained on each task, p(task)
        this dictionary also defines time-dependent task mixtures; each element of the tasks list is a dictionary that tells us 
            1. the duration of the task mixture, 
            2. the tasks in the mixture
            3. the probability of training on each task in the mixture
        - each dictionary has two keys: 'duration_batches' and 'tasks'
        - 'duration_batches' is an integer that specifies the number of batches that the task mixture will be used for
        - 'tasks' is a list of dictionaries; these dictionaries have two keys: 'name' and 'probability'
            - the value under 'name' must be a string that is a valid task name according to the task register (see omtra.tasks.register)
            - the value under 'probability' must be a positive float indicating the probability of training on this task
                - if the sum of all probabilities is not 1, they will be normalized by their sum

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
    dataset_space = set()
    task_space = set()
    for task_name, dataset_probs in dataset_task_coupling.items():
        task_space.add(task_name)
        for dataset_name, _ in dataset_probs:
            dataset_space.add(dataset_name)
    task_space = list(task_space)
    dataset_space = list(dataset_space)
    
    n_phases = len(task_phases)
    phases_task_space = set()
    phase_durations = []
    p_task_per_phase = []
    for phase_idx, phase_dict in enumerate(task_phases):
        p_task_phase = {}
        phase_durations.append(phase_dict['duration_batches'])
        for task_dict in phase_dict['tasks']:
            task_name = task_dict['name']
            phases_task_space.add(task_name)
            p_task_phase[task_name] = task_dict['probability']
        p_task_per_phase.append(p_task_phase)

    assert phases_task_space == set(task_space), "tasks in task_phases does not match tasks in dataset_task_coupling"

    if phase_durations[-1] is None:
        phase_durations[-1] = float("inf")

    if any([d is None for d in phase_durations]):
        raise ValueError("All phases must have a duration except the last phase")

    phase_durations = torch.tensor(phase_durations)
        
    # construct p_t(task), a tensor of shape (n_phases, n_tasks)
    p_task = torch.zeros(n_phases, len(task_space), dtype=torch.float32)
    for phase_idx, p_task_phase in enumerate(p_task_per_phase):
        for task_name, task_prob in p_task_phase.items():
            task_idx = task_space.index(task_name)
            p_task[phase_idx, task_idx] = task_prob

    p_task = p_task / p_task.sum(dim=1, keepdim=True) # normalize by row sum

    # construct p(dataset |task)
    # we assume that p(dataset | task) is not time depenedent, 
    p_d_given_t = torch.zeros(len(task_space), len(dataset_space), dtype=torch.float32)
    for task_idx, task_name in enumerate(task_space):
        for dataset_name, p in dataset_task_coupling[task_name]:
            dataset_idx = dataset_space.index(dataset_name)
            p_d_given_t[task_idx, dataset_idx] = p

    # normalize by row sum
    p_d_given_t = p_d_given_t / p_d_given_t.sum(dim=1, keepdim=True)

    # multiply by p(task) to get p(dataset, task), with shape (n_phases, n_tasks, n_datasets)
    p_dataset_task = p_d_given_t.unsqueeze(0) * p_task.unsqueeze(2)

    # normalize so each phase sums to 1
    per_phase_sum = p_dataset_task.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
    p_dataset_task = p_dataset_task / per_phase_sum

    td_coupling = TaskDatasetCoupling(
        p_dataset_task=p_dataset_task,
        task_space=task_space,
        dataset_space=dataset_space,
        phase_durations=phase_durations
    )
    return td_coupling