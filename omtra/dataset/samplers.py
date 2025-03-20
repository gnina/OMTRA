
import torch
from torch.utils.data import Sampler, DistributedSampler
from typing import Dict, List

from omtra.dataset.chunk_tracker import GraphChunkTracker
from omtra.dataset.multitask import MultitaskDataSet
from omtra.tasks.tasks import Task
from omtra.load.conf import TaskDatasetCoupling
from omtra.tasks.register import task_name_to_class


class MultiTaskSampler(Sampler):

    def __init__(self, 
                 multi_dataset: MultitaskDataSet,
                 td_coupling: TaskDatasetCoupling,
                 edges_per_batch: int,
                 distributed: bool = False,
                 rank: int = None,
                 num_replicas: int = None,
        ):
        super().__init__()
        self.multi_dataset = multi_dataset
        self.edges_per_batch = edges_per_batch
        
        self.distributed = distributed

        # unpack information about the task-dataset coupling
        self.td_coupling = td_coupling
        self.task_space = td_coupling.task_space
        self.dataset_space = td_coupling.dataset_space
        self.p_dataset_task = td_coupling.p_dataset_task # has shape (n_phases, n_tasks, n_datasets)
        self.n_phases, self.n_tasks, self.n_datasets = self.p_dataset_task.shape
        self.datasets = multi_dataset.datasets
        self.tasks = [task_name_to_class(task_name) for task_name in self.task_space]

        self.batch_idx = 0

        if self.distributed:
            self.num_replicas = num_replicas if num_replicas is not None else torch.distributed.get_world_size()
            self.rank = rank if rank is not None else torch.distributed.get_rank()

            dataset_frac_per_worker = 1.0 / self.num_replicas
            self.frac_start = self.rank * dataset_frac_per_worker
            self.frac_end = (self.rank + 1) * dataset_frac_per_worker
        else:
            self.rank = 0
            self.num_replicas = 1
            self.frac_start = 0
            self.frac_end = 1

    def sample_task_and_dataset(self):

        # find the current phase of training
        phase_durations = self.td_coupling.phase_durations
        phase_boundaries = phase_durations.cumsum(dim=0)
        phase_idx = torch.searchsorted(phase_boundaries, self.batch_idx, right=True).item()

        p = self.p_dataset_task[phase_idx]

        # Flatten the tensor to work with torch.multinomial
        flat_p = p.flatten()

        # Draw a single sample from the flattened distribution
        index = torch.multinomial(flat_p, 1).item()

        # Convert the flat index back to 2D indices
        n, m = p.shape
        task_idx, dataset_idx = divmod(index, m)

        return task_idx, dataset_idx
    
    def build_chunk_trackers(self):
        self.chunk_trackers: Dict[int, GraphChunkTracker] = {}
        self.td_pair_to_chunk_tracker_id = {}

        for dataset_idx, dataset_name in enumerate(self.dataset_space):
            # the following line first selects probabilities across all phases/tasks that pertain to a particular dataset
            # then sums over the phases; this yields a tensor of shape (n_tasks,) where tasks
            #  that have some-non-zero probability of being used in the dataset have a non-zero value
            sum_dataset_probs = self.p_dataset_task[:, :, dataset_idx].sum(dim=0) 
            task_idxs = sum_dataset_probs.nonzero(as_tuple=True)[0] # get the indices of the non-zero values
            task_idxs = tuple(task_idxs.tolist())
            # tasks = [self.tasks[self.task_names[task_idx]] for task_idx in task_idxs]

            chunk_tracker_args = [self.edges_per_batch, self.frac_start, self.frac_end]

            if dataset_name == 'pharmit':
                # create a single chunk tracker for all tasks
                chunk_tracker_idx = len(self.chunk_trackers)
                self.chunk_trackers[chunk_tracker_idx] = GraphChunkTracker(
                    self.datasets[dataset_name],
                    *chunk_tracker_args)
                for task_idx in task_idxs:
                    self.td_pair_to_chunk_tracker_id[(task_idx, dataset_idx)] = chunk_tracker_idx

            elif dataset_name == 'plinder':

                for plinder_link_version, plidner_dataset_object in self.datasets[dataset_name].items():
                    chunk_tracker_idx = len(self.chunk_trackers)
                    self.chunk_trackers[chunk_tracker_idx] = GraphChunkTracker(
                        plidner_dataset_object,
                        *chunk_tracker_args
                    )
                    # get the tasks that use this plinder link version
                    task_idxs = []
                    for task_idx, task_name in enumerate(self.task_space):
                        task_class = task_name_to_class(task_name)
                        if task_class.plinder_link_version == plinder_link_version:
                            task_idxs.append(task_idx)

                    for task_idx in task_idxs:
                        self.td_pair_to_chunk_tracker_id[(task_idx, dataset_idx)] = chunk_tracker_idx
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not supported")

    def get_td_pair_distributed(self):
        if self.rank == 0:
            task_idx, dataset_idx = self.sample_task_and_dataset()
            task_idx_tensor = torch.tensor(task_idx, dtype=torch.int64)
            dataset_idx_tensor = torch.tensor(dataset_idx, dtype=torch.int64)
        else:
            task_idx_tensor = torch.tensor(0, dtype=torch.int64)
            dataset_idx_tensor = torch.tensor(0, dtype=torch.int64)

        torch.distributed.broadcast(task_idx_tensor, src=0)
        torch.distributed.broadcast(dataset_idx_tensor, src=0)

        task_idx = task_idx_tensor.item()
        dataset_idx = dataset_idx_tensor.item()
        return task_idx, dataset_idx

    def __iter__(self):

        self.build_chunk_trackers()
        while True:
            if self.distributed:
                task_idx, dataset_idx = self.get_td_pair_distributed()
            else:
                task_idx, dataset_idx = self.sample_task_and_dataset()

            # get chunk tracker
            chunk_tracker_idx = self.td_pair_to_chunk_tracker_id[(task_idx, dataset_idx)]
            chunk_tracker: GraphChunkTracker = self.chunk_trackers[chunk_tracker_idx]

            # get next batch of indices
            batch_idxs = chunk_tracker.get_batch_idxs(self.tasks[task_idx])

            # construct the global indices
            global_idxs = [ (task_idx, dataset_idx, idx) for idx in batch_idxs ]
            
            yield global_idxs

    def state_dict(self):
        return {
            'batch_idx': self.batch_idx,
        }
    
    def load_state_dict(self, state_dict):
        self.batch_idx = state_dict['batch_idx']