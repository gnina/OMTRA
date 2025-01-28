import torch
from torch.utils.data import Sampler, DistributedSampler

from omtra.dataset.chunk_tracker import ChunkTracker
from omtra.dataset.multitask import MultitaskDataSet


class MultiTaskSampler(Sampler):

    def __init__(self, multi_dataset: MultitaskDataSet, batch_size):
        super().__init__(multi_dataset)
        self.multi_dataset = multi_dataset
        self.batch_size = batch_size
        

        self.task_names = multi_dataset.task_names
        self.dataset_names = multi_dataset.dataset_names
        self.p_dataset_task = multi_dataset.p_dataset_task

    def sample_task_and_dataset(self):

        p = self.p_dataset_task

        # Flatten the tensor to work with torch.multinomial
        flat_p = p.flatten()

        # Draw a single sample from the flattened distribution
        index = torch.multinomial(flat_p, 1).item()

        # Convert the flat index back to 2D indices
        n, m = p.shape
        task_idx, dataset_idx = divmod(index, m)

        return task_idx, dataset_idx
    
    def build_chunk_trackers(self):
        self.chunk_trackers = {}
        self.td_pair_to_chunk_tracker_id = {}

        for dataset_idx, dataset_name in enumerate(self.dataset_names):
            task_idxs = self.p_dataset_task[:, dataset_idx].nonzero(as_tuple=True)[0]
            # tasks = [self.tasks[self.task_names[task_idx]] for task_idx in task_idxs]

            if dataset_name == 'pharmit':
                # create a single chunk tracker for all tasks
                chunk_tracker_idx = len(self.chunk_trackers)
                self.chunk_trackers[chunk_tracker_idx] = ChunkTracker(self.datasets[dataset_name])
                for task_idx in task_idxs:
                    self.td_pair_to_chunk_tracker_id[(task_idx, dataset_idx)] = chunk_tracker_idx

            elif dataset_name == 'plinder':
                # divide tasks into those that use the apo state and those that don't
                # and create separate chunk trackers for each
                tasks = [self.tasks[self.task_names[task_idx]] for task_idx in task_idxs]
                tasks_not_using_apo = [task_idx for task_idx, task in zip(task_idxs, tasks) if not task.uses_apo]
                tasks_using_apo = [task_idx for task_idx, task in zip(task_idxs, tasks) if task.uses_apo]
                if len(tasks_not_using_apo) != 0:
                    chunk_tracker_idx = len(self.chunk_trackers)
                    self.chunk_trackers[chunk_tracker_idx] = ChunkTracker(self.datasets[dataset_name])
                    for task_idx in tasks_not_using_apo:
                        self.td_pair_to_chunk_tracker_id[(task_idx, dataset_idx)] = chunk_tracker_idx
                if len(tasks_using_apo) != 0:
                    chunk_tracker_idx = len(self.chunk_trackers)
                    self.chunk_trackers[chunk_tracker_idx] = ChunkTracker(
                        self.datasets[dataset_name],
                        apo_systems=True
                    )
                    for task_idx in tasks_using_apo:
                        self.td_pair_to_chunk_tracker_id[(task_idx, dataset_idx)] = chunk_tracker_idx

            else:
                raise NotImplementedError(f"Dataset {dataset_name} not supported")



    def __iter__(self):

        # TODO: (i'm working on it i promise)
        pass