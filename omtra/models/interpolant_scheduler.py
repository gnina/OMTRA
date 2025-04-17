import torch
import torch.nn as nn
from typing import Dict, Tuple, Union
from omtra.tasks.tasks import Task


class InterpolantScheduler(nn.Module):

    def __init__(
        self,
        schedule_type: str = "linear",
    ):
        super().__init__()

        if schedule_type != 'linear':
            raise NotImplementedError('only supports linear schedule across all modalities')


    def alpha_t(self, t: torch.Tensor, task: Task) -> torch.Tensor:
        for m in task.modalities_present:
            return t

    def alpha_t_prime(self, t: torch.Tensor, task: Task) -> torch.Tensor:
        for m in task.modalities_present:
            return torch.ones_like(t)
