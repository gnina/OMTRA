import torch
import torch.nn as nn
from typing import Dict, Tuple, Union, List
from omtra.tasks.tasks import Task


class InterpolantScheduler(nn.Module):
    def __init__(
        self,
        schedule_type: str = "linear",
    ):
        super().__init__()

        if schedule_type != "linear":
            raise NotImplementedError(
                "only supports linear schedule across all modalities"
            )

    def alpha_t(self, t: torch.Tensor, task: Task) -> List[Dict[str, torch.Tensor]]:
        alpha_t = [{} for _ in range(t.shape[0])]
        for i, time in enumerate(t):
            alpha_t[i] = {}
            for m in task.modalities_present:
                alpha_t[i][m.name] = time
        return alpha_t

    def alpha_t_prime(
        self, t: torch.Tensor, task: Task
    ) -> List[Dict[str, torch.Tensor]]:
        alpha_t_prime = [{} for _ in range(t.shape[0])]
        for i, time in enumerate(t):
            alpha_t_prime[i] = {}
            for m in task.modalities_present:
                alpha_t_prime[i][m.name] = 1.0
        return alpha_t_prime
