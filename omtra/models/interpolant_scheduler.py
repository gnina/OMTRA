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

    def weights(self, t: torch.Tensor, task: Task) -> List[Dict[str, torch.Tensor]]:
        alpha_t = {}
        beta_t = {}
        for m in task.modalities_present:
            alpha_t[m.name] = 1.0 - t
            beta_t[m.name] = t
        return alpha_t, beta_t

    def weight_derivative(
        self, t: torch.Tensor, task: Task
    ) -> List[Dict[str, torch.Tensor]]:
        alpha_t = {}
        beta_t = {}
        for m in task.modalities_present:
            alpha_t[m.name] = torch.full_like(t, -1.0)
            beta_t[m.name] = torch.full_like(t, 1.0)
        return alpha_t, beta_t
