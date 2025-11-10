from dataclasses import dataclass
from typing import List, Optional, Dict, Type
import torch
from torch import nn


default_adaln_params = [
    "shift_att", "scale_att", "gate_att",
    "shift_ff",  "scale_ff",  "gate_ff",
]

# --- Unified generator -------------------------------------------------------
class AdaLNWeightGenerator(nn.Module):

    def __init__(
        self,
        d_model: int,       # target feature dimension for each expanded chunk
        params: List[str] =default_adaln_params,
        ratio: int = 1,         # base compression factor before expansion
        out_dim: Optional[int] = None,  # optional override of final per-chunk size
    ):
        super().__init__()


        assert d_model % ratio == 0, "d_model must be divisible by ratio"
        self.ratio = ratio
        self.d_model = d_model
        self.params = params
        self.n_params = len(params)

        if out_dim is None:
            self.out_dim = d_model
        else:
            self.out_dim = out_dim 

        mlp_out_dim = self.n_params * (d_model // ratio)

        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model, mlp_out_dim, bias=True))

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @property
    def keys(self) -> List[str]:
        return self.spec.keys

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
       # x has shape [B, d_model]
       B, d = x.shape
       assert d == self.d_model

       # Pass through MLP
       flat = self.mlp(x)  # [B, n_params * (d_model // ratio)]
       flat = torch.repeat_interleave(flat, repeats=self.ratio, dim=-1)

       # split and convert to dictionary
       parts = torch.split(flat, self.out_dim, dim=-1)
       out = {k: v for k, v in zip(self.params, parts)}
       return out

# --- Modulation helper (unchanged semantic) ----------------------------------
def modulate(x, shift, scale):
    return x * (1 + scale) + shift
