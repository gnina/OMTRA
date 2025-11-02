import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_dim, frequency_embedding_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim

    @staticmethod
    def timestep_embedding(t, dim=256, max_positions=500):
        # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
        assert len(t.shape) == 1
        t = t * max_positions
        half_dim = dim // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        assert emb.shape == (t.shape[0], dim)
        return emb

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_dim)
        t_emb = self.mlp(t_freq)
        return t_emb
