"""
Confidence prediction module for OMTRA.
"""

import torch
import torch.nn as nn

class ConfidenceModule(nn.Module):
    """
    Predicts per-atom pLDDT (of a batched graph) confidence scores.

    Simple MLP that takes atom features and outputs binned confidence predictions.
    Uses 50 bins in [0, 1] range like AlphaFold.

    ref: https://github.com/google-deepmind/alphafold/blob/09ed0c5d5a32d794ed9f78b70906cbeaff0ef439/alphafold/model/modules.py#L1155

    Args:
        input_dim: Dimension of input atom features (from scalar_latents)
        hidden_dim: Hidden layer dimension
        n_bins: Number of confidence bins (default: 50)
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int = 256, # currently matches scalar_latents dim
        hidden_dim: int = 256,
        n_bins: int = 50,
        dropout: float = 0.1
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_bins)
        )

    def forward(self, atom_features: torch.Tensor) -> torch.Tensor:
        """
        Predict confidence scores for atoms.
            Args: atom_features: [N_atoms, input_dim] in super-graph format
            Returns: logits: [N_atoms, n_bins] confidence logits
        """
        return self.mlp(atom_features)