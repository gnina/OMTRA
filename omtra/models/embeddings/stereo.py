"""Width-preserving, task-gated additive requested-identity embeddings.

Encodings (`mode`):
  categorical : per-atom R/S/r/s token + per-covalent-bond E/Z token (baseline)
  relational  : RitS-style edge channels written on neighbour pairs (omtra/data/stereo_relational.py)
  both        : the two together
`atom`/`bond` are always constructed so the categorical state dict is identical to earlier runs.
"""
import torch
from torch import nn
from omtra.data.requested_stereo import ATOM_LABELS, BOND_LABELS, eligible
from omtra.data.stereo_relational import N_CHANNELS

MODES = ('categorical', 'relational', 'both')


class StereoEmbedding(nn.Module):
    def __init__(self, node_width, edge_width, mode='categorical', init='normal'):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f'Unknown stereo encoding {mode!r}; expected one of {MODES}')
        if init not in ('normal', 'zero'):
            raise ValueError(f'Unknown stereo embedding init {init!r}; expected normal|zero')
        self.mode = mode
        self.atom = nn.Embedding(len(ATOM_LABELS), node_width)
        self.bond = nn.Embedding(len(BOND_LABELS), edge_width)
        if mode != 'categorical':
            self.rel = nn.Linear(N_CHANNELS, edge_width, bias=False)
        if init == 'zero':      # exact function preservation when the rest of the network is warm-started: every token initially adds nothing
            for module in (self.atom, self.bond) + ((self.rel,) if mode != 'categorical' else ()):
                nn.init.zeros_(module.weight)

    def forward(self, g, task, enabled=True):
        # Always form a zero dependency for DDP, including A and generative tasks.
        zero = (self.atom.weight.sum() + self.bond.weight.sum()) * 0
        if self.mode != 'categorical':
            zero = zero + self.rel.weight.sum() * 0
        if not enabled or not eligible(task):
            return zero, zero
        nd, ed = g.nodes['lig'].data, g.edges['lig_to_lig'].data
        atom_out, edge_out = zero, zero
        if self.mode in ('categorical', 'both'):
            if 'stereo_atom' not in nd or 'stereo_bond' not in ed:
                raise ValueError('Missing requested stereo features on an identity-conditioned task')
            atom_out = self.atom(nd['stereo_atom'])
            bond = self.bond(ed['stereo_bond'])
            # Noncovalent complete-graph edges must not acquire a categorical stereo signal.
            edge_out = bond * (ed['e_t'] > 0).unsqueeze(-1)
        if self.mode in ('relational', 'both'):
            if 'stereo_rel' not in ed:
                raise ValueError('Missing relational stereo features on an identity-conditioned task')
            edge_out = edge_out + self.rel(ed['stereo_rel'].to(self.rel.weight.dtype)) + zero
        return atom_out, edge_out
