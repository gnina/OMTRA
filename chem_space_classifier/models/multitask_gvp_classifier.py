"""
Multitask GVP Classifier for 5-Database Classification

Predicts which chemical databases a molecule belongs to:
- CSC (task index 2)
- MCULE (task index 5)
- PubChem (task index 8)
- ZINC (task index 12)
- MolPort (task index 6)

Architecture:
    1. Shared GVP backbone (4 layers)
    2. Shared representation layer (256d)
    3. 5 task-specific branches (128d each)
    4. 5 binary classification heads

This enables learning shared molecular features while allowing
task-specific decision boundaries for each database.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from typing import Dict, List, Optional

# Import helper functions from mcule_gvp_classifier
import sys
sys.path.insert(0, '/net/dali/home/mscbio/mag1037/work/rotations/koes/OMTRA')

def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    """L2 norm of tensor clamped above a minimum value `eps`."""
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16):
    """Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1."""
    device = D.device
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


# Import HeteroGVPConv
try:
    from omtra.models.gvp import HeteroGVPConv
    HETERO_GVP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import HeteroGVPConv: {e}")
    HETERO_GVP_AVAILABLE = False
    HeteroGVPConv = None


class MultitaskGVPClassifier(nn.Module):
    """
    Multitask classifier for 5 database prediction tasks.

    Predicts binary labels for: CSC, MCULE, PubChem, ZINC, MolPort
    """

    def __init__(
        self,
        # Embedding dimensions
        n_atom_types: int = 118,
        n_charge_types: int = 10,
        n_bond_types: int = 4,
        atom_emb_dim: int = 64,
        charge_emb_dim: int = 64,
        bond_emb_dim: int = 64,

        # GVP architecture
        hidden_dim: int = 256,
        edge_dim: int = 128,
        n_vec_channels: int = 16,
        num_layers: int = 4,

        # RBF encoding
        rbf_dim: int = 32,
        rbf_dmax: float = 20.0,

        # Multitask architecture
        shared_repr_dim: int = 256,
        task_hidden_dim: int = 128,
        n_tasks: int = 5,

        # Regularization
        dropout: float = 0.2,

        # Readout
        readout: str = 'mean',
    ):
        super().__init__()

        if not HETERO_GVP_AVAILABLE:
            raise ImportError("HeteroGVPConv is required but not available!")

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.n_vec_channels = n_vec_channels
        self.num_layers = num_layers
        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax
        self.readout = readout
        self.n_tasks = n_tasks

        # ================================================================
        # EMBEDDING LAYERS
        # ================================================================

        self.atom_embedding = nn.Embedding(n_atom_types, atom_emb_dim)
        self.charge_embedding = nn.Embedding(n_charge_types, charge_emb_dim)
        self.bond_embedding = nn.Embedding(n_bond_types, bond_emb_dim)

        # ================================================================
        # FEATURE PROJECTION
        # ================================================================

        node_input_dim = atom_emb_dim + charge_emb_dim
        self.node_projection = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.edge_projection = nn.Sequential(
            nn.Linear(bond_emb_dim, edge_dim),
            nn.SiLU(),
            nn.LayerNorm(edge_dim),
        )

        # ================================================================
        # SHARED GVP BACKBONE
        # ================================================================

        self.gvp_layers = nn.ModuleList([
            HeteroGVPConv(
                node_types=['lig'],
                edge_types=['lig_to_lig'],
                scalar_size=hidden_dim,
                vector_size=n_vec_channels,
                n_cp_feats=0,
                scalar_activation=nn.SiLU,
                vector_activation=nn.Sigmoid,
                n_message_gvps=3,
                n_update_gvps=3,
                attention=False,
                rbf_dmax=rbf_dmax,
                rbf_dim=rbf_dim,
                edge_feat_size={'lig_to_lig': edge_dim},
                message_norm=100,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # ================================================================
        # READOUT
        # ================================================================

        if readout == 'mean':
            self.readout_fn = dgl.readout_nodes
            self.readout_op = 'mean'
        elif readout == 'sum':
            self.readout_fn = dgl.readout_nodes
            self.readout_op = 'sum'
        elif readout == 'max':
            self.readout_fn = dgl.readout_nodes
            self.readout_op = 'max'
        else:
            raise ValueError(f"Unknown readout: {readout}")

        # ================================================================
        # SHARED REPRESENTATION LAYER
        # ================================================================

        self.shared_repr = nn.Sequential(
            nn.Linear(hidden_dim, shared_repr_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(shared_repr_dim),
        )

        # ================================================================
        # TASK-SPECIFIC HEADS
        # ================================================================

        # Create 5 task-specific branches
        # Each has its own hidden layer before the binary classification head
        self.task_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_repr_dim, task_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(task_hidden_dim, 1),  # Binary output
            )
            for _ in range(n_tasks)
        ])

        # Task names for reference
        self.task_names = ['CSC', 'MCULE', 'PubChem', 'ZINC', 'MolPort']
        self.task_indices = [2, 5, 8, 12, 6]  # Original indices in pharmit_library

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        """
        Forward pass for multitask classification.

        Args:
            g: DGL graph with node features:
                - 'a_1_true': [num_nodes] atom types
                - 'c_1_true': [num_nodes] charges
                - 'x_1_true': [num_nodes, 3] coordinates
               And edge features:
                - 'e_1_true': [num_edges] bond types

        Returns:
            logits: [batch_size, 5] - logits for 5 binary classification tasks
        """
        device = next(self.parameters()).device

        # ================================================================
        # EXTRACT FEATURES
        # ================================================================

        atom_types = g.nodes['lig'].data['a_1_true'].to(device)
        charges = g.nodes['lig'].data['c_1_true'].to(device)
        coords = g.nodes['lig'].data['x_1_true'].to(device)
        bond_types = g.edges['lig_to_lig'].data['e_1_true'].to(device)

        # ================================================================
        # EMBED FEATURES
        # ================================================================

        atom_emb = self.atom_embedding(atom_types)
        charge_emb = self.charge_embedding(charges)

        node_features = torch.cat([atom_emb, charge_emb], dim=-1)
        node_scalars = self.node_projection(node_features)

        bond_emb = self.bond_embedding(bond_types)
        edge_features = self.edge_projection(bond_emb)

        # Initialize node vectors
        num_nodes = g.num_nodes('lig')
        node_vectors = torch.zeros(
            num_nodes, self.n_vec_channels, 3,
            device=device, dtype=torch.float32
        )

        node_positions = coords

        # ================================================================
        # PRECOMPUTE EDGE GEOMETRY
        # ================================================================

        g.nodes['lig'].data['x'] = node_positions
        g.apply_edges(
            lambda edges: {'x_diff': edges.src['x'] - edges.dst['x']},
            etype='lig_to_lig'
        )

        x_diff = g.edges['lig_to_lig'].data['x_diff']
        dij = _norm_no_nan(x_diff, keepdims=True) + 1e-8
        x_diff_normalized = x_diff / dij
        d_rbf = _rbf(dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim)

        g.edges['lig_to_lig'].data['x_diff'] = x_diff_normalized
        g.edges['lig_to_lig'].data['d'] = d_rbf

        x_diff_dict = {'lig_to_lig': x_diff_normalized}
        d_dict = {'lig_to_lig': d_rbf}

        # ================================================================
        # SHARED GVP MESSAGE PASSING
        # ================================================================

        for gvp_layer in self.gvp_layers:
            node_scalars_dict, node_vectors_dict = gvp_layer(
                g,
                scalar_feats={'lig': node_scalars},
                coord_feats={'lig': node_positions},
                vec_feats={'lig': node_vectors},
                edge_feats={'lig_to_lig': edge_features},
                x_diff=x_diff_dict,
                d=d_dict,
            )

            node_scalars = node_scalars_dict['lig']
            node_vectors = node_vectors_dict['lig']

        # ================================================================
        # GRAPH-LEVEL READOUT
        # ================================================================

        g.nodes['lig'].data['h'] = node_scalars
        graph_features = self.readout_fn(
            g, 'h', op=self.readout_op, ntype='lig'
        )

        # ================================================================
        # SHARED REPRESENTATION
        # ================================================================

        shared_repr = self.shared_repr(graph_features)

        # ================================================================
        # TASK-SPECIFIC PREDICTIONS
        # ================================================================

        # Compute logits for each task
        task_logits = []
        for task_branch in self.task_branches:
            logits = task_branch(shared_repr)  # [batch_size, 1]
            task_logits.append(logits)

        # Stack into [batch_size, 5]
        all_logits = torch.cat(task_logits, dim=1)

        return all_logits


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model):
    """Print a summary of the model architecture."""
    print("=" * 80)
    print("MULTITASK GVP CLASSIFIER SUMMARY")
    print("=" * 80)
    print(f"Total parameters: {count_parameters(model):,}")
    print()

    print("Embedding layers:")
    print(f"  Atom:   {model.atom_embedding.num_embeddings} → {model.atom_embedding.embedding_dim}")
    print(f"  Charge: {model.charge_embedding.num_embeddings} → {model.charge_embedding.embedding_dim}")
    print(f"  Bond:   {model.bond_embedding.num_embeddings} → {model.bond_embedding.embedding_dim}")
    print()

    print("Architecture:")
    print(f"  GVP layers:      {model.num_layers}")
    print(f"  Hidden dim:      {model.hidden_dim}")
    print(f"  Edge dim:        {model.edge_dim}")
    print(f"  Vector channels: {model.n_vec_channels}")
    print(f"  RBF dim:         {model.rbf_dim}")
    print(f"  Readout:         {model.readout}")
    print()

    print(f"Multitask heads: {model.n_tasks} tasks")
    for i, task_name in enumerate(model.task_names):
        print(f"  {i+1}. {task_name} (index {model.task_indices[i]})")
    print()

    print("Task branch architecture:")
    print(f"  Shared repr: {model.hidden_dim} → 256d")
    print(f"  Task-specific: 256d → 128d → 1 (per task)")
    print("=" * 80)
