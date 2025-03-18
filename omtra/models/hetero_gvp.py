import torch
from torch import nn, einsum
import dgl
import dgl.function as fn
from typing import List, Tuple, Union, Dict
import math

from omtra.models.flowmol_gvp import GVP, GVPDropout, GVPLayerNorm, _norm_no_nan, _rbf
from omtra.graph import to_canonical_etype


class HeteroGVPConv(nn.module):
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[str],
        scalar_size: Dict[str, int],
        vector_size: Dict[str, int],
        n_cp_feats: int = 0,
        scalar_activation=nn.SiLU,
        vector_activation=nn.Sigmoid,
        n_message_gvps: int = 1,
        n_update_gvps: int = 1,
        use_dst_feats: bool = False,
        rbf_dmax: float = 20,
        rbf_dim: int = 16,
        edge_feat_size: Dict[str, int] = None,
        coords_range=10,
        message_norm: Union[float, str] = 10,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.scalar_size = scalar_size
        self.vector_size = vector_size
        self.n_cp_feats = n_cp_feats
        self.scalar_activation = scalar_activation
        self.vector_activation = vector_activation
        self.n_message_gvps = n_message_gvps
        self.n_update_gvps = n_update_gvps
        self.edge_feat_size = edge_feat_size
        self.use_dst_feats = use_dst_feats
        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim
        self.dropout_rate = dropout
        self.message_norm = message_norm

        self.edge_message_fns = nn.ModuleDict()

        for etype in edge_types:
            src_ntype, _, dst_ntype = to_canonical_etype(etype)
            message_gvps = []

            for i in range(n_message_gvps):
                dim_vectors_in = vector_size[src_ntype]
                dim_feats_in = scalar_size[src_ntype]

                if i == 0:
                    dim_vectors_in += 1
                    dim_feats_in += rbf_dim + edge_feat_size.get(etype, 0)
                if use_dst_feats and i == 0:
                    dim_vectors_in += vector_size[dst_ntype]
                    dim_feats_in += scalar_size[dst_ntype]

                dim_vectors_out = (
                    vecotr_size[dst_ntype]
                    if i == n_message_gvps - 1
                    else vector_size[src_ntype]
                )
                dim_feats_out = (
                    scalar_size[dst_ntype]
                    if i == n_message_gvps - 1
                    else scalar_size[src_ntype]
                )

                message_gvps.append(
                    GVP(
                        dim_vectors_in=dim_vectors_in,
                        dim_vectors_out=dim_vectors_out,
                        n_cp_feats=n_cp_feats,
                        dim_feats_in=dim_feats_in,
                        dim_feats_out=dim_feats_out,
                        feats_activation=scalar_activation(),
                        vectors_activation=vector_activation(),
                        vector_gating=True,
                    )
                )
            self.edge_message_fns[etype] = nn.Sequential(*message_gvps)

        self.node_update_fns = nn.ModuleDict()
        self.dropout_layers = nn.ModuleDict()
        self.message_layer_norms = nn.ModuleDict()
        self.update_layer_norms = nn.ModuleDict()

        for ntype in node_types:
            update_gvps = []
            for i in range(n_update_gvps):
                update_gvps.append(
                    GVP(
                        dim_vectors_in=vector_size[ntype],
                        dim_vectors_out=vector_size[ntype],
                        n_cp_feats=n_cp_feats,
                        dim_feats_in=scalar_size[ntype],
                        dim_feats_out=scalar_size[ntype],
                        feats_activation=scalar_activation(),
                        vectors_activation=vector_activation(),
                        vector_gating=True,
                    )
                )

            self.node_update_fns[ntype] = nn.Sequential(*update_gvps)
            self.dropout_layers[ntype] = GVPDropout(self.dropout_rate)
            self.message_layer_norms[ntype] = GVPLayerNorm(scalar_size[ntype])
            self.update_layer_norms[ntype] = GVPLayerNorm(scalar_size[ntype])

        if isinstance(self.message_norm, str) and self.message_norm not in [
            "mean",
            "sum",
        ]:
            raise ValueError(
                f"message_norm must be either 'mean', 'sum', or a number, got {self.message_norm}"
            )
        elif not isinstance(self.message_norm, (float, int, str)):
            raise TypeError("message_norm must be either 'mean', 'sum', or a number")

        if self.message_norm == "mean":
            self.agg_func = fn.mean
        else:
            self.agg_func = fn.sum

    def forward(self, g: dgl.DGLGraph):
        pass
