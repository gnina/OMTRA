import torch
from torch import nn, einsum
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from typing import List, Tuple, Union, Dict
from functools import partial
import math

from omtra.models.flowmol_gvp import GVP, GVPDropout, GVPLayerNorm, _norm_no_nan, _rbf
from omtra.graph import to_canonical_etype


class HeteroGVPConv(nn.module):
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[str],
        scalar_size: int = 128,
        vector_size: int = 16,
        n_cp_feats: int = 0,
        scalar_activation=nn.SiLU,
        vector_activation=nn.Sigmoid,
        n_message_gvps: int = 1,
        n_update_gvps: int = 1,
        attention: bool = False,
        s_message_dim: int = None,
        v_message_dim: int = None,
        n_heads: int = 1,
        n_expansion_gvps: int = 1,
        use_dst_feats: bool = False,
        dst_feat_msg_reduction_factor: float = 4.0,
        rbf_dmax: float = 20,
        rbf_dim: int = 16,
        edge_feat_size: Dict[str, int] = None,
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

        # dims for message reduction and also attention
        self.s_message_dim = s_message_dim
        self.v_message_dim = v_message_dim

        if s_message_dim is None:
            self.s_message_dim = scalar_size

        if v_message_dim is None:
            self.v_message_dim = vector_size

        # determine whether we are performing compressed message passing
        if self.s_message_dim != scalar_size or self.v_message_dim != vector_size:
            self.compressed_messaging = True
        else:
            self.compressed_messaging = False

        self.node_compression = nn.ModuleDict()
        if self.compressed_messaging:
            for ntype in self.node_types:
                compression_gvps = []
                for i in range(
                    n_expansion_gvps
                ):  # implicit here that n_expansion_gvps is the same as n_compression_gvps
                    if i == 0:
                        dim_feats_in = scalar_size
                        dim_vectors_in = vector_size
                    else:
                        dim_feats_in = max(self.s_message_dim, scalar_size)
                        dim_vectors_in = max(self.v_message_dim, vector_size)

                    if i == n_expansion_gvps - 1:
                        dim_feats_out = self.s_message_dim
                        dim_vectors_out = self.v_message_dim
                    else:
                        dim_feats_out = max(self.s_message_dim, scalar_size)
                        dim_vectors_out = max(self.v_message_dim, vector_size)

                    compression_gvps.append(
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
                self.node_compression[ntype] = nn.Sequential(*compression_gvps)
        else:
            for ntype in self.node_types:
                self.node_compression[ntype] = nn.Identity()

        if attention:
            # compute number of features per attention head
            if self.s_message_dim % n_heads != 0 or self.v_message_dim % n_heads != 0:
                raise ValueError(
                    "Number of attention heads must divide the message size."
                )

            self.s_feats_per_head = self.s_message_dim // n_heads
            self.v_feats_per_head = self.v_message_dim // n_heads
            extra_scalar_feats = n_heads * 2

            self.att_weight_projection = nn.ModuleDict()
            for etype in (
                self.edge_types
            ):  # TODO: confirm that attention should be unique to edge type
                self.att_weight_project[etype] = nn.Sequential(
                    nn.Linear(extra_scalar_feats, extra_scalar_feats, bias=False),
                    nn.LayerNorm(extra_scalar_feats),
                )
        else:
            extra_scalar_feats = 0

        if use_dst_feats:
            if dst_feat_msg_reduction_factor != 1:
                s_dst_feats_for_messages = int(
                    self.s_message_dim / dst_feat_msg_reduction_factor
                )
                v_dst_feats_for_messages = int(
                    self.v_message_dim / dst_feat_msg_reduction_factor
                )
                self.dst_feat_msg_projection = nn.ModuleDict()
                for ntype in self.node_types:
                    self.dst_feat_msg_projection[ntype] = GVP(
                        dim_vectors_in=self.v_message_dim,
                        dim_vectors_out=v_dst_feats_for_messages,
                        dim_feats_in=self.s_message_dim,
                        dim_feats_out=s_dst_feats_for_messages,
                        n_cp_feats=0,
                        feats_activation=scalar_activation(),
                        vectors_activation=vector_activation(),
                    )
            else:
                s_dst_feats_for_messages = self.s_message_dim
                v_dst_feats_for_messages = self.v_message_dim
                self.dst_feat_msg_projection = nn.Identity()
        else:
            s_dst_feats_for_messages = 0
            v_dst_feats_for_messages = 0

        self.edge_message_fns = nn.ModuleDict()

        s_slope = (
            self.s_message_dim + extra_scalar_feats - scalar_size
        ) / n_message_gvps
        v_slope = (self.v_message_dim - vector_size) / n_message_gvps

        for etype in edge_types:
            src_ntype, _, dst_ntype = to_canonical_etype(etype)
            message_gvps = []

            for i in range(n_message_gvps):
                dim_vectors_in = self.v_message_dim
                dim_feats_in = self.s_message_dim

                if i == 0:
                    dim_vectors_in += 1
                    dim_feats_in += rbf_dim + edge_feat_size.get(etype, 0)
                else:
                    # if not first layer, input size is the output size of the previous layer
                    dim_feats_in = dim_feats_out
                    dim_vectors_in = dim_vectors_out

                if use_dst_feats and i == 0:
                    dim_vectors_in += v_dst_feats_for_messages
                    dim_feats_in += s_dst_feats_for_messages

                # determine number of scalars output from this layer
                # if message size is smaller than scalar size, do linear interpolation on layer sizes through the gvps
                # otherwise, jump to final output size at first gvp and stay there to the end
                if self.s_message_dim < scalar_size:
                    dim_feats_out = int(s_slope * i + scalar_size)
                    if i == n_message_gvps - 1:
                        dim_feats_out = self.s_message_dim + extra_scalar_feats
                else:
                    dim_feats_out = self.s_message_dim + extra_scalar_feats

                # same logic applied to the number of vectors output from this layer
                if self.v_message_dim < vector_size:
                    dim_vectors_out = int(v_slope * i + vector_size)
                    if i == n_message_gvps - 1:
                        dim_vectors_out = self.v_message_dim
                else:
                    dim_vectors_out = self.v_message_dim

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
            # NOTE: do we want separate message functions per ntype agnostic edge type (interaction/covalent) or consider ntype?
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
                        dim_vectors_in=vector_size,
                        dim_vectors_out=vector_size,
                        n_cp_feats=n_cp_feats,
                        dim_feats_in=scalar_size,
                        dim_feats_out=scalar_size,
                        feats_activation=scalar_activation(),
                        vectors_activation=vector_activation(),
                        vector_gating=True,
                    )
                )

            self.node_update_fns[ntype] = nn.Sequential(*update_gvps)
            self.dropout_layers[ntype] = GVPDropout(self.dropout_rate)
            self.message_layer_norms[ntype] = GVPLayerNorm(self.scalar_size)
            self.update_layer_norms[ntype] = GVPLayerNorm(self.scalar_size)

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

        # if message size is smaller than node embedding size, we need to project aggregated messages back to the node embedding size
        self.message_expansion = nn.ModuleDict()
        if self.compressed_messaging:
            for ntype in self.node_types:
                projection_gvps = []
                for i in range(n_expansion_gvps):
                    if i == 0:
                        dim_feats_in = self.s_message_dim
                        dim_vectors_in = self.v_message_dim
                    else:
                        dim_feats_in = scalar_size
                        dim_vectors_in = vector_size
                    dim_feats_out = scalar_size
                    dim_vectors_out = vector_size

                    projection_gvps.append(
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
                self.message_expansion[ntype] = nn.Sequential(*projection_gvps)
        else:
            for ntype in self.node_types:
                self.mesage_expansion[ntype] = nn.Identity()

    def forward(
        self,
        g: dgl.DGLGraph,
        scalar_feats: Dict[str, torch.Tensor],
        coord_feats: Dict[str, torch.Tensor],
        vec_feats: Dict[str, torch.Tensor],
        edge_feats: Dict[str, torch.Tensor] = None,
        x_diff: Dict[str, torch.Tensor] = None,
        d: torch.Tensor = None,
        passing_edges: List[str] = None,
    ):
        # vec_feat has shape (n_nodes, n_vectors, 3)

        with g.local_scope():
            for ntype in self.node_types:
                # TODO: make sure node attributes are safe across tasks/ntypes e.g. pharm nodes have a 'v', 'x' vs 'x_0' vs 'x_1_true'
                g.nodes[ntype].data["s"] = scalar_feats[ntype]
                g.nodes[ntype].data["x"] = coord_feats[ntype]
                g.nodes[ntype].data["v"] = vec_feats[ntype]

            # edge feature
            for etype in self.edge_types:
                if self.edge_feat_size[etype] > 0:
                    assert edge_feats.get(etype) is not None, (
                        "Edge features must be provided."
                    )
                    g.edges[etype].data["ef"] = edge_feats[etype]

            # normalize x_diff and compute rbf embedding of edge distance
            # dij = torch.norm(g.edges[self.edge_type].data['x_diff'], dim=-1, keepdim=True)
            for etype in self.edge_types:
                if x_diff is not None and d is not None:
                    g.edges[etype].data["x_diff"] = x_diff[etype]
                    g.edges[etype]["d"] = d[etype]
                if "x_diff" not in g.edges[etype].data:
                    # get vectors between node positions
                    g.apply_edges(fn.u_sub_v("x", "x", "x_diff"), etype=etype)
                    dij = (
                        _norm_no_nan(g.edges[etype].data["x_diff"], keepdims=True)
                        + 1e-8
                    )
                    g.edges[etype].data["x_diff"] = g.edges[etype].data["x_diff"] / dij
                    g.edges[etype].data["d"] = _rbf(
                        dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim
                    )

            # apply node compression
            for ntype in self.node_types:
                g.nodes[ntype].data["s"], g.nodes[ntype].data["v"] = (
                    self.node_compression[ntype](
                        (g.nodes[ntype].data["s"], g.nodes[ntype].data["v"])
                    )
                )

            if self.use_dst_feats:
                for ntype in self.node_types:
                    (
                        g.nodes[ntype].data["s_dst_msg"],
                        g.nodes[ntype].data["v_dst_msg"],
                    ) = self.dst_feat_msg_projection[ntype](
                        (g.nodes[ntype].data["s"], g.nodes[ntype].data["v"])
                    )

            # compute messages on passing_edges etypes or all
            if not passing_edges:
                passing_edges = self.edge_types

            for etype in passing_edges:
                etype_message = partial(self.message, etype)
                g.apply_edges(etype_message, etype=etype)

            # if self.attenion, multiple messages by attention weights
            if self.attention:
                for etype in self.edge_types:
                    scalar_msg, att_logits = (
                        g.edges[etype].data["scalar_msg"][:, : self.s_message_dim],
                        g.edges[etype].data["scalar_msg"][:, self.s_message_dim :],
                    )
                    att_logits = self.att_weight_projection[etype](att_logits)

                    etype_graph = g.edge_type_subgraph([etype])
                    att_weights = edge_softmax(et_graph, att_logits)

                    att_weights = edge_softmax(g, att_logits)
                    s_att_weights = att_weights[:, : self.n_heads]
                    v_att_weights = att_weights[:, self.n_heads :]
                    s_att_weights = s_att_weights.repeat_interleave(
                        self.s_feats_per_head, dim=1
                    )
                    v_att_weights = v_att_weights.repeat_interleave(
                        self.v_feats_per_head, dim=1
                    )
                    g.edges[etype].data["scalar_msg"] = scalar_msg * s_att_weights
                    g.edges[etype].data["vec_msg"] = g.edges[etype].data[
                        "vec_msg"
                    ] * v_att_weights.unsqueeze(-1)

            scalar_agg_fns = {}
            vector_agg_fns = {}
            for etype in passing_edges:
                scalar_agg_fns[etype] = (
                    fn.copy_e("scalar_msg", "m"),
                    self.agg_func("m", "scalar_msg"),
                )
                vector_agg_fns[etype] = (
                    fn.copy_e("vec_msg", "m"),
                    self.agg_func("m", "vec_msg"),
                )

            # aggregate messages from every edge
            g.multi_update_all(scalar_agg_fns)
            g.multi_update_all(vector_agg_fns)

            # get aggregated scalar and vector messages
            if isinstance(self.message_norm, str):
                z = 1
            else:
                z = self.message_norm

            updated_scalar_feats = {}
            updated_vec_feats = {}
            for ntype in self.node_types:
                scalar_msg = g.nodes[ntype].data["scalar_msg"] / z
                vec_msg = g.nodes[ntype].data["vec_msg"] / z

                # apply projection (expansion) to aggregated messages
                scalar_msg, vec_msg = self.message_expansion[ntype](
                    (scalar_msg, vec_msg)
                )

                # dropout scalar and vector messages
                scalar_msg, vec_msg = self.dropout_layers[ntype](scalar_msg, vec_msg)

                # update scalar and vector features, apply layernorm
                scalar_feat = g.nodes[ntype].data["s"] + scalar_msg
                vec_feat = g.nodes[ntype].data["v"] + vec_msg
                scalar_feat, vec_feat = self.message_layer_norms[ntype](
                    (scalar_feat, vec_feat)
                )

                # apply node update function, apply dropout to residuals, apply layernorm
                scalar_residual, vec_residual = self.node_update_fns[ntype](
                    (scalar_feat, vec_feat)
                )
                scalar_residual, vec_residual = self.dropout_layers[ntype](
                    scalar_residual, vec_residual
                )
                scalar_feat = scalar_feat + scalar_residual
                vec_feat = vec_feat + vec_residual
                scalar_feat, vec_feat = self.update_layer_norms[ntype](
                    (scalar_feat, vec_feat)
                )

                updated_scalar_feats[ntype] = scalar_feat
                updated_vec_feats[ntype] = vec_feat

            return updated_scalar_feats, updated_vec_feats

    def message(self, edges, etype):
        # concatenate x_diff and v on every edge to produce vector features
        vec_feats = [edges.data["x_diff"].unsqueeze(1), edges.src["v"]]
        if self.use_dst_feats:
            vec_feats.append(edges.dst["v"])
        vec_feats = torch.cat(vec_feats, dim=1)

        # create scalar features
        scalar_feats = [edges.src["s"], edges.data["d"]]
        if self.edge_feat_size[etype] > 0:
            scalar_feats.append(edges.data["ef"])

        if self.use_dst_feats:
            scalar_feats.append(edges.dst["s_dst_msg"])

        scalar_feats = torch.cat(scalar_feats, dim=1)

        scalar_message, vector_message = self.edge_message_fns[etype](
            (scalar_feats, vec_feats)
        )

        return {"scalar_msg": scalar_message, "vec_msg": vector_message}
