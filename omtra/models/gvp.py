import torch
from torch import nn, einsum
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from typing import List, Tuple, Union, Dict, Optional, Set
from functools import partial
import math
from torch_scatter import scatter_softmax

from omtra.data.graph import to_canonical_etype, get_inv_edge_type


# most taken from flowmol gvp (moreflowmol branch)
# heteroconv adapted from GVPConv in flowmol gvp
# from line_profiler import LineProfiler, profile


# helper functions
def exists(val):
    return val is not None


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    """
    L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    """
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    """
    device = D.device
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


# the classes GVP, GVPDropout, and GVPLayerNorm are taken from lucidrains' geometric-vector-perceptron repository
# https://github.com/lucidrains/geometric-vector-perceptron/tree/main
# some adaptations have been made to these classes to make them more consistent with the original GVP paper/implementation
# specifically, using _norm_no_nan instead of torch's built in norm function, and the weight intialiation scheme for Wh and Wu


class GVP(nn.Module):
    def __init__(
        self,
        dim_vectors_in,
        dim_vectors_out,
        dim_feats_in,
        dim_feats_out,
        n_cp_feats=0,  # number of cross-product features added to hidden vector features
        hidden_vectors=None,
        feats_activation=nn.SiLU(),
        vectors_activation=nn.Sigmoid(),
        vector_gating=True,
        xavier_init=False,
    ):
        super().__init__()
        self.dim_vectors_in = dim_vectors_in
        self.dim_feats_in = dim_feats_in
        self.n_cp_feats = n_cp_feats

        self.dim_vectors_out = dim_vectors_out
        dim_h = (
            max(dim_vectors_in, dim_vectors_out)
            if hidden_vectors is None
            else hidden_vectors
        )

        # create Wh matrix
        wh_k = 1 / math.sqrt(dim_vectors_in)
        self.Wh = torch.zeros(dim_vectors_in, dim_h, dtype=torch.float32).uniform_(
            -wh_k, wh_k
        )
        self.Wh = nn.Parameter(self.Wh)

        # create Wcp matrix if we are using cross-product features
        if n_cp_feats > 0:
            wcp_k = 1 / math.sqrt(dim_vectors_in)
            self.Wcp = torch.zeros(
                dim_vectors_in, n_cp_feats * 2, dtype=torch.float32
            ).uniform_(-wcp_k, wcp_k)
            self.Wcp = nn.Parameter(self.Wcp)

        # create Wu matrix
        if (
            n_cp_feats > 0
        ):  # the number of vector features going into Wu is increased by n_cp_feats if we are using cross-product features
            wu_in_dim = dim_h + n_cp_feats
        else:
            wu_in_dim = dim_h
        wu_k = 1 / math.sqrt(wu_in_dim)
        self.Wu = torch.zeros(wu_in_dim, dim_vectors_out, dtype=torch.float32).uniform_(
            -wu_k, wu_k
        )
        self.Wu = nn.Parameter(self.Wu)

        if isinstance(vectors_activation, nn.Sigmoid):
            self.fuse_activation = True
        else:
            self.fuse_activation = False
        self.vectors_activation = vectors_activation

        self.to_feats_out = nn.Sequential(
            nn.Linear(dim_h + n_cp_feats + dim_feats_in, dim_feats_out),
            feats_activation,
        )

        # branching logic to use old GVP, or GVP with vector gating
        if vector_gating:
            self.scalar_to_vector_gates = nn.Linear(dim_feats_out, dim_vectors_out)
            if xavier_init:
                nn.init.xavier_uniform_(self.scalar_to_vector_gates.weight, gain=1)
                nn.init.constant_(self.scalar_to_vector_gates.bias, 0)
        else:
            self.scalar_to_vector_gates = None

        # self.scalar_to_vector_gates = nn.Linear(dim_feats_out, dim_vectors_out) if vector_gating else None

    # @profile
    def forward(self, data):
        feats, vectors = data
        b, n, _, v, c = *feats.shape, *vectors.shape

        # feats has shape (batch_size, n_feats)
        # vectors has shape (batch_size, n_vectors, 3)

        assert c == 3 and v == self.dim_vectors_in, "vectors have wrong dimensions"
        assert n == self.dim_feats_in, "scalar features have wrong dimensions"

        # Vh = einsum(
        #     "b v c, v h -> b h c", vectors, self.Wh
        # )  # has shape (batch_size, dim_h, 3)
        # 1) move channel dim to front
        v = vectors.transpose(1, 2)            # [B, C, V]
        # 2) batched matmul
        Vh_mat = torch.matmul(v, self.Wh)      # [B, C, H]
        # 3) restore to [B, H, C]
        Vh = Vh_mat.transpose(1, 2)      # [B, H, C]

        # if we are including cross-product features, compute them here
        if self.n_cp_feats > 0:
            # convert dim_vectors_in vectors to n_cp_feats*2 vectors
            # Vcp = einsum(
            #     "b v c, v p -> b p c", vectors, self.Wcp
            # )  # has shape (batch_size, n_cp_feats*2, 3)

            # compute Vcp via matmul instead of einsum
            v_ = vectors.transpose(1, 2)               # [B, C, V]
            Vcp_mat = torch.matmul(v_, self.Wcp)       # [B, C, P]
            Vcp = Vcp_mat.transpose(1, 2)              # [B, P, C]


            # split the n_cp_feats*2 vectors into two sets of n_cp_feats vectors
            cp_src, cp_dst = torch.split(
                Vcp, self.n_cp_feats, dim=1
            )  # each has shape (batch_size, n_cp_feats, 3)
            # take the cross product of the two sets of vectors
            cp = torch.linalg.cross(
                cp_src, cp_dst, dim=-1
            )  # has shape (batch_size, n_cp_feats, 3)

            # add the cross product features to the hidden vector features
            Vh = torch.cat(
                (Vh, cp), dim=1
            )  # has shape (batch_size, dim_h + n_cp_feats, 3)

        # Vu = einsum(
        #     "b h c, h u -> b u c", Vh, self.Wu
        # )  # has shape (batch_size, dim_vectors_out, 3)
        # compute Vu via matmul instead of einsum
        vh_ = Vh.transpose(1, 2)                   # [B, C, H]
        Vu_mat = torch.matmul(vh_, self.Wu)        # [B, C, U]
        Vu = Vu_mat.transpose(1, 2)                # [B, U, C]

        sh = _norm_no_nan(Vh)

        s = torch.cat((feats, sh), dim=1)

        feats_out = self.to_feats_out(s)

        if exists(self.scalar_to_vector_gates):
            gating = self.scalar_to_vector_gates(feats_out)
            gating = gating.unsqueeze(dim=-1)
        else:
            gating = _norm_no_nan(Vu)

        vectors_out = self.vectors_activation(gating) * Vu

        # if torch.isnan(feats_out).any() or torch.isnan(vectors_out).any():
        #     raise ValueError("NaNs in GVP forward pass")

        return (feats_out, vectors_out)


class _VDropout(nn.Module):
    """
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    """

    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        """
        :param x: `torch.Tensor` corresponding to vector channels
        """
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class GVPDropout(nn.Module):
    """Separate dropout for scalars and vectors."""

    def __init__(self, rate):
        super().__init__()
        self.vector_dropout = _VDropout(rate)
        self.feat_dropout = nn.Dropout(rate)

    def forward(self, feats, vectors):
        return self.feat_dropout(feats), self.vector_dropout(vectors)


class GVPLayerNorm(nn.Module):
    """Normal layer norm for scalars, nontrainable norm for vectors."""

    def __init__(self, feats_h_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.feat_norm = nn.LayerNorm(feats_h_size)

    def forward(self, data):
        feats, vectors = data

        normed_feats = self.feat_norm(feats)

        vn = _norm_no_nan(vectors, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True) + self.eps) + self.eps
        normed_vectors = vectors / vn
        return normed_feats, normed_vectors


class HeteroGVPConv(nn.Module):
    def __init__(
        self,
        node_types: Union[List[str], Set[str]],
        edge_types: Union[List[str], Set[str]],
        scalar_size: int = 128,
        vector_size: int = 16,
        n_cp_feats: int = 0,
        scalar_activation=nn.SiLU,
        vector_activation=nn.Sigmoid,
        n_message_gvps: int = 1,
        n_update_gvps: int = 1,
        attention: bool = False,
        att_type: str = 'crosstype',
        s_message_dim: int = None,
        v_message_dim: int = None,
        n_heads: int = 1,
        n_expansion_gvps: int = 1,
        use_dst_feats: bool = False,
        dst_feat_msg_reduction_factor: float = 4.0,
        rbf_dmax: float = 20,
        rbf_dim: int = 16,
        edge_feat_size: Optional[Dict[str, int]] = None,
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
        self.attention = attention
        self.att_type = att_type
        self.n_heads = n_heads

        if not self.edge_feat_size:
            self.edge_feat_size = {etype: 0 for etype in self.edge_types}

        if self.att_type == 'crosstype':
            self.att_func = self.compute_att_weights_crosstype
        elif self.att_type == 'pertype':
            self.att_func = self.compute_att_weights_per_type

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
                inv_etype = get_inv_edge_type(etype)
                if inv_etype in self.att_weight_projection:
                    self.att_weight_projection[etype] = self.att_weight_projection[inv_etype]
                    continue
                
                self.att_weight_projection[etype] = nn.Sequential(
                    nn.Linear(extra_scalar_feats, extra_scalar_feats, bias=False),
                    nn.LayerNorm(extra_scalar_feats),
                )
        else:
            extra_scalar_feats = 0

        if use_dst_feats:
            self.dst_feat_msg_projection = nn.ModuleDict()
            if dst_feat_msg_reduction_factor != 1:
                s_dst_feats_for_messages = int(
                    self.s_message_dim / dst_feat_msg_reduction_factor
                )
                v_dst_feats_for_messages = int(
                    self.v_message_dim / dst_feat_msg_reduction_factor
                )
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
                self.dst_feat_msg_projection[ntype] = nn.Identity()
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
            inv_etype = get_inv_edge_type(etype)
            if inv_etype in self.edge_message_fns:
                self.edge_message_fns[etype] = self.edge_message_fns[inv_etype]
                continue
            message_gvps = []

            for i in range(n_message_gvps):
                dim_vectors_in = self.v_message_dim
                dim_feats_in = self.s_message_dim

                if i == 0:
                    dim_vectors_in += 1
                    dim_feats_in += rbf_dim + self.edge_feat_size.get(etype, 0)
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
            self.cross_reducer = "mean"
        else:
            self.agg_func = fn.sum
            self.cross_reducer = "sum"

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
                self.message_expansion[ntype] = nn.Identity()

    # @profile
    def forward(
        self,
        g: dgl.DGLGraph,
        scalar_feats: Dict[str, torch.Tensor],
        coord_feats: Dict[str, torch.Tensor],
        vec_feats: Dict[str, torch.Tensor],
        edge_feats: Optional[Dict[str, torch.Tensor]] = None,
        x_diff: Optional[Dict[str, torch.Tensor]] = None,
        d: Optional[Dict[str, torch.Tensor]] = None,
        passing_edges: Optional[List[str]] = None,
    ):
        # vec_feat has shape (n_nodes, n_vectors, 3)

        with g.local_scope():
            for ntype in self.node_types:
                if g.num_nodes(ntype) == 0:
                    continue
                # TODO: make sure node attributes are safe across tasks/ntypes e.g. pharm nodes have a 'v', 'x' vs 'x_0' vs 'x_1_true'
                g.nodes[ntype].data["s"] = scalar_feats[ntype]
                g.nodes[ntype].data["x"] = coord_feats[ntype]
                g.nodes[ntype].data["v"] = vec_feats[ntype]

            # edge feature
            for etype in self.edge_types:
                if etype not in g.etypes or g.num_edges(etype) == 0:
                    continue
                if self.edge_feat_size[etype] > 0:
                    assert edge_feats.get(etype) is not None, (
                        "Edge features must be provided."
                    )
                    g.edges[etype].data["ef"] = edge_feats[etype]

            # normalize x_diff and compute rbf embedding of edge distance
            # dij = torch.norm(g.edges[self.edge_type].data['x_diff'], dim=-1, keepdim=True)
            for etype in self.edge_types:
                if etype not in g.etypes or g.num_edges(etype) == 0:
                    continue
                if x_diff is not None and d is not None:
                    g.edges[etype].data["x_diff"] = x_diff[etype]
                    g.edges[etype].data["d"] = d[etype]
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
                if g.num_nodes(ntype) == 0:
                    continue
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
                if etype not in g.etypes or g.num_edges(etype) == 0:
                    continue
                etype_message = partial(self.message, etype=etype)
                g.apply_edges(etype_message, etype=etype)

            # if self.attenion, multiple messages by attention weights
            if self.attention:
                self.att_func(g, passing_edges)

            scalar_agg_fns = {}
            vector_agg_fns = {}
            for etype in passing_edges:
                if etype not in g.etypes or g.num_edges(etype) == 0:
                    continue
                """
                g.update_all(
                    fn.copy_e("scalar_msg", "m"),
                    self.agg_func("m", "scalar_msg"),
                    etype=etype,
                )
                g.update_all(
                    fn.copy_e("vec_msg", "m"),
                    self.agg_func("m", "vec_msg"),
                    etype=etype,
                )
                """
                
                
                scalar_agg_fns[etype] = (
                    fn.copy_e("scalar_msg", "m"),
                    self.agg_func("m", "scalar_msg"),
                )
                vector_agg_fns[etype] = (
                    fn.copy_e("vec_msg", "m"),
                    self.agg_func("m", "vec_msg"),
                )

            g.multi_update_all(scalar_agg_fns, cross_reducer=self.cross_reducer)
            g.multi_update_all(vector_agg_fns, cross_reducer=self.cross_reducer)

            # get aggregated scalar and vector messages
            if isinstance(self.message_norm, str):
                z = 1
            else:
                z = self.message_norm

            updated_scalar_feats = {}
            updated_vec_feats = {}
            for ntype in self.node_types:
                if g.num_nodes(ntype) == 0:
                    continue
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

    def compute_att_weights_crosstype(self, g: dgl.DGLHeteroGraph, passing_edges: List[str]):


        # collect valid etypes
        valid_etypes = []
        dst_ntypes = set()
        for etype in passing_edges:
            if etype not in g.etypes or g.num_edges(etype) == 0:
                continue
            valid_etypes.append(etype)
            _, _, dst_ntype = to_canonical_etype(etype)
            dst_ntypes.add(dst_ntype)

        ntype_mins = {}
        current_min = 0
        for dst_ntype in dst_ntypes:
            ntype_mins[dst_ntype] = current_min
            current_min += g.num_nodes(dst_ntype)
            
        # collect logits and destination node ids for all passing edge types
        logits_list = []
        dst_list = []
        lengths = []
        etypes = []
        for etype in valid_etypes:
            _, _, dst_ntype = to_canonical_etype(etype)
            # extract raw attention logits and project
            att_logits = g.edges[etype].data["scalar_msg"][:, self.s_message_dim:]
            att_logits = self.att_weight_projection[etype](att_logits)
            # get destination node indices
            _, dst = g.edges(etype=etype)

            # convert dst to indicies unique to that ntype
            dst = dst + ntype_mins[dst_ntype]

            logits_list.append(att_logits)
            dst_list.append(dst)
            lengths.append(att_logits.size(0))

        # nothing to do if no logits collected
        if not logits_list:
            return
        # flatten across all edge types
        flat_logits = torch.cat(logits_list, dim=0)
        flat_dst = torch.cat(dst_list, dim=0)
        # normalize logits per destination node over all edge types
        norm_flat = scatter_softmax(flat_logits, flat_dst, dim=0)
        # split normalized logits back into original per-type segments
        split_norms = torch.split(norm_flat, lengths, dim=0)
        for etype, norms in zip(valid_etypes, split_norms):
            # split into scalar and vector head weights
            s_att, v_att = torch.split(norms, [self.n_heads, self.n_heads], dim=1)
            # expand weights to full feature dimensions
            s_weights = s_att.repeat_interleave(self.s_feats_per_head, dim=1)
            v_weights = v_att.repeat_interleave(self.v_feats_per_head, dim=1)
            # apply to scalar messages
            scalar_msg = g.edges[etype].data["scalar_msg"][:, : self.s_message_dim]
            g.edges[etype].data["scalar_msg"] = scalar_msg * s_weights
            # apply to vector messages
            vec_msg = g.edges[etype].data["vec_msg"]
            g.edges[etype].data["vec_msg"] = vec_msg * v_weights.unsqueeze(-1)

    def compute_att_weights_per_type(self, g: dgl.DGLHeteroGraph, passing_edges: List[str]):
        """
        Per-edge-type attention normalization: runs edge_softmax separately on each relation type.
        """

        # collect attention logits per type
        per_type_logits = {}
        for etype in passing_edges:
            if etype not in g.etypes or g.num_edges(etype) == 0:
                continue
            # Split out raw attention logits
            att_logits = g.edges[etype].data["scalar_msg"][:, self.s_message_dim:]
            # Project logits
            att_logits = self.att_weight_projection[etype](att_logits)
            per_type_logits[etype] = att_logits

        # normalize logits 
        norm_logits_per_type = edge_softmax(g, per_type_logits)

        # multiply messages by their logits
        for etype in passing_edges:
            canonical_etype = to_canonical_etype(etype)

            # Split head weights and expand to feature dimensions
            s_att, v_att = torch.split(norm_logits_per_type[canonical_etype], [self.n_heads, self.n_heads], dim=1)
            s_att_weights = s_att.repeat_interleave(self.s_feats_per_head, dim=1)
            v_att_weights = v_att.repeat_interleave(self.v_feats_per_head, dim=1)

            # Apply to scalar and vector messages
            scalar_msg = g.edges[etype].data["scalar_msg"][:, : self.s_message_dim]
            g.edges[etype].data["scalar_msg"] = scalar_msg * s_att_weights
            orig_vec = g.edges[etype].data["vec_msg"]
            g.edges[etype].data["vec_msg"] = orig_vec * v_att_weights.unsqueeze(-1)

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
