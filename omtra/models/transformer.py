import torch
import dgl
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch_scatter
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import List, Dict, Optional
from einops.layers.torch import Rearrange
from einops import rearrange
from omtra.data.graph.layout import GraphLayout
from omtra.utils.graph import g_local_scope
from omtra.models.gvp import _rbf

from omtra.models.layers import Mlp
from omtra.models.dit import QKNormTransformerEncoderLayer


class AttentionPairBias(nn.Module):
    """Attention pair bias layer."""

    def __init__(
        self,
        c_s: int,
        c_z: Optional[int] = None,
        num_heads: Optional[int] = None,
        inf: float = 1e6,
        compute_pair_bias: bool = True,
    ) -> None:
        """Initialize the attention pair bias layer.

        Parameters
        ----------
        c_s : int
            The input sequence dimension.
        c_z : int
            The input pairwise dimension.
        num_heads : int
            The number of heads.
        inf : float, optional
            The inf value, by default 1e6

        """
        super().__init__()

        assert c_s % num_heads == 0

        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf

        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)

        self.compute_pair_bias = compute_pair_bias
        if compute_pair_bias:
            self.proj_z = nn.Sequential(
                nn.LayerNorm(c_z),
                nn.Linear(c_z, num_heads, bias=False),
                Rearrange("b ... h -> b h ..."),
            )
        else:
            self.proj_z = Rearrange("b ... h -> b h ...")

        self.proj_o = nn.Linear(c_s, c_s, bias=False)
        with torch.no_grad():
            self.proj_o.weight.fill_(0.0)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        k_in: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        s : torch.Tensor
            The input sequence tensor (B, S, D)
        z : torch.Tensor
            The input pairwise tensor or bias (B, N, N, D)
        mask : torch.Tensor
            The nodewise tensor mask (B, N)

        Returns
        -------
        torch.Tensor
            The output sequence tensor.

        """
        B = s.shape[0]

        # Compute projections
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

        bias = self.proj_z(z)

        g = self.proj_g(s).sigmoid()

        # with torch.autocast("cuda", enabled=False):
        # Compute attention weights
        attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
        attn = attn / (self.head_dim**0.5) + bias.float()
        attn = attn + (1 - mask[:, None, None].float()) * -self.inf
        # attn = attn + (1 - mask.unsqueeze(1).float()) * -self.inf
        attn = attn.softmax(dim=-1)

        # Compute output
        o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)

        o = o.reshape(B, -1, self.c_s)
        o = self.proj_o(g * o)

        return o


class PairTransformerLayer(nn.Module):
    """
    Custom transformer layer that can be interleaved with standard TransformerEncoderLayer.
    """
    def __init__(self, hidden_dim, pair_dim, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attn = AttentionPairBias(
            c_s=hidden_dim,
            c_z=pair_dim,
            num_heads=num_heads,
            compute_pair_bias=True
            )
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )
        
    def forward(self, x, p, mask):

        _x = self.norm1(x)

        x = x + self.attn(_x, p, mask, _x)

        x = x + self.mlp(self.norm2(x))
        return x


class AtomOffsetEncoder(nn.Module):
    def __init__(self, catompair: int):
        super().__init__()
        # LinearNoBias layers
        self.lin_d   = nn.Linear(3, catompair, bias=False)  # for d_lm (R^3 -> R^C)
        self.lin_inv = nn.Linear(1, catompair, bias=False)  # for 1/(1+||d||^2)
    
    def forward(self, ref_pos: torch.Tensor) -> torch.Tensor:
        """
        ref_pos: (B, N, 3)
        returns:
            p_lm: (B, N, N, Catompair)
        """
        # idea taken from AF3 AtomAttentionEncoder, aglorithm 5 in AF3 paper

        # (2) d_lm = f_l^ref_pos - f_m^ref_pos  -> shape (B, N, N, 3)
        d_lm = ref_pos.unsqueeze(2) - ref_pos.unsqueeze(1)              # (B, N, N, 3)

        # (4) p_lm = LinearNoBias(d_lm)
        p_lm = self.lin_d(d_lm)                    # (B, N, N, C)

        # (5) p_lm += LinearNoBias( 1 / (1 + ||d_lm||^2) ) * v_lm
        dist2 = (d_lm ** 2).sum(dim=-1, keepdim=True)         # (B, N, N, 1)
        inv   = 1.0 / (1.0 + dist2)
        p_lm = p_lm + self.lin_inv(inv)

        return p_lm

class LigandPairBiasEmbedder(nn.Module):
    """Computes ligand pair bias and applies the pair-biased attention layer."""

    def __init__(
        self,
        hidden_dim: int,
        pair_dim: int,
        num_heads: int,
        rbf_count: int = 24,
        rbf_d_min: float = 0.0,
        rbf_d_max: float = 10.0,
    ) -> None:
        super().__init__()
        self.pair_dim = pair_dim
        self.rbf_count = rbf_count
        self.rbf_d_min = rbf_d_min
        self.rbf_d_max = rbf_d_max
        # self.atom_offset_encoder = AtomOffsetEncoder(catompair=pair_dim)

        # self.rbf_proj = nn.Sequential(
        #     nn.Linear(pair_dim*2 + rbf_count, pair_dim*2, bias=False),
        #     nn.SiLU(),
        #     nn.Linear(pair_dim*2, pair_dim, bias=False),
        #     nn.SiLU(),
        #     nn.Linear(pair_dim, pair_dim, bias=False),
        #     nn.LayerNorm(pair_dim)
        # )
        self.rbf_proj = nn.Sequential(
            nn.Linear(pair_dim + rbf_count, pair_dim, bias=False),
            nn.LayerNorm(pair_dim),
        )

        # self.s_i_proj = nn.Linear(hidden_dim, pair_dim, bias=False)
        # self.s_j_proj = nn.Linear(hidden_dim, pair_dim, bias=False)

        self.layer = PairTransformerLayer(
            hidden_dim=hidden_dim,
            pair_dim=pair_dim,
            num_heads=num_heads,
        )

    def forward(
        self,
        lig_feats: torch.Tensor,
        lig_mask: torch.Tensor,
        lig_pos: torch.Tensor,
        pair_feats: torch.Tensor,
    ) -> torch.Tensor:
        if lig_feats.size(1) == 0:
            return lig_feats

        device = lig_feats.device
        lig_pos = lig_pos.to(device)
        lig_mask = lig_mask.to(device)

        # proj_inputs = [pair_feats]

        pair_bias = pair_feats

        # inject scalar feature contributions to pair bias
        # single_projection = (self.s_i_proj(lig_feats).unsqueeze(2) + self.s_j_proj(lig_feats).unsqueeze(1))
        # proj_inputs.append(single_projection)

        # inject pairwise distances into pair bias via RBFs
        pair_dists = torch.cdist(lig_pos, lig_pos, p=2.0)
        offset_bias = _rbf(
            pair_dists,
            D_min=self.rbf_d_min,
            D_max=self.rbf_d_max,
            D_count=self.rbf_count,
        )
        # offset_bias = self.atom_offset_encoder(lig_pos)
        # proj_inputs.append(offset_bias)
        # rbf_proj_input = torch.cat(proj_inputs, dim=-1)
        rbf_proj_input = torch.cat((pair_bias, offset_bias), dim=-1)
        # pair_bias = pair_bias + self.rbf_proj(rbf_proj_input)
        pair_bias = self.rbf_proj(rbf_proj_input)

        single_feats = self.layer(lig_feats, pair_bias, lig_mask)

        return single_feats, pair_bias


class TransformerWrapper(nn.Module):
    """
    - Concatenate scalar + flattened vector features per node
    - Pre-MLP per node type to capture node-type specific info
    - Pack all node types into a shared transformer for cross-type attention
    - Map d_model back to original size S
    """
    def __init__(self,
                 node_types: List[str],
                 n_hidden_scalars: int,
                 n_vec_channels: int,
                 d_model: int = 256,
                 pair_dim: int = 32,
                 n_layers: int = 4,
                 n_heads: int = 8,
                 dim_ff: int | None = None,
                 dropout: float = 0.1,
                 use_residual: bool = True,
                 use_qk_norm: bool = False,
                 ):
        super().__init__()
        self.ntype_order = list(node_types)
        self.S = n_hidden_scalars
        self.C = n_vec_channels
        self.d_model = d_model
        self.use_residual = use_residual
        self.pair_dim = pair_dim
        self.use_qk_norm = use_qk_norm

        # in_dim = n_hidden_scalars + 3 * n_vec_channels
        if dim_ff is None:
            dim_ff = 4 * d_model
        
        # # pre-MLP per node type
        # self.pre_mlp = NodeTypeMLP(self.ntype_order, in_dim=in_dim, d_model=d_model, dropout=dropout)

        # Create interleaved layers
        self.ligand_embedder = LigandPairBiasEmbedder(
            hidden_dim=d_model,
            pair_dim=pair_dim,
            num_heads=n_heads,
        )
        self.layers = nn.ModuleList()

        # Add standard TransformerEncoderLayers
        for _ in range(n_layers):
            if use_qk_norm:
                layer = QKNormTransformerEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                )
            else:
                layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_ff,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
                )
            self.layers.append(layer)

        # map scalars + coords to d_model, linear transformation of coords
        self.in_proj = nn.Sequential(
            nn.Linear(self.S + 3 + n_vec_channels*3, d_model, bias=False),
            # nn.LayerNorm(self.S + 3),
        )

        # map d_model back to scalars only
        self.out_proj = nn.Linear(d_model, self.S, bias=True)

        self.trfmr_node_feat_key = "temp_key"
        self.trfmr_pair_feat_key = "temp_pair_key"


    @g_local_scope
    def forward(
        self,
        g,
        scalar_feats: Dict[str, torch.Tensor],
        vec_feats: Dict[str, torch.Tensor],
        coord_feats: Dict[str, torch.Tensor],
        edge_feats=None,
        x_diff=None,
        d=None,
        **kwargs,
    ):
        # Concatenate scalars with coordinates
        for ntype in scalar_feats:
            scal = scalar_feats.get(ntype)
            coords = coord_feats.get(ntype)
            vecs = vec_feats.get(ntype) # has shape (N, C, 3)

            # flatten vec feats
            # TODO: is there a better way to use einops for this? 
            vecs = rearrange(vecs, 'N C D -> N (C D)')  # (N, 3*C)

            # TODO: projection doesn't need to happen here. it can happen
            # after we pack all node types together.
            feat_input = torch.cat([scal, coords, vecs], dim=-1)  # (N, S + 3)
            out  = self.in_proj(feat_input)   # (N, d_model)
            
            # add input feature to graph
            g.nodes[ntype].data[self.trfmr_node_feat_key] = out

        # insert lig_to_lig feats into graph
        # TODO: possibly shouldn't hard-code this bc we will also have npnde_to_npnde feats later
        for etype in edge_feats.keys():
            g.edges[etype].data[self.trfmr_pair_feat_key] = edge_feats[etype]

        layout, padded_node_feats, attention_masks, padded_edge_feats = GraphLayout.layout_and_pad(
            g,
            allowed_feat_names=[self.trfmr_node_feat_key, self.trfmr_pair_feat_key, 'x_t'],
        )

        # do attention with pair bias for ligand nodes        
        init_lig_feats, lig_pair_feats = self.ligand_embedder(
            lig_feats=padded_node_feats['lig'][self.trfmr_node_feat_key],
            lig_mask=attention_masks['lig'],
            lig_pos=padded_node_feats['lig']['x_t'],
            pair_feats=padded_edge_feats['lig_to_lig'][self.trfmr_pair_feat_key],
        )
        padded_node_feats['lig'][self.trfmr_node_feat_key] = init_lig_feats
        padded_edge_feats['lig_to_lig'][self.trfmr_pair_feat_key] = lig_pair_feats


        # Pack all node types to one sequence
        X_list, M_list, sizes = [], [], []
        for ntype in self.ntype_order:
            bucket = padded_node_feats.get(ntype)
            if bucket is None or self.trfmr_node_feat_key not in bucket:
                sizes.append(0)
                continue
            X = bucket[self.trfmr_node_feat_key]  # (B, n_max, d_model)
            if self.use_qk_norm:
                M = attention_masks[ntype].to(torch.bool)
            else:
                M = (~attention_masks[ntype]).to(torch.bool)
            X_list.append(X)
            M_list.append(M)
            sizes.append(X.size(1))


        X_all = torch.cat(X_list, dim=1) # (B, n_all, d_model)
        M_all = torch.cat(M_list, dim=1)

        Y_all = X_all

        for layer in self.layers:
            Y_all = layer(Y_all, src_key_padding_mask=M_all)

        # back to per-ntype padded tensors
        offset = 0
        for ntype, nmax in zip(self.ntype_order, sizes):
            if nmax == 0:
                continue
            padded_node_feats[ntype][self.trfmr_node_feat_key] = Y_all[:, offset:offset + nmax, :]
            offset += nmax

        # back to DGL graph
        layout.padded_sequence_to_graph(
            g, 
            padded_node_feats, 
            attention_masks=attention_masks, 
            padded_edge_feats=padded_edge_feats, 
            inplace=True,
        )
        
        out_scalars: Dict[str, torch.Tensor] = {}
        for ntype, H_old in scalar_feats.items():
            Y_ntype = g.nodes[ntype].data.get(self.trfmr_node_feat_key)
            if Y_ntype is None:
                # pass through if this ntype wasn't present
                out_scalars[ntype] = H_old
                continue

            H_new = self.out_proj(Y_ntype)  # (N, S)
            if self.use_residual and H_new.shape == H_old.shape:
                H_new = H_new + H_old
            out_scalars[ntype] = H_new

        edge_feats_out = {}
        # for etype in edge_feats:
        #     edge_feats_out[etype] = g.edges[etype].data[self.trfmr_pair_feat_key]

        return out_scalars, edge_feats_out
