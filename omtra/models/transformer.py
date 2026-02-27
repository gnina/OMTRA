import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
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
from omtra.models.dit import DiTLayer


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
        # attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
        # attn = attn / (self.head_dim**0.5) + bias.float()
        # attn = attn + (1 - mask[:, None, None].float()) * -self.inf
        # # attn = attn + (1 - mask.unsqueeze(1).float()) * -self.inf
        # attn = attn.softmax(dim=-1)

        # # Compute output
        # o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype) # has shape (B, N, n_heads, head_dim)
        
        combined_attn_mask = bias.float() + (1 - mask[:, None, None, :].float()) * -self.inf
        
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), # SDPA expects (B, n_heads, N, head_dim)
            attn_mask=combined_attn_mask,
        ).transpose(1, 2)

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


class AllNodePairBiasEmbedder(nn.Module):
    """Computes pair features for all node types using node features + RBF distance embeddings.
    
    This layer constructs pair features from:
    1. Projected single (node) features: s_i + s_j
    2. RBF embeddings of pairwise distances
    3. Optionally splices in pre-computed ligand pair features
    """

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

        # Project single features to pair space
        self.s_i_proj = nn.Linear(hidden_dim, pair_dim, bias=False)
        self.s_j_proj = nn.Linear(hidden_dim, pair_dim, bias=False)

        self.e_proj = nn.Linear(pair_dim, pair_dim, bias=False)

        self.rbf_proj = nn.Linear(rbf_count, pair_dim, bias=False)

        self.combine = nn.Sequential(
            nn.Linear(pair_dim * 4, pair_dim),
            nn.SiLU(),
            nn.Linear(pair_dim, pair_dim),
            nn.LayerNorm(pair_dim)
        )

        # Project concatenated pair features (single contributions + RBF) to pair_dim
        # self.pair_proj = nn.Sequential(
        #     nn.Linear(pair_dim + rbf_count, pair_dim, bias=False),
        #     nn.LayerNorm(pair_dim),
        # )

        self.layer = PairTransformerLayer(
            hidden_dim=hidden_dim,
            pair_dim=pair_dim,
            num_heads=num_heads,
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,   # added for new pair bias
        node_pos: torch.Tensor,
        node_mask: torch.Tensor,
        # lig_pair_feats: Optional[torch.Tensor] = None,
        # lig_size: int = 0,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        node_feats : torch.Tensor
            All node features concatenated (B, N_all, hidden_dim)
        node_pos : torch.Tensor
            All node positions concatenated (B, N_all, 3)
        node_mask : torch.Tensor
            Attention mask for all nodes (B, N_all), 1 for valid, 0 for padding
        lig_pair_feats : torch.Tensor, optional
            Pre-computed ligand pair features (B, N_lig, N_lig, pair_dim)
        lig_size : int
            Number of ligand nodes (used to splice in lig_pair_feats)

        Returns
        -------
        torch.Tensor
            Updated node features (B, N_all, hidden_dim)
        """
        if node_feats.size(1) == 0:
            return node_feats

        device = node_feats.device
        B, N_all, _ = node_feats.shape

        # Compute pair features from single node features
        s_i = self.s_i_proj(node_feats).unsqueeze(2)  # (B, N, 1, pair_dim)
        s_j = self.s_j_proj(node_feats).unsqueeze(1)  # (B, 1, N, pair_dim)
        #pair_from_single = s_i + s_j  # (B, N, N, pair_dim)

        # Compute pairwise distances and RBF embeddings
        pair_dists = torch.cdist(node_pos, node_pos, p=2.0)  # (B, N, N)
        dist_rbf = _rbf(
            pair_dists,
            D_min=self.rbf_d_min,
            D_max=self.rbf_d_max,
            D_count=self.rbf_count,
        )  # (B, N, N, rbf_count)

        d = self.rbf_proj(dist_rbf)

        e = self.e_proj(edge_feats)

        pair = torch.cat([
            s_i.expand_as(e),
            s_j.expand_as(e),
            d,
            e
        ], dim=-1)

        pair_feats = self.combine(pair)

        # # Concatenate and project to get pair features
        # pair_input = torch.cat([pair_from_single, dist_rbf], dim=-1)
        # pair_feats = self.pair_proj(pair_input)  # (B, N, N, pair_dim)

        # # Add pre-computed ligand pair features if provided (additive to preserve gradients)
        # if lig_pair_feats is not None and lig_size > 0:
        #     # pair_feats = pair_feats.clone()
        #     pair_feats[:, :lig_size, :lig_size, :] = pair_feats[:, :lig_size, :lig_size, :] + lig_pair_feats

        # Apply pair-biased attention
        out_feats = self.layer(node_feats, pair_feats, node_mask)

        return out_feats


class TransformerWrapper(nn.Module):
    """
    - Concatenate scalar + flattened vector features per node
    - Pre-MLP per node type to capture node-type specific info
    - Pack all node types into a shared transformer for cross-type attention
    - Map d_model back to original size S
    """
    def __init__(self,
                 node_types: List[str],
                 edge_types: List[str],     # For all-to-all pair bias
                 n_hidden_scalars: int,
                 n_vec_channels: int,
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
        self.etype_order = list(edge_types)
        self.S = n_hidden_scalars
        self.C = n_vec_channels
        self.d_model = n_hidden_scalars
        self.use_residual = use_residual
        self.pair_dim = pair_dim
        self.use_qk_norm = use_qk_norm

        # in_dim = n_hidden_scalars + 3 * n_vec_channels
        if dim_ff is None:
            dim_ff = 4 * self.d_model
        
        # # pre-MLP per node type
        # self.pre_mlp = NodeTypeMLP(self.ntype_order, in_dim=in_dim, d_model=d_model, dropout=dropout)

        # Create interleaved layers
        self.ligand_embedder = LigandPairBiasEmbedder(
            hidden_dim=self.d_model,
            pair_dim=pair_dim,
            num_heads=n_heads,
        )
        
        # All-node pair bias embedder (applied after ligand embedder, before DiT layers)
        self.all_node_embedder = AllNodePairBiasEmbedder(
            hidden_dim=self.d_model,
            pair_dim=pair_dim,
            num_heads=n_heads,
        )
        
        self.layers = nn.ModuleList()

        # Add standard TransformerEncoderLayers
        for _ in range(n_layers):
            if use_qk_norm:
                layer = DiTLayer(
                    d_model=self.d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                )
            else:
                layer = TransformerEncoderLayer(
                d_model=self.d_model,
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
            nn.Linear(self.S + 3 + n_vec_channels*3, self.d_model, bias=False),
            # nn.LayerNorm(self.S + 3),
        )

        # map d_model back to scalars only
        self.out_proj = nn.Linear(self.d_model, self.S, bias=True)

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
        global_conditioning=None,
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
        # init_lig_feats, lig_pair_feats = self.ligand_embedder(
        #     lig_feats=padded_node_feats['lig'][self.trfmr_node_feat_key],
        #     lig_mask=attention_masks['lig'],
        #     lig_pos=padded_node_feats['lig']['x_t'],
        #     pair_feats=padded_edge_feats['lig_to_lig'][self.trfmr_pair_feat_key],
        # )
        # padded_node_feats['lig'][self.trfmr_node_feat_key] = init_lig_feats
        # padded_edge_feats['lig_to_lig'][self.trfmr_pair_feat_key] = lig_pair_feats


        # Pack all node types to one sequence
        X_list, M_list, X_sizes = [], [], []
        for ntype in self.ntype_order:
            bucket = padded_node_feats.get(ntype)
            if bucket is None or self.trfmr_node_feat_key not in bucket:
                X_sizes.append(0)
                continue
            X = bucket[self.trfmr_node_feat_key]  # (B, n_max, d_model)
            if self.use_qk_norm:
                M = attention_masks[ntype].to(torch.bool)
            else:
                M = (~attention_masks[ntype]).to(torch.bool)
            X_list.append(X)
            M_list.append(M)
            X_sizes.append(X.size(1))

        X_all = torch.cat(X_list, dim=1) # (B, n_all, d_model)
        M_all = torch.cat(M_list, dim=1)

        # Pack all edge types into one (B, n_all, n_all, pair_dim) matrix
        # Compute per-node type offsets using X_sizes
        ntype_to_offset = {}
        _off = 0
        for ntype, nmax in zip(self.ntype_order, X_sizes):
            ntype_to_offset[ntype] = _off
            _off += nmax
        n_all = _off
        
        E_all = torch.zeros(                                # (B, n_all, n_all, pair_dim)
            X_all.shape[0], n_all, n_all, self.pair_dim,
            device=X_all.device, dtype=X_all.dtype,
        )
        # Map etype string -> (src_type, dst_type) from the graph's canonical edge types
        etype_to_srcdst = {etype: (src, dst) for src, etype, dst in g.canonical_etypes}
        for etype, e_bucket in padded_edge_feats.items():
            if self.trfmr_pair_feat_key not in e_bucket:
                continue
            srcdst = etype_to_srcdst.get(etype)
            if srcdst is None:
                continue
            src_type, dst_type = srcdst
            src_off = ntype_to_offset.get(src_type)
            dst_off = ntype_to_offset.get(dst_type)
            if src_off is None or dst_off is None:
                continue
            e_feats = e_bucket[self.trfmr_pair_feat_key]  # (B, n_src, n_dst, pair_dim)
            n_src, n_dst = e_feats.size(1), e_feats.size(2)     # number of src and dst nodes in edges of type etype
            E_all[:, src_off:src_off + n_src, dst_off:dst_off + n_dst, :] = e_feats # slice into edge feat matrix

        # Concatenate positions for all node types
        pos_list = []
        assert self.ntype_order[0] == 'lig', "First node type must be 'lig' or else all pair embedding injects pair features incorrectly"
        for ntype in self.ntype_order:
            bucket = padded_node_feats.get(ntype)
            if bucket is None or 'x_t' not in bucket:
                continue
            pos_list.append(bucket['x_t'])  # (B, n_max, 3)
        all_pos = torch.cat(pos_list, dim=1)  # (B, n_all, 3)

        # Get ligand pair features and size for splicing
        # lig_pair_feats = padded_edge_feats.get('lig_to_lig', {}).get(self.trfmr_pair_feat_key, None)
        # lig_size = sizes[self.ntype_order.index('lig')] if 'lig' in self.ntype_order else 0

        # Apply all-node pair bias attention (reuses ligand pair features)
        # Note: all_node_embedder expects mask where 1=valid, but we may have inverted mask
        # depending on use_qk_norm. PairTransformerLayer expects 1=valid.
        if self.use_qk_norm:
            all_node_mask = M_all  # already 1=valid for qk_norm path
        else:
            all_node_mask = ~M_all  # invert back: True (invalid) -> 0, False (valid) -> 1
        
        Y_all = self.all_node_embedder(
            node_feats=X_all,
            edge_feats=E_all, 
            node_pos=all_pos,
            node_mask=all_node_mask.float(),
            #lig_pair_feats=lig_pair_feats,
            #lig_size=lig_size,
        )

        for layer in self.layers:
            # TODO: remove this once we prove that the qknorm + adaln works better than vanilla transformer
            if self.use_qk_norm:
                kwargs = {
                    'src_key_padding_mask': M_all,
                    'c': global_conditioning,
                }
            else:
                kwargs = {
                    'src_key_padding_mask': M_all,
                }
            Y_all = layer(Y_all, **kwargs)

        # back to per-ntype padded tensors
        offset = 0
        for ntype, nmax in zip(self.ntype_order, X_sizes):
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
