import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch_scatter
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import List, Dict, Optional
from einops.layers.torch import Rearrange
from omtra.data.graph.layout import GraphLayout
from omtra.utils.graph import g_local_scope
from omtra.models.gvp import _rbf

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    

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
                 use_residual: bool = True):
        super().__init__()
        self.ntype_order = list(node_types)
        self.S = n_hidden_scalars
        self.C = n_vec_channels
        self.d_model = d_model
        self.use_residual = use_residual
        self.pair_dim = pair_dim

        # in_dim = n_hidden_scalars + 3 * n_vec_channels
        if dim_ff is None:
            dim_ff = 4 * d_model
        
        # # pre-MLP per node type
        # self.pre_mlp = NodeTypeMLP(self.ntype_order, in_dim=in_dim, d_model=d_model, dropout=dropout)

        # Create interleaved layers
        self.layers = nn.ModuleList()
        
        # Add one CustomTransformerLayer first
        custom_layer = PairTransformerLayer(
            hidden_dim=d_model,
            pair_dim=pair_dim,
            num_heads=n_heads,
        )
        self.layers.append(custom_layer)
        
        # Add standard TransformerEncoderLayers
        for i in range(n_layers):
            std_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
            )
            self.layers.append(std_layer)

        # map scalars + coords to d_model, linear transformation of coords
        self.in_proj = nn.Sequential(
            nn.Linear(self.S + 3, d_model, bias=False),
            # nn.LayerNorm(self.S + 3),
        )

        # map d_model back to scalars only
        self.out_proj = nn.Linear(d_model, self.S, bias=True)

        self.trfmr_feat_key = "temp_key"


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
        for ntype in self.ntype_order:
            scal = scalar_feats.get(ntype)
            if scal is None or scal.numel() == 0:
                continue
            coords = coord_feats.get(ntype)
            if coords is None or coords.numel() == 0:
                coords = scal.new_zeros((scal.shape[0], 3))

            # TODO: projection doesn't need to happen here. it can happen
            # after we pack all node types together.
            feat_input = torch.cat([scal, coords], dim=-1)  # (N, S + 3)
            out  = self.in_proj(feat_input)   # (N, d_model)
            
            # add input feature to graph
            g.nodes[ntype].data[self.trfmr_feat_key] = out

        layout, padded_node_feats, attention_masks = GraphLayout.layout_and_pad(g)

        # do attention with pair bias for ligand nodes
        lig_feats = padded_node_feats['lig'][self.trfmr_feat_key]
        lig_mask = attention_masks['lig'].to(lig_feats.device)
        B, S, D = lig_feats.shape
        pair_mask = lig_mask.unsqueeze(1) & lig_mask.unsqueeze(2)

        # get dense edge features
        lig_edge_feats = edge_feats.get("lig_to_lig")

        pair_bias = lig_feats.new_zeros(
            (lig_feats.size(0), S, S, self.pair_dim)
        )

        src, dst = g.edges(etype="lig_to_lig")

        src = src.to(lig_feats.device)
        dst = dst.to(lig_feats.device)

        node_batch = layout.node_batch_idxs['lig'].to(
            lig_feats.device
        )
        node_offsets = layout.node_offsets['lig'].to(
            lig_feats.device
        )

        node_ids = torch.arange(
            node_batch.shape[0],
            device=lig_feats.device,
            dtype=torch.long,
        )
        node_pos = node_ids - node_offsets[node_batch]

        src_pos = node_pos[src]
        dst_pos = node_pos[dst]
        edge_batch = node_batch[src]

        pair_bias[edge_batch, src_pos, dst_pos] = lig_edge_feats
        pair_bias[edge_batch, dst_pos, src_pos] = lig_edge_feats

        # compute pairwise distances
        lig_pos = padded_node_feats['lig']['x_t']
        pair_dists = torch.cdist(lig_pos, lig_pos, p=2.0)  # (N_lig, N_lig)


        # embed pairwise distances and add to pair bias
        pair_bias = pair_bias + _rbf(pair_dists, D_min=0.0, D_max=12.0, D_count=self.pair_dim)


        # apply transformer layer with pair bias
        padded_node_feats['lig'][self.trfmr_feat_key] = self.layers[0](
            lig_feats, pair_bias, lig_mask
        )

        # TODO: the attention with pair bias on ligand nodes doesn't need to be
        # just the "first of N layers"
        # this can be a special pre-encoding step before we pass through self.layers
        # specifically i think we should refactor so that attention with pair bias on ligand nodes
        # is its own separate class, so here in TransformerWrapper.forward we just do something like
        # padded_node_feats['lig'][self.trfmr_feat_key] = self.ligand_embedder(...)

        # end - attention with pair bias for ligand nodes


        # Pack all node types to one sequence
        X_list, M_list, sizes = [], [], []
        for ntype in self.ntype_order:
            bucket = padded_node_feats.get(ntype)
            if bucket is None or self.trfmr_feat_key not in bucket:
                sizes.append(0)
                continue
            X = bucket[self.trfmr_feat_key]  # (B, n_max, d_model)
            M = (~attention_masks[ntype]).to(torch.bool)
            X_list.append(X)
            M_list.append(M)
            sizes.append(X.size(1))


        X_all = torch.cat(X_list, dim=1) # (B, n_all, d_model)
        M_all = torch.cat(M_list, dim=1)

        Y_all = X_all
        start_layer_idx = 1

        for layer in self.layers[start_layer_idx:]:
            Y_all = layer(Y_all, src_key_padding_mask=M_all)

        # back to per-ntype padded tensors
        offset = 0
        for ntype, nmax in zip(self.ntype_order, sizes):
            if nmax == 0:
                continue
            padded_node_feats[ntype][self.trfmr_feat_key] = Y_all[:, offset:offset + nmax, :]
            offset += nmax

        # back to DGL graph
        layout.padded_sequence_to_graph(
            g, padded_node_feats, attention_masks=attention_masks, inplace=True
        )

        # back to original size
        out_scalars: Dict[str, torch.Tensor] = {}

        for ntype, H_old in scalar_feats.items():
            Y_ntype = g.nodes[ntype].data.get(self.trfmr_feat_key)
            if Y_ntype is None:
                # pass through if this ntype wasn't present
                out_scalars[ntype] = H_old
                continue

            H_new = self.out_proj(Y_ntype)  # (N, S)
            if self.use_residual and H_new.shape == H_old.shape:
                H_new = H_new + H_old
            out_scalars[ntype] = H_new

        return out_scalars, vec_feats
