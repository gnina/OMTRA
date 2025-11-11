import torch
import torch.nn as nn
import torch.nn.functional as F
from omtra.models.layers import Mlp
from omtra.models.adaln import AdaLNWeightGenerator, modulate


class QKNormMultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, init_scale_factor: float = 11.4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)


        # init scale factor of 11.4 is obtained from 97.5th percentile of training data sequence lengths
        # as described in https://arxiv.org/abs/2010.04245
        # specifically in our case it is based on the distribution of number of atoms in the pharmit dataset
        # we generate the scale factor on-the-fly based on global conditioning information (time+task embedding)
        # but we initialize the scale-generating MLP so that it starts out producing this value

        self.qk_scale_generator = AdaLNWeightGenerator(
            d_model=d_model,
            out_dim=1,
            params=['qk_scale'],
        )
        nn.init.constant_(self.qk_scale_generator.mlp[1].bias, init_scale_factor)

    def _shape(self, x, B, L):
        # (B, L, D) -> (B, nH, L, dH)
        return x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)


    def qk_norm(self, q, k, c, eps=1e-6):
        # q, k: (B, nH, L, dH)
        # c: (B, D)

        qk_scale = self.qk_scale_generator(c)['qk_scale']  # (B, 1)
        qk_scale = qk_scale.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)

        q_norm = q / (q.norm(dim=-1, keepdim=True) + eps)
        k_norm = k / (k.norm(dim=-1, keepdim=True) + eps)

        q_norm = q_norm * qk_scale # multiply queries by the scale factor
        # this is matheatically equivalent to scaling the dot products by qk_scale

        return q_norm, k_norm

    def forward(self, x, c, key_padding_mask=None, attn_mask=None):
        """
        x: (B, L, D) if batch_first, else (L, B, D)
        c: (B, D) conditioning tensor for QK-norm 
        key_padding_mask: (B, L) with True for PADs
        attn_mask: (L, L) or (B * nH, L, L) float or bool
        """

        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._shape(q, B, L)
        k = self._shape(k, B, L)
        v = self._shape(v, B, L)  # (B, nH, L, dH)

        # QK-normalization
        # q, k = self.qk_norm(q, k, c)

        # Convert key_padding_mask -> an additive attn_mask if present
        # F.scaled_dot_product_attention supports attn_mask (broadcasted) and no key_padding_mask,
        # so we fold padding into attn_mask.
        if key_padding_mask is not None:
            # key_padding_mask: (B, L), True for positions to mask
            # We want a mask of shape (B, 1, 1, L) broadcastable to (B, nH, L, L)
            pad_mask = key_padding_mask[:, None, None, :]  # (B, 1, 1, L)
            # Use -inf for masked positions
            pad_mask = pad_mask.masked_fill(pad_mask, float("-inf"))

            if attn_mask is None:
                attn_mask = pad_mask
            else:
                # broadcast-friendly sum if user also supplied per-head/per-token mask
                attn_mask = attn_mask + pad_mask

        # scaled_dot_product_attention expects (B, nH, L, dH)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            # scale=1.0,
        )  # (B, nH, L, dH)

        # back to (B, L, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        attn_output = self.out_proj(attn_output)

        return attn_output
    

class QKNormTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1,
        batch_first=True,
        mlp_ratio=4.0,
    ):
        super().__init__()

        if batch_first != True:
            raise NotImplementedError("Only batch_first=True is supported currently.")

        self.self_attn = QKNormMultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Feedforward
        hidden_features = int(d_model * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=d_model, 
            hidden_features=hidden_features, 
            drop=0.0,
            act_layer=approx_gelu,
        )

        # Norms
        self.adaln_generator = AdaLNWeightGenerator(d_model)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, x, c, src_mask=None, src_key_padding_mask=None):
        """
        src: (B, L, D) if batch_first else (L, B, D)
        c: (B, D) conditioning tensor for AdaLN
        src_mask: same semantics as in nn.TransformerEncoderLayer
        src_key_padding_mask: (B, L)
        """
        adaln_params = self.adaln_generator(c)  # Dict[str, Tensor] each of shape (B, d_model)
        for key in adaln_params:
            adaln_params[key] = adaln_params[key].unsqueeze(1)  # (B, 1, d_model) for broadcasting

        _x = self.norm1(x) * (1 + adaln_params['scale_att']) + adaln_params['shift_att']

        x = x + self.self_attn(
            _x,
            key_padding_mask=src_key_padding_mask,
            attn_mask=src_mask,
            c=c,
        ) * adaln_params['gate_att']


        _x = self.norm2(x) * (1 + adaln_params['scale_ff']) + adaln_params['shift_ff']

        x = x + self.mlp(_x) * adaln_params['gate_ff']

        return x
    
# TODO: need to use an adaln layer for the QK-norm scaling parameter as well