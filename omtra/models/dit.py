import torch
import torch.nn as nn
import torch.nn.functional as F
from omtra.models.layers import Mlp


class QKNormMultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
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

    def _shape(self, x, B, L):
        # (B, L, D) -> (B, nH, L, dH)
        return x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def qk_norm(q, k, eps=1e-6):
        # q, k: (B, nH, L, dH)
        q_norm = q / (q.norm(dim=-1, keepdim=True) + eps)
        k_norm = k / (k.norm(dim=-1, keepdim=True) + eps)
        return q_norm, k_norm

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        """
        x: (B, L, D) if batch_first, else (L, B, D)
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
        q, k = self.qk_norm(q, k)

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
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        src: (B, L, D) if batch_first else (L, B, D)
        src_mask: same semantics as in nn.TransformerEncoderLayer
        src_key_padding_mask: (B, L)
        """
        x = src

        x = x + self.self_attn(
            self.norm1(x),
            key_padding_mask=src_key_padding_mask,
            attn_mask=src_mask,
        )
        x = x + self.mlp(self.norm2(x))

        return x