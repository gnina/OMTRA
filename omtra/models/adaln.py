from dataclasses import dataclass
from typing import List, Literal, Optional, Dict, Type
import torch
from torch import nn

Domain = Literal["s", "v"]

@dataclass(frozen=True)
class ChunkSpec:
    key: str
    domain: Domain
    ratio_override: Optional[int] = None

@dataclass(frozen=True)
class AdaLNSpec:
    chunks: List[ChunkSpec]
    @property
    def n_s(self) -> int: return sum(c.domain == "s" for c in self.chunks)
    @property
    def n_v(self) -> int: return sum(c.domain == "v" for c in self.chunks)
    @property
    def keys(self) -> List[str]: return [c.key for c in self.chunks]

class AdaLNParams(dict):
    """Dict[str, torch.Tensor] with tensors shaped to scalar_size or vector_size."""
    pass

# ------- Specs -------
GRAPH_CONV_SPEC = AdaLNSpec([
    ChunkSpec("s_msa_shift","s"), ChunkSpec("s_msa_scale","s"),
    ChunkSpec("s_msg_norm","s"),  ChunkSpec("s_msa_gate","s"),
    ChunkSpec("s_ff_shift","s"),  ChunkSpec("s_ff_scale","s"),
    ChunkSpec("s_ff_gate","s"),
    ChunkSpec("v_msa_scale","v"), ChunkSpec("v_msg_norm","v"),
    ChunkSpec("v_msa_gate","v"),  ChunkSpec("v_ff_scale","v"),
    ChunkSpec("v_ff_gate","v"),
])

POSITION_UPDATE_SPEC = AdaLNSpec([
    ChunkSpec("s_shift","s"), ChunkSpec("s_scale","s"),
    ChunkSpec("v_scale","v"),
])

# ------- Base generator (spec-driven) -------
class AdaLNWeightGenerator(nn.Module):
    spec: AdaLNSpec  # subclasses set this

    def __init__(
        self,
        scalar_size: int,
        vector_size: int,
        *,
        s_ratio: int = 1,
        v_ratio: int = 1,
        mlp_hidden: Optional[int] = None,
        s_out_dim = None,
        v_out_dim = None,
    ):
        super().__init__()
        if not hasattr(self, "spec"):
            raise TypeError("Subclass must set class attribute 'spec' to an AdaLNSpec.")

        assert isinstance(s_ratio, int) and s_ratio >= 1
        assert isinstance(v_ratio, int) and v_ratio >= 1
        assert scalar_size % s_ratio == 0
        assert vector_size % v_ratio == 0

        self.scalar_size = scalar_size
        self.vector_size = vector_size
        self.s_ratio = s_ratio
        self.v_ratio = v_ratio

        if s_out_dim is not None:
            assert s_out_dim % s_ratio == 0
            n_s_gen = s_out_dim // s_ratio
            self.s_out_dim = s_out_dim
        else:
            n_s_gen = (scalar_size // s_ratio) if scalar_size > 0 else 0
            self.s_out_dim = scalar_size
        if v_out_dim is not None:
            assert v_out_dim % v_ratio == 0
            n_v_gen = v_out_dim // v_ratio
            self.v_out_dim = v_out_dim
        else:
            n_v_gen = (vector_size // v_ratio) if vector_size > 0 else 0
            self.v_out_dim = vector_size
        self._n_s_gen = n_s_gen
        self._n_v_gen = n_v_gen

        out_dim = self.spec.n_s * n_s_gen + self.spec.n_v * n_v_gen

        if mlp_hidden is None:
            self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(scalar_size, out_dim, bias=True))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(scalar_size, mlp_hidden), nn.SiLU(),
                nn.Linear(mlp_hidden, out_dim, bias=True),
            )

    @property
    def keys(self) -> List[str]:
        return self.spec.keys

    def forward(self, x: torch.Tensor) -> AdaLNParams:
        B = x.shape[0]
        flat = self.mlp(x)  # [B, out_dim]

        # ordered split: all scalar-domain chunks (spec order), then vector-domain chunks
        s_chunks = [c for c in self.spec.chunks if c.domain == "s"]
        v_chunks = [c for c in self.spec.chunks if c.domain == "v"]

        s_total = len(s_chunks) * self._n_s_gen
        s_flat, v_flat = flat[:, :s_total], flat[:, s_total:]

        s_parts = list(torch.split(s_flat, self._n_s_gen, dim=-1)) if s_total > 0 else []
        v_parts = list(torch.split(v_flat, self._n_v_gen, dim=-1)) if v_flat.shape[-1] > 0 else []

        out = AdaLNParams()

        for part, ch in zip(s_parts, s_chunks):
            r = ch.ratio_override or self.s_ratio
            exp = part.repeat_interleave(r, dim=-1)  # -> [B, scalar_size]
            if exp.shape[-1] != self.s_out_dim:
                raise ValueError(f"{ch.key}: expanded {exp.shape[-1]} != scalar_size {self.scalar_size}")
            out[ch.key] = exp

        for part, ch in zip(v_parts, v_chunks):
            r = ch.ratio_override or self.v_ratio
            exp = part.repeat_interleave(r, dim=-1)  # -> [B, vector_size]
            if exp.shape[-1] != self.v_out_dim:
                raise ValueError(f"{ch.key}: expanded {exp.shape[-1]} != vector_size {self.vector_size}")
            out[ch.key] = exp

        return out

# ------- Preset subclasses (no spec knowledge required by callers) -------
class GraphConvAdaLN(AdaLNWeightGenerator):
    spec = GRAPH_CONV_SPEC

class PositionUpdateAdaLN(AdaLNWeightGenerator):
    spec = POSITION_UPDATE_SPEC


class ScalarAdaLN(AdaLNWeightGenerator):
    spec = AdaLNSpec([
        ChunkSpec('s_shift',"s"),
        ChunkSpec('s_scale',"s"),
    ])

class HomoEdgeUpdateAdaLN(ScalarAdaLN):
    pass

class HeteroEdgeUpdateAdaLN(AdaLNWeightGenerator):
    spec = AdaLNSpec([
        ChunkSpec('s_shift_src',"s"),
        ChunkSpec('s_scale_src',"s"),
        ChunkSpec('s_shift_dst',"s"),
        ChunkSpec('s_scale_dst',"s"),
    ])

# ------- Optional: registry + factory for string presets -------
_ADALN_REGISTRY: Dict[str, Type[AdaLNWeightGenerator]] = {
    "graph_conv": GraphConvAdaLN,
    "position_update":  PositionUpdateAdaLN,
}

def make_adaln(preset: str, **kwargs) -> AdaLNWeightGenerator:
    cls = _ADALN_REGISTRY[preset]
    return cls(**kwargs)

#################

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class FinalLayer(nn.Module):
    def __init__(self, n_scalars: int, n_vec_channels: int, n_edge_feats: int):
        super().__init__()

        pass

