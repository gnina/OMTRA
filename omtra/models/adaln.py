import torch
import torch.nn as nn

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
class AdaLNWeightGenerator(nn.Module):
    def __init__(self, scalar_size, vector_size, s_ratio: int = 1, v_ratio: int = 1):
        super().__init__()
        self.scalar_size = scalar_size
        self.vector_size = vector_size
        self.s_ratio = s_ratio
        self.v_ratio = v_ratio

        # enforce scalar size and vector size divid evenly by ratios
        assert scalar_size % s_ratio == 0
        assert vector_size % v_ratio == 0
        assert isinstance(s_ratio, int) and s_ratio >= 1
        assert isinstance(v_ratio, int) and v_ratio >= 1

        self.n_s_params_gen = self.scalar_size // self.s_ratio
        self.n_v_params_gen = self.vector_size // self.v_ratio

        self.mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(scalar_size, 7*self.n_s_params_gen + 5*self.n_v_params_gen, bias=True)
            )

    def forward(self, x):
        # x has shape (batch_size, scalar_size)

        params = self.mlp(x)
        scalar_params = params[:, :7*self.n_s_params_gen]
        vector_params = params[:, 7*self.n_s_params_gen:]

        scalar_params = scalar_params.repeat_interleave(self.s_ratio, dim=-1).chunk(7, dim=-1)
        vector_params = vector_params.repeat_interleave(self.v_ratio, dim=-1).chunk(5, dim=-1)

        params_dict = {
            's_msa_shift': scalar_params[0],
            's_msa_scale': scalar_params[1],
            's_msg_norm': scalar_params[2],
            's_msa_gate': scalar_params[3],
            's_ff_shift': scalar_params[4],
            's_ff_scale': scalar_params[5],
            's_ff_gate': scalar_params[6],
            'v_msa_scale': vector_params[0],
            'v_msg_norm': vector_params[1],
            'v_msa_gate': vector_params[2],
            'v_ff_scale': vector_params[3],
            'v_ff_gate': vector_params[4],
        }
        
        return params_dict
    

class FinalLayer(nn.Module):
    def __init__(self, n_scalars: int, n_vec_channels: int, n_edge_feats: int):
        super().__init__()

        pass