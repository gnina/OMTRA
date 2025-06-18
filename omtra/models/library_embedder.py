import torch
import torch.nn as nn
import dgl

class LibraryEmbedder(nn.Module):
    def __init__(self, num_libs: int, emb_dim: int):
        """
        num_libs: number of distinct source‐libraries (L)
        emb_dim:  size of the library embedding
        """
        super().__init__()
        self.emb_dim = emb_dim
        # learnable per‐library vectors
        self.lib_weights = nn.Embedding(num_libs, emb_dim)

    def forward(self, sys_data: torch.Tensor):
        """
        g:         a batched DGLHeteroGraph of B molecules
        sys_data:  Bool or FloatTensor of shape (B, L)
                   sys_data[i,j]==1 if mol i is in library j
        Returns the same graph with node‐features updated in‐place
        """

        # 1) compute graph‐level embedding: (B, emb_dim)
        mask = sys_data.float() if sys_data.dtype==torch.bool else sys_data
        # matmul over library‐axis
        graph_embs = mask @ self.lib_weights.weight    # (B, emb_dim)

        return graph_embs

# ------------------
# Example usage:

# assume you have:
#   G       = dgl.batch(list_of_graphs)          # a batched heterograph
#   sys_data = torch.BoolTensor(                  # shape (B, L)
#       [[1,0,0,1,0],
#        [0,1,0,0,1],
#         ...]
#   ).to(G.device)



# now G.nodes[ntype].data['h'] has been incremented by the library‐derived vector
