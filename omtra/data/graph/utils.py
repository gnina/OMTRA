import dgl
import torch
from typing import Tuple

def get_upper_edge_mask(g: dgl.DGLHeteroGraph, etype: str):
        """Returns a boolean mask for the edges that lie in the upper triangle of the adjacency matrix for each molecule in the batch."""
        # this algorithm assumes that the edges are ordered such that the upper triangle edges come first, followed by the lower triangle edges for each graph in the batch
        # and then those graph-wise edges are concatenated together
        # you can see that this is indeed how the edges are constructed by inspecting data_processing.dataset.MoleculeDataset.__getitem__
        edges_per_mol = g.batch_num_edges(etype=etype)
        ul_pattern = torch.tensor([1,0]).repeat(g.batch_size).to(g.device)
        n_edges_pattern = (edges_per_mol/2).int().repeat_interleave(2)
        upper_edge_mask = ul_pattern.repeat_interleave(n_edges_pattern).bool()
        return upper_edge_mask

def get_node_batch_idxs_ntype(g: dgl.DGLHeteroGraph, ntype: str):
    """Returns a tensor of integers indicating which graph each node belongs to for a given node type."""
    node_batch_idx = torch.arange(g.batch_size, device=g.device)
    node_batch_idx = node_batch_idx.repeat_interleave(g.batch_num_nodes(ntype=ntype))
    return node_batch_idx

def get_edge_batch_idxs_etype(g: dgl.DGLHeteroGraph, etype: str):
    """Returns a tensor of integers indicating which batch each edge belongs to."""
    edge_batch_idx = torch.arange(g.batch_size, device=g.device)
    edge_batch_idx = edge_batch_idx.repeat_interleave(g.batch_num_edges(etype=etype))
    return edge_batch_idx

def get_node_batch_idxs(g: dgl.DGLHeteroGraph):
     node_batch_idxs = {}
     for ntype in g.ntypes:
            node_batch_idxs[ntype] = get_node_batch_idxs_ntype(g, ntype)

def get_edge_batch_idxs(g: dgl.DGLHeteroGraph):
    edge_batch_idxs = {}
    for etype in g.etypes:
        edge_batch_idxs[etype] = get_edge_batch_idxs_etype(g, etype)

def get_batch_idxs(g: dgl.DGLHeteroGraph) -> Tuple[dict, dict]:
    """Returns two tensors of integers indicating which molecule each node and edge belongs to."""
    node_batch_idx = get_node_batch_idxs(g)
    edge_batch_idx = get_edge_batch_idxs(g)
    return node_batch_idx, edge_batch_idx