import dgl
import torch
from typing import Tuple, List

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
    return node_batch_idxs

def get_edge_batch_idxs(g: dgl.DGLHeteroGraph):
    edge_batch_idxs = {}
    for etype in g.etypes:
        edge_batch_idxs[etype] = get_edge_batch_idxs_etype(g, etype)
    return edge_batch_idxs

def get_batch_idxs(g: dgl.DGLHeteroGraph) -> Tuple[dict, dict]:
    """Returns two tensors of integers indicating which molecule each node and edge belongs to."""
    node_batch_idx = get_node_batch_idxs(g)
    edge_batch_idx = get_edge_batch_idxs(g)
    return node_batch_idx, edge_batch_idx

def get_batch_info(g: dgl.DGLHeteroGraph) -> Tuple[dict,dict]:
    batch_num_nodes = {}
    for ntype in g.ntypes:
        batch_num_nodes[ntype] = g.batch_num_nodes(ntype)

    batch_num_edges = {}
    for etype in g.etypes:
        batch_num_edges[etype] = g.batch_num_edges(etype)

    return batch_num_nodes, batch_num_edges

def get_edges_per_batch(edge_node_idxs: torch.Tensor, batch_size: int, node_batch_idxs: torch.Tensor):
    device = edge_node_idxs.device
    batch_idxs = torch.arange(batch_size, device=device)
    batches_with_edges, edges_per_batch = torch.unique_consecutive(node_batch_idxs[edge_node_idxs], return_counts=True)
    edges_per_batch_full = torch.zeros_like(batch_idxs)
    edges_per_batch_full[batches_with_edges] = edges_per_batch
    return edges_per_batch_full


def copy_graph(g: dgl.DGLHeteroGraph, n_copies: int) -> List[dgl.DGLHeteroGraph]:
    """Create n_copies copies of an unbatched DGL heterogeneous graph."""
    
    # get edge indicies
    e_idxs_dict = {}
    for etype in g.canonical_etypes:
        e_idxs_dict[etype] = g.edges(form='uv', etype=etype)

    # get number of nodes
    num_nodes_dict = {}
    for ntype in g.ntypes:
        num_nodes_dict[ntype] = g.num_nodes(ntype=ntype)

    # make copies of graph
    g_copies = [ dgl.heterograph(e_idxs_dict, num_nodes_dict=num_nodes_dict, device=g.device) for _ in range(n_copies) ]

    # transfer over node features
    for ntype in g.ntypes:
        for feat_name in g.nodes[ntype].data.keys():

            src_feat = g.nodes[ntype].data[feat_name].detach() # get the feature on the source graph

            # add a clone to each copy
            for copy_idx in range(n_copies):
                g_copies[copy_idx].nodes[ntype].data[feat_name] = src_feat.clone()

    # transfer over edge features
    for etype in g.canonical_etypes:
        for feat_name in g.edges[etype].data.keys():
            src_feat = g.edges[etype].data[feat_name].detach()
            for copy_idx in range(n_copies):
                g_copies[copy_idx].edges[etype].data[feat_name] = src_feat.clone()

    return g_copies


def build_lig_edge_idxs(n_atoms: int) -> torch.Tensor:
    """Generate edge indicies for lig_to_lig; a fully-connected graph but with upper and lower triangle edges separated."""
    upper_edge_idxs = torch.triu_indices(n_atoms, n_atoms, offset=1)

    # get lower triangle edges by swapping source and destination of upper_edge_idxs
    lower_edge_idxs = torch.stack((upper_edge_idxs[1], upper_edge_idxs[0]))

    edges = torch.cat((upper_edge_idxs, lower_edge_idxs), dim=1)
    return edges