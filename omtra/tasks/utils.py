import dgl
from typing import Dict, List, Tuple
from omtra.tasks.tasks import Task
from omtra.tasks.modalities import Modality
from omegaconf import DictConfig
import torch

from omtra.data.graph.edge_factory import get_edge_builders
from omtra.data.graph.utils import get_batch_info, get_edges_per_batch
from omtra.data.graph import get_edges_for_task
from omtra.data.graph import to_canonical_etype

# from line_profiler import LineProfiler, profile

# @profile
def build_edges(g: dgl.DGLHeteroGraph, 
                task: Task, 
                node_batch_idx: Dict[str, torch.Tensor], 
                graph_config,
                etype_subset: List[str] = None,
                lig_only: bool = False,
                ) -> dgl.DGLHeteroGraph:
    batch_num_nodes, batch_num_edges = get_batch_info(g)
    batch_size = g.batch_size
    
    edge_builders = get_edge_builders(graph_config)
    predetermined_edges = ["lig_to_lig", "npnde_to_npnde"]
    etypes = get_edges_for_task(task, graph_config=graph_config)
    if etype_subset is not None:
        etypes = [ x for x in etype_subset if x in etypes ]
    
    for etype in etypes:
        if etype in predetermined_edges or "covalent" in etype:
            continue
        
        if lig_only and "lig" not in etype:
            continue
        
        src_ntype, _, dst_ntype = to_canonical_etype(etype)

        # if we don't have any of the nodes for this edge type, skip it
        # for example, the task supports npndes but the training example just doesn't have any
        if g.num_nodes(src_ntype) == 0 or g.num_nodes(dst_ntype) == 0:
            continue
        
        src_pos, dst_pos = g.nodes[src_ntype].data["x_t"], g.nodes[dst_ntype].data["x_t"]
        builder_fn = edge_builders.get(etype)
        if builder_fn is None:
            raise NotImplementedError(f"Error getting edge builder for {etype}")
        
        edge_idxs = builder_fn(src_pos, dst_pos, node_batch_idx[src_ntype], node_batch_idx[dst_ntype])

        if etype == 'prot_atom_to_prot_atom' and graph_config.get('cov_prot_edges', False):
            edge_idxs = get_combined_edges(g, edge_idxs)

        g.add_edges(edge_idxs[0], edge_idxs[1], etype=etype)
        canonical_etype = (src_ntype, etype, dst_ntype)
        batch_num_edges[canonical_etype] = get_edges_per_batch(edge_idxs[0], batch_size, node_batch_idx[src_ntype])
        
    g.set_batch_num_edges(batch_num_edges)
    g.set_batch_num_nodes(batch_num_nodes)
    
    return g

def remove_edges(g: dgl.DGLHeteroGraph, lig_only: bool = False) -> dgl.DGLHeteroGraph:
    batch_num_nodes, batch_num_edges = get_batch_info(g)
    batch_size = g.batch_size
    
    predetermined_edges = ["lig_to_lig", "npnde_to_npnde"]
    etypes = g.etypes
    
    for etype in etypes:
        if etype in predetermined_edges or "covalent" in etype:
            continue
        
        if lig_only and "lig" not in etype:
            continue
        
        if g.num_edges(etype) == 0:
            continue
        src_ntype, _, dst_ntype = to_canonical_etype(etype)
        canonical_etype = (src_ntype, etype, dst_ntype)
        
        # edges_to_remove = g.edges(etype=etype)
        device = g.device
        g.remove_edges(torch.arange(g.num_edges(etype), device=device), etype=etype)
        batch_num_edges[canonical_etype] = torch.zeros(batch_size, dtype=torch.int64, device=device)
    
    g.set_batch_num_edges(batch_num_edges)
    g.set_batch_num_nodes(batch_num_nodes)
    
    return g

def get_combined_edges(g, new_edge_idxs: torch.Tensor):

    # new_edge_idxs has shape (2, n_edges, )
    cov_edge_idxs = torch.stack(g.edges(etype='prot_atom_covalent_prot_atom'), dim=0) # has shape (2, n_edges)

    # concatenate along the edge dimension → [2, n_a + n_b]
    edges = torch.cat([cov_edge_idxs, new_edge_idxs], dim=1)
    # transpose to shape [n_edges, 2] so we can unique rows
    edges_t = edges.t()
    # unique rows → returns sorted unique edge pairs
    unique_edges_t = torch.unique(edges_t, dim=0)
    # transpose back to [2, n_union]
    return unique_edges_t.t()
    
    
    