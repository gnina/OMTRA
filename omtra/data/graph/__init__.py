import torch
import dgl
from typing import Dict
import itertools
from omegaconf import DictConfig

node_types = ['lig', 'prot_atom', 'prot_res', 'pharm']

# construct the full list of possible edge types
edge_types = [f"{ntype}_to_{ntype}" for ntype in node_types]
edge_types += [f'{src_ntype}_to_{dst_ntype}' for src_ntype, dst_ntype in itertools.permutations(node_types, 2)]

def to_canonical_etype(etype: str):
    src_ntype, dst_ntype = etype.split('_to_')
    return (src_ntype, etype, dst_ntype)

def get_inv_edge_type(etype: str):
    src_ntype, dst_ntype = etype.split('_to_')
    return f"{dst_ntype}_to_{src_ntype}"


# TODO: if protein structure changes during generation, then we would need to do knn-random graph computation on the fly, which we don't know how to do yet
# TODO: in general, protein structure edges need to be computed on the fly
# TODO: we could incorporate some bas eedge structure based on  the protein identity (beacuse this doesn't change) but this is not sufficient bc
# structure components need to talk to each other based on locality
# perhaps fully-connected at the residue level and then knn or radius at the atom level with preference for inter-residue edges
# so for now, initial graph construction, edges only created for ligand and pharmacophore nodes


def build_complex_graph(
    node_data: Dict[str, Dict[str, torch.Tensor]],
    edge_idxs: Dict[str, torch.Tensor],
    edge_data: Dict[str, Dict[str, torch.Tensor]],
):
    
    # check that all node types are valid
    node_types_present = list(node_data.keys())
    for ntype in node_types_present:
        assert ntype in node_types, f"Node type {ntype} not recognized."

    # check that all edge types are valid
    etypes_in_edge_idxs = list(edge_idxs.keys())
    etypes_in_edge_data = list(edge_data.keys())
    for etype in set(etypes_in_edge_idxs + etypes_in_edge_data):
        assert etype in edge_types, f"Edge type {etype} not recognized."

    # check that every edge type included in edge_data is also included in edge_idxs
    for etype in etypes_in_edge_data:
        assert etype in etypes_in_edge_idxs, f"Edge type {etype} present in edge_data but not in edge_idxs."

    # get num_nodes for dgl graph construction
    num_nodes = {}
    for ntype in node_types:
        if ntype in node_data:
            num_nodes[ntype] = node_data[ntype]['x'].shape[0]
        else:
            num_nodes[ntype] = 0

    # get edge data for dgl graph construction
    edge_construction_dict = {}
    for etype in edge_types:
        canonical_etype = to_canonical_etype(etype)
        if etype in edge_idxs:
            edge_construction_dict[canonical_etype] = (edge_idxs[etype][0], edge_idxs[etype][1])
        else:
            edge_construction_dict[canonical_etype] = ([], [])

    # TODO: follow graph_config to construct other edges
    # examples: res-res fully connected? pharm-pharm fully connected?

    # construct dgl graph
    g = dgl.heterograph(edge_construction_dict, num_nodes_dict=num_nodes)

    # add node data to graph
    for ntype in node_types_present:
        for feature_name, feature_data in node_data[ntype].items():
            g.nodes[ntype].data[feature_name] = feature_data

    # add edge data to graph
    for etype in edge_data:
        for feature_name, feature_data in edge_data[etype].items():
            g.edges[etype].data[feature_name] = feature_data

    return g

def approx_n_edges(etype: str, graph_config: DictConfig, num_nodes_dict: Dict[str, int]):
    src_ntype, etype, dst_ntype = to_canonical_etype(etype)
    src_n, dst_n = num_nodes_dict[src_ntype], num_nodes_dict[dst_ntype]
    graph_type = graph_config.edges[etype]['type']
    if graph_type == 'complete':
         n_edges = src_n * dst_n
    elif graph_type == 'knn':
        k = graph_config.edges[etype]['k']
        n_edges = src_n * k
    elif graph_type == 'radius':
        raise NotImplementedError('have not come up with an approximate number of edges for radius graph')
    else:
        raise ValueError(f"Graph type {graph_type} not recognized.")
    
    return n_edges
