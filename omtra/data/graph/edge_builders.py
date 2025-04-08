import torch_cluster as tc
from omtra.data.graph.register import register_edge_builder
import torch

def radius_graph(node_positions, radius, max_num_neighbors=1000):
    edge_idxs = tc.radius_graph(node_positions, 
                             radius, 
                             batch=None, 
                             loop=False, 
                             max_num_neighbors=max_num_neighbors)
    return edge_idxs

@register_edge_builder("complete")
def complete_graph(src_pos, dst_pos, src_batch_idx, dst_batch_idx):
    edge_idxs = hetero_radius(src_pos, dst_pos, src_batch_idx, dst_batch_idx, r=float("Inf"))
    return edge_idxs

@register_edge_builder("radius")
def hetero_radius(src_pos, dst_pos, src_batch_idx, dst_batch_idx, r, max_num_neighbors=1000):
    edge_idxs = tc.radius(x=dst_pos, y=src_pos, batch_x=dst_batch_idx, batch_y=src_batch_idx, r=r, max_num_neighbors=max_num_neighbors)
    return edge_idxs

@register_edge_builder("knn")
def hetero_knn(src_pos, dst_pos, src_batch_idx, dst_batch_idx, k):
    edge_idxs = tc.knn(x=dst_pos, y=src_pos, batch_x=dst_batch_idx, batch_y=src_batch_idx, k=k)
    return edge_idxs

@register_edge_builder("symmetric")
def symmetric(builder_fn, src_pos, dst_pos, src_batch_idx, dst_batch_idx):
    edge_idxs = builder_fn(dst_pos, src_pos, dst_batch_idx, src_batch_idx)
    return edge_idxs[[1,0]]
    
