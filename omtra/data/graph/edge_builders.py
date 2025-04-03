import torch_cluster as tc

def radius_graph(node_positions, radius, max_num_neighbors=1000):
    edge_idxs = tc.radius_graph(node_positions, 
                             radius, 
                             batch=None, 
                             loop=False, 
                             max_num_neighbors=max_num_neighbors)
    return edge_idxs

def complete_graph(node_positions):
    n_nodes = node_positions.shape[0]
    edge_idxs = tc.radius_graph(node_positions, 
                             float('inf'), 
                             batch=None, 
                             loop=False, 
                             max_num_neighbors=n_nodes+2)
    return edge_idxs


def hetero_radius(src_pos, dst_pos, src_batch_idx, dst_batch_idx, r, max_num_neighbors=1000):
    edge_idxs = tc.radius(x=dst_pos, y=src_pos, batch_x=dst_batch_idx, batch_y=src_batch_idx, r=r, max_num_neighbors=max_num_neighbors)
    return edge_idxs

def hetero_knn(src_pos, dst_pos, src_batch_idx, dst_batch_idx, k):
    edge_idxs = tc.knn(x=dst_pos, y=src_pos, batch_x=dst_batch_idx, batch_y=src_batch_idx, k=k)
    return edge_idxs
