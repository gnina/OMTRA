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