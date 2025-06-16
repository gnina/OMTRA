import numpy as np
from functools import wraps
import dgl


def build_lookup_table(batch_num_nodes):
    n_graphs = len(batch_num_nodes)
    lookup_table = np.zeros((n_graphs, 2), dtype=int)
    lookup_table[:, 1] = np.cumsum(batch_num_nodes, axis=0)
    lookup_table[1:, 0] = lookup_table[:-1, 1]
    return lookup_table

def g_local_scope(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        # Determine if the first argument is 'self' or the graph
        if isinstance(args[0], (dgl.DGLGraph, dgl.DGLHeteroGraph)):
            graph = args[0]
            remaining_args = args[1:]
        else:
            self, graph = args[0], args[1]
            remaining_args = args[2:]
        
        # Enter a fresh local scope for all graph data
        with graph.local_scope():
            # Call your function body
            if 'self' in locals():
                return f(self, graph, *remaining_args, **kwargs)
            else:
                return f(graph, *remaining_args, **kwargs)
    return wrapper