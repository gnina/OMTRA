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
    def wrapper(graph: dgl.DGLGraph, *args, **kwargs):
        # enter a fresh local scope for all graph data
        with graph.local_scope():
            # call your function body
            return f(graph, *args, **kwargs)
    return wrapper