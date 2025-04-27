import numpy as np


def build_lookup_table(batch_num_nodes):
    n_graphs = len(batch_num_nodes)
    lookup_table = np.zeros((n_graphs, 2), dtype=int)
    lookup_table[:, 1] = np.cumsum(batch_num_nodes, axis=0)
    lookup_table[1:, 0] = lookup_table[:-1, 1]
    return lookup_table


# TODO: i don't think we should be using this; just the modalities
canonical_node_features = {
    "lig": ["x", "a", "c"],
    "npnde": ["x", "a", "c"],
    "pharm": ["x", "a", "v"],
    "prot_atom": ["x", "a", "e", "r"],
    "prot_res": ["x", "r"],
}
