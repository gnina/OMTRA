import torch
import dgl
from torch_scatter import segment_csr
from torch.distributions import Binomial
from omtra.tasks.modalities import Modality
from functools import partial


def purity_sampling(
    g: dgl.DGLHeteroGraph,
    modality: Modality,
    xt,
    x1_probs,
    unmask_prob,
    mask_index,
    batch_size,
    batch_num_nodes,
    device,
    upper_edge_mask,
):
    masked_nodes = xt == mask_index  # mask of which nodes are currently unmasked
    purities = x1_probs.max(-1)[
        0
    ]  # the highest probability of any category for each node

    if not modality.is_node:
        # recreate a version of purities that includes lower edges
        purities_ul = torch.zeros(g.num_edges(), device=device, dtype=purities.dtype)
        purities_ul[upper_edge_mask] = purities
        purities = purities_ul

        # duplicate masked_nodes for lower edges, mark all lower edges as unmasked
        masked_nodes_ul = torch.zeros(g.num_edges(), device=device, dtype=torch.bool)
        masked_nodes_ul[upper_edge_mask] = masked_nodes
        masked_nodes = masked_nodes_ul

    # compute the number of masked nodes per graph in the batch
    indptr = torch.zeros(batch_size + 1, device=device, dtype=torch.long)
    indptr[1:] = batch_num_nodes.cumsum(0)
    masked_nodes_per_graph = segment_csr(
        masked_nodes.long(), indptr
    )  # has shape (batch_size,)

    # set purities of unmasked nodes to -1
    purities[~masked_nodes] = -1

    # sample the number of nodes to unmask per graph
    n_unmask_per_graph = Binomial(
        total_count=masked_nodes_per_graph, probs=unmask_prob
    ).sample()

    with g.local_scope():
        if not modality.is_node:
            data_src = g.edges[modality.entity_name].data
            topk_func = partial(dgl.topk_edges, etype=modality.entity_name)
        else:
            data_src = g.nodes[modality.entity_name].data
            topk_func = partial(dgl.topk_nodes, ntype=modality.entity_name)

        data_src["purity"] = purities.unsqueeze(-1)
        k = int(n_unmask_per_graph.max())

        if k != 0:
            _, topk_idxs_batched = topk_func(g, feat="purity", k=k, sortby=0)

            # topk_idxs contains indicies relative to each batch
            # but we need to convert them to batched-graph indicies
            topk_idxs_batched = topk_idxs_batched + indptr[:-1].unsqueeze(-1)

    # slice out the top k nodes for each graph
    if k != 0:
        col_indices = torch.arange(k, device=device).unsqueeze(0)
        mask = col_indices < n_unmask_per_graph.unsqueeze(1)

        # Apply mask to get only desired indices
        nodes_to_unmask = topk_idxs_batched[mask]
    else:
        nodes_to_unmask = torch.tensor([], device=device, dtype=torch.long)

    if not modality.is_node:
        will_unmask = torch.zeros(g.num_edges(), dtype=torch.bool, device=device)
        will_unmask[nodes_to_unmask] = True
        will_unmask = will_unmask[upper_edge_mask]
    else:
        will_unmask = torch.zeros_like(xt, dtype=torch.bool)
        will_unmask[nodes_to_unmask] = True

    return will_unmask
