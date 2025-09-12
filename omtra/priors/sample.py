import torch
from omtra.priors.prior_factory import get_prior
from omtra.data.graph.utils import get_upper_edge_mask
import functools
import dgl
from omtra.tasks.modalities import name_to_modality
from omtra.tasks.tasks import Task
from omtra.data.graph.utils import get_node_batch_idxs_ntype

def sample_priors(
        g: dgl.DGLHeteroGraph, 
        task_class: Task, 
        prior_fns: dict, 
        training: bool, 
        com: torch.Tensor = None,
        fake_atoms: bool = False
        ):
    for modality_name in prior_fns:
        prior_name, prior_func = prior_fns[modality_name] # get prior name and function
        modality = name_to_modality(modality_name) # get the modality object

        modality_has_fake_atoms = modality_name in ['lig_a', 'lig_cond_a'] and fake_atoms

        # skip modalities that are not present in the graph (for example a system with no npndes)
        if modality.is_node and g.num_nodes(modality.entity_name) == 0:
            continue
        elif not modality.is_node and g.num_edges(modality.entity_name) == 0:
            continue

        # fetch the target data from the graph object
        g_data_loc = g.nodes if modality.is_node else g.edges

        if modality.is_node:
            n = g.num_nodes(modality.entity_name)
        else:
            n = g.num_edges(modality.entity_name)

        if modality.is_categorical:
            d = modality.n_categories
        elif modality.data_key == 'x':
            d = 3
        elif modality.data_key == 'v':
            d = (4, 3) # hard-coded assumpotion of 4 pharmacophore vector features 
        else:
            raise ValueError(f'unaccounted for modality {modality.name}')
        
        # determine args for the prior function
        if 'apo' in prior_name:
            args = [g_data_loc[modality.entity_name].data[f'{modality.data_key}_0'], ]
        elif training or 'target_dependent' in prior_name:
            target_data = g_data_loc[modality.entity_name].data[f'{modality.data_key}_1_true']
            args = [target_data, ]
            if modality.is_categorical:
                args.append(modality.n_categories+int(modality_has_fake_atoms))
        else:
            # inference time priors, which are different than train time priors, for reasons i forget :(
            n = g.num_nodes(modality.entity_name) if modality.is_node else g.num_edges(modality.entity_name)
            if modality.is_categorical:
                # for categorical data, we need to pass the number of categories
                args = (n, modality.n_categories+int(modality_has_fake_atoms))
            elif modality.data_key == 'x':
                # for x data, we need to pass the number of dimensions
                args = [n, 3]
            elif modality.data_key == 'v':
                # for v data, we need to pass the number of dimensions
                args = [n, [4, 3]]

        # draw a sample from the prior
        prior_sample = prior_func(*args).to(g.device)

        # for edge features, make sure upper and lower triangle are the same
        # TODO: this logic may change if we decide to do something other fully-connected lig-lig edges
        # this actually does nothing in the case of a masked CTMC prior but :shrug:, would be necessary if for example it was a uniform CTMC prior
        if modality.entity_name == 'lig_to_lig':
            upper_edge_mask = get_upper_edge_mask(g, modality.entity_name)
            prior_sample[~upper_edge_mask] = prior_sample[upper_edge_mask]

        # add the prior sample to the graph
        g_data_loc[modality.entity_name].data[f'{modality.data_key}_0'] = prior_sample


    # if pharmacophore and ligand structure are being generated
    # move pharmacophore prior COM to the ligand prior COM
    groups_generated = task_class.groups_generated
    pharm_gen = 'pharmacophore' in groups_generated
    lig_gen = 'ligand_structure' in groups_generated
    if training and pharm_gen and lig_gen:
        if g.num_nodes('pharm') > 0:
            # get the pharmacophore prior COM
            ph_com = dgl.readout_nodes(g, feat='x_0', op='mean', ntype='pharm')

            # get the ligand prior COM
            com = dgl.readout_nodes(g, feat='x_0', op='mean', ntype='lig')

            # move the pharmacophore prior to the ligand prior COM
            node_batch_idxs = get_node_batch_idxs_ntype(g, 'pharm')
            g.nodes['pharm'].data['x_0'] += (com - ph_com)[node_batch_idxs]
    
    if not training and lig_gen:
        assert com.shape == (g.batch_size, 3)
        node_batch_idxs = get_node_batch_idxs_ntype(g, 'lig')

        # move the ligand prior positions COM to the designated COM
        current_lig_com = dgl.readout_nodes(g, feat='x_0', op='mean', ntype='lig')
        g.nodes['lig'].data['x_0'] += (com - current_lig_com)[node_batch_idxs]

    if not training and pharm_gen:
        if g.num_nodes('pharm') > 0:
            node_batch_idxs = get_node_batch_idxs_ntype(g, 'pharm')
            current_pharm_com = dgl.readout_nodes(g, feat='x_0', op='mean', ntype='pharm')
            g.nodes['pharm'].data['x_0'] += (com - current_pharm_com)[node_batch_idxs]

    return g