"""Shared evaluation adapter: explicit supplied, unknown, and oracle modes."""
from rdkit import Chem
from omtra.data.requested_stereo import attach_identity, attach_geometry, molecule_from_graph


def system_identity_key(row):
    if 'system_id' in row:
        return str(row['system_id']) + '|' + str(row['ligand_id'])
    return str(row['rec_pdb']) + '|' + str(row['lig_id'])


def attach_inference_stereo(graphs, task, system_info, identities=None,
                            allow_oracle=False, unknown=False):
    if sum((identities is not None, bool(allow_oracle), bool(unknown))) != 1:
        raise ValueError('Choose exactly one supplied identity, unknown, or oracle stereo mode')
    if len(graphs) != len(system_info):
        raise ValueError('Stereo identities and graphs have different lengths')
    for i, graph in enumerate(graphs):
        if unknown:
            attach_identity(graph, task, molecule_from_graph(graph, geometry=False), 'supplied')
        elif allow_oracle:
            attach_geometry(graph, task, 'oracle_reference')
        else:
            key = system_identity_key(system_info.iloc[i])
            if key not in identities:
                raise ValueError(f'Missing supplied stereo identity: {key}')
            identity = identities[key]
            from omtra.data.supplied_identity import molecule_from_supplied_record
            mol = molecule_from_supplied_record(identity)
            attach_identity(graph, task, mol, 'supplied')
    return graphs


_GUIDANCE_NODE_KEYS = ('stereo_atom', 'stereo_source', 'stereo_centre', 'stereo_hidden', 'stereo_rel_nb', 'stereo_rel_sign')
_GUIDANCE_EDGE_KEYS = ('stereo_bond', 'stereo_rel_ez_ids', 'stereo_rel_ez_sign', 'stereo_rel')


def attach_guidance_alternative(graphs, task):
    """For classifier-free stereo guidance: store each graph's "unspecified" stereo tensors as `<key>_unk`.

    The graphs keep their requested (given/oracle) stereo; the alternative is the unknown-mode encoding built from
    the ligand chemistry alone (no coordinates, no tags), identical to `attach_inference_stereo(unknown=True)`.
    Call after the requested identity was attached.
    """
    for graph in graphs:
        nd, ed = graph.nodes['lig'].data, graph.edges['lig_to_lig'].data
        saved_n = {k: nd[k].clone() for k in _GUIDANCE_NODE_KEYS if k in nd}
        saved_e = {k: ed[k].clone() for k in _GUIDANCE_EDGE_KEYS if k in ed}
        attach_identity(graph, task, molecule_from_graph(graph, geometry=False), 'supplied')
        nd, ed = graph.nodes['lig'].data, graph.edges['lig_to_lig'].data
        alt_atom, alt_bond, alt_rel = nd['stereo_atom'].clone(), ed['stereo_bond'].clone(), ed['stereo_rel'].clone()
        for k, v in saved_n.items():
            nd[k] = v
        for k, v in saved_e.items():
            ed[k] = v
        nd['stereo_atom_unk'], ed['stereo_bond_unk'], ed['stereo_rel_unk'] = alt_atom, alt_bond, alt_rel
    return graphs
