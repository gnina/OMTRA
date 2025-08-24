# store common formats for priors


# typical prior setup for de novo ligand
denovo_ligand = {
    'lig_x': {
        'type': 'gaussian',
        'params': {'ot': True}
    }
}
for modality in 'ace':
    denovo_ligand[f'lig_{modality}'] = dict(type='masked')

denovo_ligand_condensed = {'lig_x': {'type': 'gaussian', 'params': {'ot': True}},
                           'lig_e_condensed': dict(type='masked'),
                           'lig_cond_a': dict(type='masked')}

# de novo ligand with extra atom features
denovo_ligand_extra_feats = {
    'lig_x': {
        'type': 'gaussian',
        'params': {'ot': True}
    }
}
for modality in ['a', 'c', 'e', 'impl_H', 'aro', 'hyb', 'ring', 'chiral']:
    denovo_ligand_extra_feats[f'lig_{modality}'] = dict(type='masked')


# typical prior setup for ligand conformer
ligand_conformer = {
    'lig_x': {
        'type': 'gaussian',
        'params': {'ot': True, 'permutation': False}
    }
}

# typical prior setup for de novo pharmacophore
denovo_pharmacophore = {
    'pharm_x': {
        'type': 'gaussian',
        'params': {'ot': True}
    },
    'pharm_a': dict(type='masked'),
    'pharm_v': dict(type='gaussian')
}