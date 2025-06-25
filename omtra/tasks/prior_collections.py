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


# de novo ligand with extra atom features
denovo_ligand_extra_feats = {
    'lig_x': {
        'type': 'gaussian',
        'params': {'ot': True}
    }
}
for modality in ['a', 'c', 'e', 'impl_H', 'aro', 'hyb', 'ring', 'chiral']:  # TODO: correct?
    denovo_ligand_extra_feats[f'lig_{modality}'] = dict(type='masked')


# typical prior setup for ligand conformer
ligand_conformer = {
    'lig_x': {
        'type': 'gaussian',
        'params': {'ot': True, 'permutation': False}
    }
}

# ligand conformer with extra atom features
ligand_conformer_extra_feats = {
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