

denovo_ligand = {
    'lig_x': {
        'type': 'continuous_delta',
    }
}

for modality in 'ace':
    denovo_ligand[f'lig_{modality}'] = dict(type='two_state_mask')

ligand_conformer = {
    'lig_x': {
        'type': 'continuous_delta',
    }
}

denovo_pharmacophore = {
    'pharm_x': {
        'type': 'continuous_delta',
    },
    'pharm_a': dict(type='two_state_mask'),
    'pharm_v': dict(type='continuous_delta')
}

protein = {
    'prot_atom': dict(type='continuous_delta'),
    'npnde_x': dict(type='continuous_delta')
}