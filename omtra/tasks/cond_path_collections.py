

denovo_ligand = {
    'lig_x': {
        'type': 'continuous_interpolant',
    }
}

for modality in 'ace':
    denovo_ligand[f'lig_{modality}'] = dict(type='ctmc_mask')

ligand_conformer = {
    'lig_x': {
        'type': 'continuous_interpolant',
    }
}

denovo_pharmacophore = {
    'pharm_x': {
        'type': 'continuous_interpolant',
    },
    'pharm_a': dict(type='ctmc_mask'),
    'pharm_v': dict(type='continuous_interpolant')
}

protein = {
    'prot_atom': dict(type='continuous_interpolant'),
    'npnde_x': dict(type='continuous_interpolant')
}