

denovo_ligand = {
    'lig_x': {
        'type': 'continuous_interpolant',
    }
}

for modality in 'ace':
    denovo_ligand[f'lig_{modality}'] = dict(type='ctmc_mask')

denovo_ligand_condensed = {'lig_x': dict(type='continuous_interpolant'),
                           'lig_e_condensed': dict(type='ctmc_mask'),
                           'lig_cond_a': dict(type='ctmc_mask')}


denovo_ligand_extra_feats = {
    'lig_x': {
        'type': 'continuous_interpolant',
    }
}

for modality in ['a', 'c', 'e', 'impl_H', 'aro', 'hyb', 'ring', 'chiral']: 
    denovo_ligand_extra_feats[f'lig_{modality}'] = dict(type='ctmc_mask')


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
    'prot_atom_x': dict(type='continuous_interpolant'),
    'npnde_x': dict(type='continuous_interpolant')
}