"""Versioned physical-check contract; absent/unknown required checks fail closed."""
import pandas as pd

SCHEMA_VERSION = 1
MOLECULE_CHECKS = (
    'mol_pred_loaded', 'sanitization', 'inchi_convertible', 'all_atoms_connected', 'no_radicals',
    'bond_lengths', 'bond_angles', 'internal_steric_clash', 'aromatic_ring_flatness',
    'non-aromatic_ring_non-flatness', 'double_bond_flatness', 'internal_energy',
)
IDENTITY_CHECKS = ('mol_true_loaded','molecular_formula','molecular_bonds',
                   'double_bond_stereochemistry','tetrahedral_chirality')
ENVIRONMENT_CHECKS = ('mol_cond_loaded','protein-ligand_maximum_distance',
    'minimum_distance_to_protein','minimum_distance_to_organic_cofactors',
    'minimum_distance_to_inorganic_cofactors','minimum_distance_to_waters',
    'volume_overlap_with_protein','volume_overlap_with_organic_cofactors',
    'volume_overlap_with_inorganic_cofactors','volume_overlap_with_waters')


def required_checks(identity=False, protein=True, prefix='pb_'):
    return [prefix+c for c in MOLECULE_CHECKS + (IDENTITY_CHECKS if identity else ()) +
            (ENVIRONMENT_CHECKS if protein else ())]


def evaluate_checks(df, required):
    values = df.reindex(columns=required)
    known = values.notna().all(axis=1)
    true = values.isin([True, 1, 'True', 'true', '1', '1.0']).all(axis=1)
    return known & true, ~known
