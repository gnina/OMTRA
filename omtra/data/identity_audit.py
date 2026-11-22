"""Keep requested identity separate from compatibility with a coordinate target."""
import numpy as np
from rdkit import Chem

from omtra.data.requested_stereo import labels_from_identity, ATOM_LABELS, BOND_LABELS


def molecular_stereo_compatibility(identity, coordinates, max_matches=4096):
    """Supplied stereo constraints modulo chemical automorphisms, with missing tags unconstrained.

    This is a whole-molecule identity check. Atom-index centre agreement remains
    available separately through assess_stereo_compatibility.
    """
    result = {'compatible': False, 'specified_atom_centers': 0, 'specified_bonds': 0}
    try:
        query = Chem.RemoveHs(Chem.Mol(identity))
        atoms, bonds = labels_from_identity(query)
        result.update(specified_atom_centers=int(np.isin(atoms, [2, 3, 4, 5]).sum()),
                      specified_bonds=sum(v in (2, 3) for v in bonds.values()))
        if (atoms == 6).any() or 4 in bonds.values():
            raise ValueError('Requested identity contains unsupported stereochemistry')
        xyz = np.asarray(coordinates, dtype=float)
        if xyz.shape != (query.GetNumAtoms(), 3) or not np.isfinite(xyz).all():
            raise ValueError('Coordinates must be finite and follow the heavy-atom mapping')
        query.RemoveAllConformers()
        Chem.AssignStereochemistry(query, cleanIt=True, force=True)
        observed = Chem.Mol(query)
        Chem.RemoveStereochemistry(observed)
        conformer = Chem.Conformer(query.GetNumAtoms()); conformer.Set3D(True)
        for i, pos in enumerate(xyz):
            conformer.SetAtomPosition(i, pos.tolist())
        observed.AddConformer(conformer)
        Chem.AssignStereochemistryFrom3D(observed, replaceExistingTags=True)
        Chem.AssignStereochemistry(observed, cleanIt=True, force=True)
        actual_atoms, actual_bonds = labels_from_identity(observed)
        specified = np.flatnonzero(np.isin(atoms, [2, 3, 4, 5]))
        def agrees(mapping):
            if any(int(atoms[i]) != int(actual_atoms[mapping[i]]) for i in specified):
                return False
            return all(actual_bonds.get(tuple(sorted((mapping[i], mapping[j]))), 0) == desired
                       for (i, j), desired in bonds.items() if desired in (2, 3))
        # Use the same semantic labels as the model, rather than raw RDKit chiral
        # tags that may constrain sites our schema classifies as unspecified.
        if agrees(tuple(range(query.GetNumAtoms()))):
            result['compatible'] = True
        else:
            matches = observed.GetSubstructMatches(query, useChirality=False, uniquify=False, maxMatches=max_matches + 1)
            result['compatible'] = any(agrees(mapping) for mapping in matches)
            if not result['compatible'] and len(matches) > max_matches:
                raise ValueError('Molecular symmetry mapping budget exceeded; identity comparison unresolved')
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    return result


def assess_stereo_compatibility(identity, coordinates):
    """Check specified stereo only, without filling unspecified request labels.

    ``coordinates`` must follow the heavy-atom order of ``identity``. Callers must
    establish atom/charge/bond correspondence and source provenance separately.
    A compatible result does not certify that the source was independent of 3D.
    Neither the input identity nor the coordinate array is modified.
    """
    result = {'compatible': False, 'atom_conflicts': [], 'bond_conflicts': [],
              'unspecified_atom_centers': 0, 'unspecified_bonds': 0,
              'specified_atom_centers': 0, 'specified_bonds': 0}
    try:
        mol = Chem.RemoveHs(Chem.Mol(identity))
        desired_atoms, desired_bonds = labels_from_identity(mol)
        result.update(unspecified_atom_centers=int((desired_atoms == 1).sum()),
                      unspecified_bonds=sum(v == 1 for v in desired_bonds.values()),
                      specified_atom_centers=int(np.isin(desired_atoms, [2, 3, 4, 5]).sum()),
                      specified_bonds=sum(v in (2, 3) for v in desired_bonds.values()))
        if (desired_atoms == 6).any() or 4 in desired_bonds.values():
            raise ValueError('Requested identity contains unsupported stereochemistry')
        xyz = np.asarray(coordinates, dtype=float)
        if xyz.shape != (mol.GetNumAtoms(), 3) or not np.isfinite(xyz).all():
            raise ValueError('Coordinates must be finite and follow the heavy-atom mapping')
        observed = Chem.Mol(mol)
        observed.RemoveAllConformers()
        Chem.RemoveStereochemistry(observed)
        conformer = Chem.Conformer(mol.GetNumAtoms())
        conformer.Set3D(True)
        for i, pos in enumerate(xyz):
            conformer.SetAtomPosition(i, pos.tolist())
        observed.AddConformer(conformer)
        Chem.AssignStereochemistryFrom3D(observed, replaceExistingTags=True)
        actual_atoms, actual_bonds = labels_from_identity(observed)
        for index in np.flatnonzero(np.isin(desired_atoms, [2, 3, 4, 5]) & (desired_atoms != actual_atoms)):
            result['atom_conflicts'].append({
                'atom_index': int(index), 'element': mol.GetAtomWithIdx(int(index)).GetSymbol(),
                'requested': ATOM_LABELS[int(desired_atoms[index])],
                'observed': ATOM_LABELS[int(actual_atoms[index])]})
        for pair, desired in sorted(desired_bonds.items()):
            actual = actual_bonds.get(pair, 0)
            if desired in (2, 3) and desired != actual:
                result['bond_conflicts'].append({'atom_indices': list(pair),
                    'requested': BOND_LABELS[desired], 'observed': BOND_LABELS[actual]})
        result['compatible'] = not result['atom_conflicts'] and not result['bond_conflicts']
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    return result
