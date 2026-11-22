"""Chemistry-only mapping of a supplied identity into a fixed ligand atom order.

No coordinate matching, protonation changes or stereo completion are allowed.
Multiple atom mappings are accepted only if they encode the same semantic request.
"""
import numpy as np
from rdkit import Chem


def chemistry_signature(mol):
    mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(mol)
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)


def request_signature(mol):
    from omtra.data.requested_stereo import labels_from_identity
    atoms, bonds = labels_from_identity(mol)
    return tuple(int(v) for v in atoms), tuple(sorted((i, j, int(v)) for (i, j), v in bonds.items()))


def map_supplied_identity(graph_mol, smiles, max_matches=4096):
    from omtra.data.requested_stereo import labels_from_identity
    if max_matches < 1:
        raise ValueError('A positive mapping budget is required')
    source = Chem.MolFromSmiles(smiles)
    if source is None:
        return {'status': 'invalid_smiles'}
    if any(a.GetIsotope() for a in source.GetAtoms()):
        return {'status': 'unsupported_isotope'}
    source = Chem.RemoveHs(source)
    graph = Chem.Mol(graph_mol)
    graph.RemoveAllConformers()
    Chem.RemoveStereochemistry(graph)
    Chem.SanitizeMol(graph)
    if chemistry_signature(source) != chemistry_signature(graph):
        return {'status': 'chemistry_mismatch'}
    atoms, bonds = labels_from_identity(source)
    if (atoms == 6).any() or 4 in bonds.values():
        return {'status': 'unsupported_stereo'}
    matches = graph.GetSubstructMatches(source, useChirality=False, uniquify=False, maxMatches=max_matches + 1)
    if len(matches) > max_matches:
        return {'status': 'mapping_budget_exceeded', 'mappings_at_least': len(matches)}
    if not matches:
        return {'status': 'no_atom_mapping'}
    signatures = set()
    mappings = []
    for source_to_graph in matches:
        graph_to_source = np.argsort(source_to_graph).tolist()
        mapped_bonds = tuple(sorted((min(source_to_graph[i], source_to_graph[j]),
                                     max(source_to_graph[i], source_to_graph[j]), int(v))
                                    for (i, j), v in bonds.items()))
        signatures.add((tuple(int(v) for v in atoms[graph_to_source]), mapped_bonds))
        mappings.append(graph_to_source)
    if len(signatures) != 1:
        return {'status': 'ambiguous_stereo_mapping', 'mappings': len(matches), 'distinct_requests': len(signatures)}
    return {'status': 'mapped', 'smiles': smiles, 'graph_to_identity': min(mappings), 'mappings': len(matches)}


def require_reviewed_if_marked(identity):
    """Unreviewed audit candidates must not silently become benchmark input."""
    if isinstance(identity, dict) and identity.get('review_status', 'approved') != 'approved':
        raise ValueError('Supplied identity is an unreviewed audit candidate')


def molecule_from_supplied_record(identity):
    require_reviewed_if_marked(identity)
    if isinstance(identity, str):
        identity = {'smiles': identity}
    mol = Chem.MolFromSmiles(identity['smiles'])
    if mol is None:
        raise ValueError('Invalid supplied identity SMILES')
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        raise ValueError('Unsupported isotope identity')
    mol = Chem.RemoveHs(mol)
    mapping = identity.get('graph_to_identity', list(range(mol.GetNumAtoms())))
    if (any(type(j) is not int for j in mapping)
            or sorted(mapping) != list(range(mol.GetNumAtoms()))):
        raise ValueError('Invalid supplied atom mapping')
    return Chem.RenumberAtoms(mol, mapping)


def same_indexed_chemistry(a, b):
    def signature(mol):
        mol = Chem.Mol(mol); Chem.SanitizeMol(mol)
        atoms = tuple((atom.GetAtomicNum(), atom.GetFormalCharge()) for atom in mol.GetAtoms())
        bonds = tuple(sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                              b.GetBondTypeAsDouble()) for b in mol.GetBonds()))
        return atoms, bonds
    return signature(a) == signature(b)
