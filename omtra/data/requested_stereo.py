"""Semantic requested stereochemistry, independent of atom order and reference poses.

Geometry reconstruction is a supervised-data adapter. Inference callers must supply
identity or explicitly opt into the separately marked oracle benchmark adapter.
"""
from functools import lru_cache
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

ATOM_LABELS = ('none', 'unspecified', 'R', 'S', 'r', 's', 'unsupported')
BOND_LABELS = ('none', 'unspecified', 'E', 'Z', 'unsupported')
SOURCES = {'supplied': 1, 'training_geometry': 2, 'oracle_reference': 3}
FEATURIZER_VERSION = 1
CIP_MAX_ITERATIONS = 12_500_000   # measured on 57 ligands that hit a 100,000 cap in real runs: 12.5M labels 55 of 57 with every call <= 3.1 s; the other 2 (unlimited: > 60 s each, the ~10 minute stalls) fall back to unspecified labels


def eligible(task):
    return 'ligand_identity_condensed' in task.groups_fixed or 'ligand_identity' in task.groups_fixed


def labels_from_identity(mol):
    """Read supplied tags; never infer missing tags from a conformer."""
    mol = Chem.Mol(mol)
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    # Per-molecule cap on the CIP labeler's work. Without it, one pathological ligand (a highly symmetric or very large one) can run for ~10 minutes inside this call, and because every
    # rank waits for its slowest data worker, ALL GPUs idle meanwhile (seen twice in the first 600 updates of the 4-GPU runs, py-spy: labels_from_identity -> AssignCIPLabels).
    # RDKit: 'most structures need < 10,000 iterations; 1,250,000 take about 1 second'. When the cap is hit RDKit raises RuntimeError (MaxIterationsExceeded); the dataset code
    # (multitask.py) already treats RuntimeError as 'unlabelable training ligand': all stereo labels unspecified, logged as STEREO_FALLBACK. 0 = unlimited (old behaviour).
    rdCIPLabeler.AssignCIPLabels(mol, maxRecursiveIterations=CIP_MAX_ITERATIONS)
    atoms = np.zeros(mol.GetNumAtoms(), dtype=np.int64)
    bonds = {}
    for stereo in Chem.FindPotentialStereo(mol):
        if stereo.type == Chem.StereoType.Atom_Tetrahedral:
            atoms[stereo.centeredOn] = 1
        elif stereo.type == Chem.StereoType.Bond_Double:
            bond = mol.GetBondWithIdx(stereo.centeredOn)
            bonds[tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))] = 1
        elif str(stereo.type).startswith('Atom_'):
            atoms[stereo.centeredOn] = 6
        elif str(stereo.type).startswith('Bond_'):
            bond=mol.GetBondWithIdx(stereo.centeredOn)
            bonds[tuple(sorted((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx())))] = 4
    for atom in mol.GetAtoms():
        if atom.GetIsotope():
            raise ValueError('Isotope identity is unsupported by the current OMTRA chemistry schema')
        if atom.GetChiralTag() not in (Chem.ChiralType.CHI_UNSPECIFIED,Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            atoms[atom.GetIdx()]=6
            continue
        if atom.HasProp('_CIPCode'):
            value = atom.GetProp('_CIPCode')
            atoms[atom.GetIdx()] = ATOM_LABELS.index(value) if value in ATOM_LABELS else 6
    for bond in mol.GetBonds():
        key = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        if bonds.get(key)==4:continue
        if bond.HasProp('_CIPCode'):
            value = bond.GetProp('_CIPCode')
            bonds[key] = BOND_LABELS.index(value) if value in BOND_LABELS else 4
        elif bond.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ):
            bonds[key] = 2 if bond.GetStereo() == Chem.BondStereo.STEREOE else 3
    return atoms, bonds


def clear_stereo(g):
    g.nodes['lig'].data['h_eligible'] = torch.zeros(g.num_nodes('lig'), dtype=torch.float32, device=g.device)
    g.nodes['lig'].data['is_hydrogen'] = torch.zeros(g.num_nodes('lig'), dtype=torch.float32, device=g.device)
    g.nodes['lig'].data['stereo_atom'] = torch.zeros(g.num_nodes('lig'), dtype=torch.long, device=g.device)
    g.nodes['lig'].data['stereo_source'] = torch.zeros(g.num_nodes('lig'), dtype=torch.long, device=g.device)
    # True tetrahedral-centre mask (R/S/r/s before any hiding) and the hidden-label mask; used by the
    # chirality loss and for logging. Both stay zero where no stereo is requested.
    g.nodes['lig'].data['stereo_centre'] = torch.zeros(g.num_nodes('lig'), dtype=torch.long, device=g.device)
    # Handedness sign (+1/-1) of the requested configuration at each tetrahedral unit's centre; used only by the optional correction features.
    g.nodes['lig'].data['stereo_unit_z'] = torch.zeros(g.num_nodes('lig'), dtype=torch.float32, device=g.device)
    g.nodes['lig'].data['stereo_hidden'] = torch.zeros(g.num_nodes('lig'), dtype=torch.long, device=g.device)
    g.edges['lig_to_lig'].data['stereo_bond'] = torch.zeros(g.num_edges('lig_to_lig'), dtype=torch.long, device=g.device)
    from omtra.data.stereo_relational import _empty
    _empty(g)


def attach_identity(g, task, mol, source='supplied'):
    clear_stereo(g)
    if not eligible(task):
        return g
    if source not in SOURCES:
        raise ValueError(f'Unknown stereo identity source: {source}')
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        raise ValueError('Isotope identity is unsupported by the current OMTRA chemistry schema')
    with_hs = Chem.Mol(mol)
    mol = Chem.RemoveHs(with_hs)
    if mol.GetNumAtoms() != g.num_nodes('lig') and with_hs.GetNumAtoms() == g.num_nodes('lig'):
        # The ligand graph itself keeps some explicit hydrogens (e.g. [H-], H2, hydrogens on exotic
        # ligands); the mapping is then the graph's own atom order, so do not strip them.
        mol = with_hs
    if mol.GetNumAtoms() != g.num_nodes('lig'):
        raise ValueError('Requested identity must have an explicit atom mapping to the ligand graph')
    atoms, bonds = labels_from_identity(mol)
    # Phosphorus is never a stereocentre here (label 0 = 'not a centre'): in a phosphate two of the four neighbours are the same group written two ways, so the tag depends on a
    # drawing convention (99.7% of PLINDER phosphorus centres are such pseudo-centres). With stereo_centre = 0
    # a phosphorus atom gets no label input, no correction arrow, no relational edge and no margin loss, in training and at inference.
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'P':
            atoms[atom.GetIdx()] = 0
    # Check chemistry as well as length: silent atom reordering changes requested R/S.
    from omtra.constants import lig_atom_type_map, charge_map
    from omtra.data.condensed_atom_typing import CondensedAtomTyper
    nd = g.nodes['lig'].data
    if 'cond_a_1_true' in nd:
        feats = CondensedAtomTyper(fake_atoms=False).cond_a_to_feats(nd['cond_a_1_true'].cpu().numpy())
        elements, charges = feats['a'], np.asarray(charge_map)[feats['c']]
    else:
        elements = nd['a_1_true'].cpu().numpy()
        charges = np.asarray(charge_map)[nd['c_1_true'].cpu().numpy()]
    if any(mol.GetAtomWithIdx(i).GetSymbol() != lig_atom_type_map[int(a)] or
           mol.GetAtomWithIdx(i).GetFormalCharge() != int(charges[i]) for i, a in enumerate(elements)):
        raise ValueError('Supplied stereo identity atom order/chemistry does not match graph')
    src, dst = g.edges(etype='lig_to_lig')
    orders = g.edges['lig_to_lig'].data['e_1_true'].cpu().tolist()
    graph_mol = molecule_from_graph(g)
    for a, b, order in zip(src.cpu().tolist(), dst.cpu().tolist(), orders):
        bond = mol.GetBondBetweenAtoms(a, b)
        graph_bond = graph_mol.GetBondBetweenAtoms(a, b)
        actual = 0 if bond is None else int(bond.GetBondTypeAsDouble())
        # Sanitization perceives aromaticity in both Kekule encodings. Equivalent
        # resonance assignments need not choose the same individual double bonds.
        if bond is not None and graph_bond is not None and bond.GetIsAromatic() and graph_bond.GetIsAromatic():
            continue
        if actual != order:
            raise ValueError('Supplied identity bonds do not match graph atom mapping')
    from omtra.models.embeddings.implicit_h import h_site_metadata
    h_eligible, is_hydrogen = h_site_metadata(mol)
    nd['h_eligible'] = torch.tensor(h_eligible, dtype=torch.float32, device=g.device)
    nd['is_hydrogen'] = torch.tensor(is_hydrogen, dtype=torch.float32, device=g.device)
    nd['stereo_atom'] = torch.as_tensor(atoms, device=g.device)
    nd['stereo_centre'] = ((nd['stereo_atom'] >= 2) & (nd['stereo_atom'] <= 5)).long()
    nd['stereo_source'] = torch.full((len(atoms),), SOURCES[source], dtype=torch.long, device=g.device)
    g.edges['lig_to_lig'].data['stereo_bond'] = torch.tensor(
        [bonds.get(tuple(sorted((a, b))), 0) for a, b in zip(src.cpu().tolist(), dst.cpu().tolist())],
        dtype=torch.long, device=g.device)
    from omtra.data.stereo_relational import attach_relational
    attach_relational(g, mol, atoms, bonds, source)
    from omtra.data.stereo_units import attach_unit_signs
    g.stereo_unit_info = attach_unit_signs(g, mol, source)
    return g


def hide_stereo(g, p_all=0.0, p_partial=0.0, partial_rate=0.5, generator=None):
    """Training-only stereo hiding: replace requested R/S and E/Z labels by 'unspecified'.

    Per molecule: with probability p_all every specified label is hidden (the "stereo unknown"
    mode), with probability p_partial each centre/double bond is hidden independently with
    probability partial_rate, otherwise nothing is hidden. Bond decisions are shared by both
    directions of an undirected bond. The true centre mask `stereo_centre` is never changed.
    Returns 0 (none), 1 (partial) or 2 (all).
    """
    if p_all < 0 or p_partial < 0 or p_all + p_partial > 1 or not 0 <= partial_rate <= 1:
        raise ValueError('Invalid stereo hiding probabilities')
    nd, ed = g.nodes['lig'].data, g.edges['lig_to_lig'].data
    if (p_all == 0 and p_partial == 0) or 'stereo_atom' not in nd:
        return 0
    atoms, bonds = nd['stereo_atom'], ed['stereo_bond']
    spec_a = (atoms >= 2) & (atoms <= 5)
    spec_b = (bonds >= 2) & (bonds <= 3)
    if not (bool(spec_a.any()) or bool(spec_b.any())):
        return 0
    u = float(torch.rand((), generator=generator))
    if u < p_all:
        mode, hide_a, hide_b = 2, spec_a, spec_b
    elif u < p_all + p_partial:
        mode = 1
        hide_a = spec_a & (torch.rand(atoms.shape, generator=generator).to(atoms.device) < partial_rate)
        src, dst = g.edges(etype='lig_to_lig')
        key = torch.minimum(src, dst).long() * (g.num_nodes('lig') + 1) + torch.maximum(src, dst).long()
        _, inverse = torch.unique(key.cpu(), return_inverse=True)
        per_bond = torch.rand(int(inverse.max()) + 1 if len(inverse) else 0, generator=generator) < partial_rate
        hide_b = spec_b & per_bond[inverse].to(bonds.device)
    else:
        return 0
    atoms[hide_a] = 1
    bonds[hide_b] = 1
    nd['stereo_hidden'] = (nd['stereo_hidden'].bool() | hide_a).long()   # keeps centres already hidden by hide_unverified (all-zero before, so unchanged for the other sources)
    from omtra.data.stereo_relational import refresh_relational_edges
    refresh_relational_edges(g)
    return mode



def molecule_from_graph(g, geometry=False):
    """Reconstruct clean supervised chemistry; callers decide if reference access is allowed."""
    from omtra.constants import lig_atom_type_map, charge_map
    from omtra.data.condensed_atom_typing import CondensedAtomTyper
    nd = g.nodes['lig'].data
    if 'cond_a_1_true' in nd:
        feats = CondensedAtomTyper(fake_atoms=False).cond_a_to_feats(nd['cond_a_1_true'].cpu().numpy())
        elements, charges = feats['a'], np.asarray(charge_map)[feats['c']]
    else:
        elements, charges = nd['a_1_true'].cpu().numpy(), np.asarray(charge_map)[nd['c_1_true'].cpu().numpy()]
    mol = Chem.RWMol()
    for element, charge in zip(elements, charges):
        atom = Chem.Atom(lig_atom_type_map[int(element)])
        atom.SetFormalCharge(int(charge))
        mol.AddAtom(atom)
    src, dst = g.edges(etype='lig_to_lig')
    for a, b, order in zip(src.cpu().tolist(), dst.cpu().tolist(), g.edges['lig_to_lig'].data['e_1_true'].cpu().tolist()):
        if a < b and order:
            mol.AddBond(a, b, (None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)[order])
    mol = mol.GetMol()
    Chem.SanitizeMol(mol)
    if geometry:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, xyz in enumerate(nd['x_1_true'].cpu().tolist()):
            conf.SetAtomPosition(i, xyz)
        mol.AddConformer(conf)
        Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
    return mol


def attach_geometry(g, task, source='training_geometry'):
    if not eligible(task):
        clear_stereo(g)
        return g
    return attach_identity(g, task, molecule_from_graph(g, geometry=True), source)


def validate_inference(g, task, allow_oracle=False):
    if not eligible(task):
        clear_stereo(g)
        return
    nd = g.nodes['lig'].data
    if 'stereo_atom' not in nd or 'stereo_source' not in nd or 'stereo_bond' not in g.edges['lig_to_lig'].data:
        raise ValueError('Stereo model requires supplied ligand identity with explicit stereo (including unspecified)')
    allowed = (1, 3) if allow_oracle else (1,)
    if not all(int(v) in allowed for v in nd['stereo_source'].unique()):
        raise ValueError('Reference/training geometry is not an authorized inference identity source')


def hide_unverified(g, keep_atoms, keep_bonds):
    """Matched-geometry training labels: keep a pose-derived label only for the listed centres (graph atom indices) and E/Z bonds (undirected (i, j) pairs);
    every other specified label becomes 'unspecified'. The true centre mask `stereo_centre` is never changed, so the chirality loss still sees every centre.
    Returns (atoms_hidden, bonds_hidden) counts of undirected centres/bonds."""
    nd, ed = g.nodes['lig'].data, g.edges['lig_to_lig'].data
    if 'stereo_atom' not in nd:
        return 0, 0
    atoms, bonds = nd['stereo_atom'], ed['stereo_bond']
    spec_a = (atoms >= 2) & (atoms <= 5)
    spec_b = (bonds >= 2) & (bonds <= 3)
    keep_a = torch.zeros_like(spec_a)
    if len(keep_atoms):
        keep_a[torch.as_tensor(list(keep_atoms), dtype=torch.long, device=atoms.device)] = True
    src, dst = g.edges(etype='lig_to_lig')
    keep_set = {(min(int(i), int(j)), max(int(i), int(j))) for i, j in keep_bonds}
    keep_b = torch.tensor([(min(a, b), max(a, b)) in keep_set for a, b in zip(src.cpu().tolist(), dst.cpu().tolist())], dtype=torch.bool, device=bonds.device)
    hide_a, hide_b = spec_a & ~keep_a, spec_b & ~keep_b
    n_atoms, n_bonds = int(hide_a.sum()), int(hide_b.sum()) // 2
    atoms[hide_a] = 1
    bonds[hide_b] = 1
    nd['stereo_hidden'] = (nd['stereo_hidden'].bool() | hide_a).long()
    from omtra.data.stereo_relational import refresh_relational_edges
    refresh_relational_edges(g)
    return n_atoms, n_bonds
