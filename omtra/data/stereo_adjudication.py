"""CIP-free comparison of a supplied (annotation) stereo request with crystal coordinates, per centre.

Tags are compared on the SAME molecule object (same bond order), once as annotated and once re-assigned from the coordinates, so no CIP ranking or atom
mapping choice is involved. A centre is `pseudo` when its neighbours tie once bond orders, charges and H counts are ignored (phosphate oxygens that
are equivalent up to protonation/resonance): its tag then depends only on which equivalent oxygen was drawn as P=O, carries no chemical stereo and is
reported separately from `real` centres. Nothing here edits an identity or a target.
"""
from rdkit import Chem

TETRA = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
BOND_STEREO = (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS)


def tautomer_blind_ties(mol, centre):
    """True if the centre's neighbours are indistinguishable once bond order, charge and H count are ignored."""
    work = Chem.RWMol(mol)
    for bond in work.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in work.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
        atom.SetIsAromatic(False)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    ranks = list(Chem.CanonicalRankAtoms(work, breakTies=False, includeChirality=False))
    neighbours = [n.GetIdx() for n in mol.GetAtomWithIdx(centre).GetNeighbors()]
    # Only terminal same-element neighbours (e.g. the oxygens of a phosphate) can be equivalent through protonation or
    # tautomerism. Non-terminal substituents that differ only by bond order or hydrogen count (vinyl vs ethyl, CHO vs
    # CH2OH) are genuinely different groups and keep their stereo label.
    terminal = [n for n in neighbours if mol.GetAtomWithIdx(n).GetDegree() == 1]
    return len({ranks[n] for n in terminal}) < len(terminal)


def compare_with_coordinates(identity, xyz):
    """identity: heavy-atom molecule in graph order with the annotated stereo; xyz: (n, 3) crystal coordinates in the same order.

    Returns {'atoms': [{atom, element, status, pseudo}], 'bonds': [{bond, agree}]} with status in agree | disagree | geometry_unspecified.
    """
    geometry = Chem.Mol(identity)
    geometry.RemoveAllConformers()
    conformer = Chem.Conformer(geometry.GetNumAtoms())
    for i in range(geometry.GetNumAtoms()):
        conformer.SetAtomPosition(i, [float(v) for v in xyz[i]])
    geometry.AddConformer(conformer, assignId=True)
    Chem.AssignStereochemistryFrom3D(geometry, replaceExistingTags=True)
    atoms, bonds = [], []
    for atom in identity.GetAtoms():
        if atom.GetChiralTag() not in TETRA:
            continue
        i = atom.GetIdx()
        observed = geometry.GetAtomWithIdx(i).GetChiralTag()
        status = 'agree' if observed == atom.GetChiralTag() else ('disagree' if observed in TETRA else 'geometry_unspecified')
        atoms.append({'atom': i, 'element': atom.GetSymbol(), 'status': status, 'pseudo': tautomer_blind_ties(identity, i)})
    for bond in identity.GetBonds():
        if bond.GetStereo() in BOND_STEREO:
            observed = geometry.GetBondWithIdx(bond.GetIdx()).GetStereo()
            bonds.append({'bond': bond.GetIdx(), 'agree': str(observed) == str(bond.GetStereo())})
    return {'atoms': atoms, 'bonds': bonds}


def classify(comparison):
    """agree | pseudo_only (only non-stereogenic phosphate-type tags disagree) | real (a stereogenic centre or E/Z bond disagrees)."""
    real = any(a['status'] != 'agree' and not a['pseudo'] for a in comparison['atoms']) or any(not b['agree'] for b in comparison['bonds'])
    pseudo = any(a['status'] != 'agree' and a['pseudo'] for a in comparison['atoms'])
    return 'real' if real else ('pseudo_only' if pseudo else 'agree')


def strip_pseudo_tags(mol):
    """Copy of `mol` with the chiral tags of tautomer-blind (non-stereogenic) centres removed; every other tag is kept.

    Phosphate-type tags depend only on which of several equivalent oxygens was drawn as P=O, so comparing them with a reference or a pose is a coin
    flip. Removing them from a *request used for scoring* leaves the stereogenic constraints; it does not change the identity supplied to a model.
    """
    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        if atom.GetChiralTag() in TETRA and tautomer_blind_ties(mol, atom.GetIdx()):
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return out
