"""Open-valence direction vectors (David's implicit-hydrogen idea).

For every ligand atom the vector -sum_j unit(x_j - x_i) over its covalent neighbours points toward the free
valence: the implicit-hydrogen direction on a stereocentre with one hydrogen (the bisector for CH2).
It is a deterministic, rotation-equivariant function of the current coordinates x_t and the fixed bond graph,
so it carries no information the heavy-atom geometry does not already hold; it only makes the fourth
tetrahedral vertex explicit as a vector feature.
"""
import torch


def open_valence_vectors(g, x, etype='lig_to_lig', bond_key='e_t'):
    src, dst = g.edges(etype=etype)
    covalent = g.edges[etype].data[bond_key] > 0
    s, d = src[covalent], dst[covalent]
    out = torch.zeros_like(x)
    if s.numel() == 0:
        return out
    v = x[d] - x[s]
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    out.index_add_(0, s, -v)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Masked, bounded implicit-hydrogen direction. Only sp3 centres with three heavy single-bonded neighbours and
# exactly one hydrogen in total carry the feature; it is zero everywhere else. Eligibility is a property of the fixed chemistry
# alone (no stereo tags, no conformer, nothing label-dependent) and is attached to the graph as node data `h_eligible` and
# `is_hydrogen` by omtra.data.requested_stereo.attach_identity, identically in training and inference.
# ---------------------------------------------------------------------------------------------------------------------
H_EPS = 1e-3


def h_site_metadata(mol):
    """Per-atom (h_eligible, is_hydrogen) float lists in graph order from chemistry only (RDKit total-H semantics)."""
    from rdkit import Chem
    work = Chem.Mol(mol)
    Chem.SanitizeMol(work)
    eligible, is_h = [], []
    for atom in work.GetAtoms():
        z = atom.GetAtomicNum()
        is_h.append(1.0 if z == 1 else 0.0)
        if z <= 1:
            eligible.append(0.0)
            continue
        neighbours = list(atom.GetNeighbors())
        heavy = [n for n in neighbours if n.GetAtomicNum() > 1]
        hydrogens = [n for n in neighbours if n.GetAtomicNum() == 1]
        ok = (not atom.GetIsAromatic()
              and atom.GetHybridization() == Chem.HybridizationType.SP3
              and atom.GetNumRadicalElectrons() == 0
              and len(heavy) == 3 and len(hydrogens) <= 1
              and all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds())
              and atom.GetTotalNumHs(includeNeighbors=True) == 1)
        eligible.append(1.0 if ok else 0.0)
    return eligible, is_h


def h_directions(g, x, etype='lig_to_lig', bond_key='e_t', eps=H_EPS):
    """Bounded dimensionless H direction at eligible ligand atoms of a (batched) graph; zeros elsewhere.

    Virtual-H site: q = -sum_j u_ij over its three heavy neighbours, H = q / max(1, |q|) (never normalised up).
    Actual-H site: the unit direction to the hydrogen neighbour. Rotation-equivariant, reflection-equivariant (a polar
    vector); it adds no information about the requested enantiomer, it makes the fourth tetrahedral vertex explicit.
    """
    nd = g.nodes['lig'].data
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    if 'h_eligible' not in nd or g.num_edges(etype) == 0:
        return out
    if not bool(torch.isfinite(x).all()):
        raise ValueError('Non-finite ligand coordinates reached the hydrogen-direction feature')
    src, dst = g.edges(etype=etype)
    covalent = g.edges[etype].data[bond_key] > 0
    s, d = src[covalent], dst[covalent]
    if s.numel() == 0:
        return out
    xf = x.float()
    v = xf[d] - xf[s]
    u = v / torch.sqrt((v * v).sum(-1, keepdim=True) + eps ** 2)
    to_h = nd['is_hydrogen'][d] > 0.5
    q = torch.zeros_like(out)
    q.index_add_(0, s[~to_h], -u[~to_h])
    h_dir = torch.zeros_like(out)
    h_dir.index_add_(0, s[to_h], u[to_h])
    has_h = torch.zeros(x.shape[0], device=x.device).index_add_(0, s[to_h], torch.ones(int(to_h.sum()), device=x.device)) > 0
    virtual = q / q.norm(dim=-1, keepdim=True).clamp_min(1.0)
    out = torch.where(has_h.unsqueeze(-1), h_dir, virtual)
    return out * nd['h_eligible'].float().unsqueeze(-1)
