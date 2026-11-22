"""Relational (RitS-style) stereo labels on the complete ligand graph.

Instead of a per-atom R/S token, handedness is written on edges between the neighbours of each
stereocentre. Neighbours are ordered by an atom-order-independent canonical rank computed on the
stereo-stripped graph (no CIP convention is involved). For a centre with ordered neighbours
(n0, n1, n2[, anchor]) the sign of the signed volume of (n0, n1, n2) around the centre is
+1 (counter-clockwise from the anchor / implicit hydrogen) or -1:

    channel 0  `cw` count   : edge u->v is a forward edge of the neighbour cycle
    channel 1  `ccw` count  : edge u->v is the reverse of a forward edge
    channel 2  anchor       : the fourth (lowest-ranked) neighbour of a 4-neighbour centre, to n0..n2
    channel 3  same side    : the two highest-ranked substituents across a double bond are cis
    channel 4  opposite     : ... are trans

The sign comes from the crystal geometry when training and from an RDKit conformer of the supplied
molecule at inference, so it never touches pose coordinates of the thing being generated. Centres whose
neighbours tie in rank, or whose reference geometry is nearly planar, receive no label.
"""
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

N_CHANNELS = 5
MIN_VOLUME = 0.15


def canonical_ranks(mol):
    stripped = Chem.Mol(mol)
    Chem.RemoveStereochemistry(stripped)
    return list(Chem.CanonicalRankAtoms(stripped, breakTies=False, includeChirality=False))


def reference_coords(mol, source, seed=0):
    """Coordinates whose local handedness defines the relational labels."""
    if source in ('training_geometry', 'oracle_reference') and mol.GetNumConformers():
        return np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    work = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(work, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(work, params) != 0:
            return None
    work = Chem.RemoveHs(work)
    return np.asarray(work.GetConformer().GetPositions(), dtype=np.float64)


def _volume(xyz, c, n):
    v = xyz[list(n)] - xyz[c]
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    return float(np.dot(v[0], np.cross(v[1], v[2])))


def _torsion(xyz, a, b, c, d):
    b0, b1, b2 = xyz[a] - xyz[b], xyz[c] - xyz[b], xyz[d] - xyz[c]
    b1n = b1 / max(np.linalg.norm(b1), 1e-9)
    v, w = b0 - np.dot(b0, b1n) * b1n, b2 - np.dot(b2, b1n) * b1n
    return float(np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w)))


def relational_structure(mol, atom_labels, bond_labels, xyz):
    """Per-centre neighbour order + sign, and per-double-bond substituent pair + cis/trans.

    atom_labels: ints in requested_stereo.ATOM_LABELS order (2..5 = specified centre);
    bond_labels: dict {(i, j) sorted: label} with 2 = E, 3 = Z (CIP names only select which bonds are stereo).
    Returns (centres, bonds, skipped) where skipped counts centres/bonds that could not be labelled.
    """
    ranks = canonical_ranks(mol)
    centres, bonds, skipped = {}, {}, {'tie': 0, 'planar': 0, 'degree': 0}
    for atom in mol.GetAtoms():
        c = atom.GetIdx()
        if not 2 <= int(atom_labels[c]) <= 5:
            continue
        nbrs = sorted((n.GetIdx() for n in atom.GetNeighbors()), key=lambda i: -ranks[i])
        if len(nbrs) not in (3, 4):
            skipped['degree'] += 1
            continue
        if len({ranks[i] for i in nbrs}) != len(nbrs):
            skipped['tie'] += 1
            continue
        vol = _volume(xyz, c, nbrs[:3])
        if abs(vol) < MIN_VOLUME:
            skipped['planar'] += 1
            continue
        centres[c] = (nbrs[0], nbrs[1], nbrs[2], nbrs[3] if len(nbrs) == 4 else -1, 1 if vol > 0 else -1)
    for (i, j), label in bond_labels.items():
        if label not in (2, 3):
            continue
        sides = []
        for centre, other in ((i, j), (j, i)):
            subs = [n.GetIdx() for n in mol.GetAtomWithIdx(centre).GetNeighbors() if n.GetIdx() != other]
            subs.sort(key=lambda k: -ranks[k])
            if not subs or (len(subs) == 2 and ranks[subs[0]] == ranks[subs[1]]):
                sides = None
                break
            sides.append(subs[0])
        if sides is None:
            skipped['tie'] += 1
            continue
        angle = _torsion(xyz, sides[0], i, j, sides[1])
        bonds[(i, j)] = (sides[0], sides[1], 1 if abs(angle) < np.pi / 2 else -1)
    return centres, bonds, skipped


def _empty(g):
    n, e = g.num_nodes('lig'), g.num_edges('lig_to_lig')
    nd, ed = g.nodes['lig'].data, g.edges['lig_to_lig'].data
    dev = g.device
    nd['stereo_rel_nb'] = torch.full((n, 4), -1, dtype=torch.long, device=dev)
    nd['stereo_rel_sign'] = torch.zeros(n, dtype=torch.long, device=dev)
    ed['stereo_rel_ez_ids'] = torch.full((e, 2), -1, dtype=torch.long, device=dev)
    ed['stereo_rel_ez_sign'] = torch.zeros(e, dtype=torch.long, device=dev)
    ed['stereo_rel'] = torch.zeros(e, N_CHANNELS, device=dev)


def attach_relational(g, mol, atom_labels, bond_labels, source):
    """Store the per-centre structure on the (unbatched) graph and build the visible edge channels."""
    _empty(g)
    xyz = reference_coords(mol, source)
    if xyz is None:
        return {'embedding_failed': True}
    if not bool((np.asarray(atom_labels) >= 2).any()) and not bond_labels:
        return {}
    centres, bonds, skipped = relational_structure(mol, atom_labels, bond_labels, xyz)
    nd, ed = g.nodes['lig'].data, g.edges['lig_to_lig'].data
    for c, (a, b, d, anchor, sign) in centres.items():
        nd['stereo_rel_nb'][c] = torch.tensor([a, b, d, anchor], device=g.device)
        nd['stereo_rel_sign'][c] = sign
    src, dst = g.edges(etype='lig_to_lig')
    for (i, j), (a, b, sign) in bonds.items():
        mask = ((src == i) & (dst == j)) | ((src == j) & (dst == i))
        ed['stereo_rel_ez_ids'][mask] = torch.tensor([a, b], device=g.device)
        ed['stereo_rel_ez_sign'][mask] = sign
    refresh_relational_edges(g)
    return skipped


def refresh_relational_edges(g):
    """Rebuild `stereo_rel` from the *currently visible* labels (after any hiding). Unbatched graphs only."""
    nd, ed = g.nodes['lig'].data, g.edges['lig_to_lig'].data
    if 'stereo_rel_nb' not in nd:
        return
    e = g.num_edges('lig_to_lig')
    out = torch.zeros(e, N_CHANNELS)
    atoms, nb, sign = nd['stereo_atom'].cpu(), nd['stereo_rel_nb'].cpu(), nd['stereo_rel_sign'].cpu()
    visible = ((atoms >= 2) & (atoms <= 5) & (sign != 0)).nonzero().flatten()
    if len(visible):
        n, s = nb[visible], sign[visible]
        u = torch.cat([n[:, 0], n[:, 1], n[:, 2]])
        v = torch.cat([n[:, 1], n[:, 2], n[:, 0]])
        ss = s.repeat(3)
        fu, fv = torch.where(ss > 0, u, v), torch.where(ss > 0, v, u)
        forward = g.edge_ids(fu.to(g.device), fv.to(g.device), etype='lig_to_lig').cpu()
        backward = g.edge_ids(fv.to(g.device), fu.to(g.device), etype='lig_to_lig').cpu()
        out[:, 0].index_add_(0, forward, torch.ones(len(forward)))
        out[:, 1].index_add_(0, backward, torch.ones(len(backward)))
        has_anchor = n[:, 3] >= 0
        if bool(has_anchor.any()):
            anchor = n[has_anchor, 3].repeat(3)
            ring = torch.cat([n[has_anchor, 0], n[has_anchor, 1], n[has_anchor, 2]])
            for a, b in ((anchor, ring), (ring, anchor)):
                ids = g.edge_ids(a.to(g.device), b.to(g.device), etype='lig_to_lig').cpu()
                out[:, 2].index_add_(0, ids, torch.ones(len(ids)))
    stereo_bond = ed['stereo_bond'].cpu()
    ez_sign = ed['stereo_rel_ez_sign'].cpu()
    on = ((stereo_bond >= 2) & (stereo_bond <= 3) & (ez_sign != 0)).nonzero().flatten()
    if len(on):
        ids = ed['stereo_rel_ez_ids'].cpu()[on]
        sgn = ez_sign[on]
        for a, b in ((ids[:, 0], ids[:, 1]), (ids[:, 1], ids[:, 0])):
            eid = g.edge_ids(a.to(g.device), b.to(g.device), etype='lig_to_lig').cpu()
            out[eid, 3] = (sgn > 0).float()
            out[eid, 4] = (sgn < 0).float()
    ed['stereo_rel'] = out.clamp(max=2.0).to(ed['stereo_bond'].device)
