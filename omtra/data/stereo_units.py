"""Handedness signs of the requested configuration at every tetrahedral unit (input to the optional stereo-correction features).

The sign z_i = sign(chi_i(x_ref)) uses the same signed volume and the same ascending-atom-index neighbour order as the correction module and the margin
loss, so it is consistent with whatever atom order the graph has (odd neighbour permutations change the sign convention together with the neighbour order).
Reference coordinates:
  * training_geometry / oracle_reference : the graph's own crystal coordinates (x_1_true), exactly like the training labels;
  * supplied identity                    : an RDKit ETKDG conformer of the SUPPLIED molecule only (never any pose). The conformer is then checked
    independently: RDKit chiral tags re-derived from its 3D structure must equal the requested tags at EVERY unit; units where they differ get no sign
    (z = 0) and are counted, so an arbitrary conformer can never invent a requested configuration. Units that are undefined in the identity are not
    units at all (stereo_centre = 0) and therefore never receive a sign.
Nearly planar reference units (|chi| < 0.05) also get z = 0.
"""
import numpy as np
import torch
from rdkit import Chem

from omtra.aux_losses.stereo_geometry import graph_units, chi_tensor, PLANAR_MIN
from rdkit.Chem import AllChem


def supplied_conformer(mol, seed=0):
    """ETKDG conformer of the supplied molecule in the SAME atom order as `mol` (explicit hydrogens that are graph nodes are kept). None on failure."""
    # Add explicit hydrogens so RDKit can build a physically sensible 3D structure; they are dropped again below (they are not graph nodes).
    work = Chem.AddHs(Chem.Mol(mol))
    # ETKDG: RDKit's distance-geometry conformer generator, with a fixed random seed so the result is reproducible.
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    # If the first embedding fails, retry with random starting coordinates; return None if that fails too (then no centre gets a sign).
    if AllChem.EmbedMolecule(work, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(work, params) != 0:
            return None
    # Keep only the original heavy atoms, in the original atom order (AddHs appends the new hydrogens at the END).
    xyz = np.asarray(work.GetConformer().GetPositions(), dtype=np.float64)[:mol.GetNumAtoms()]   # AddHs appends new hydrogens at the end
    return xyz if xyz.shape[0] == mol.GetNumAtoms() else None


def conformer_matches_tags(mol, xyz, atoms):
    """Per-atom bool: tags re-derived from the 3D coordinates equal the requested chiral tags (same mol object => same neighbour order)."""
    # Safety check of the conformer: copy the molecule, give it the conformer's coordinates, and let RDKit re-derive every chiral tag from that 3D structure.
    work = Chem.Mol(mol)
    work.RemoveAllConformers()
    conf = Chem.Conformer(work.GetNumAtoms())
    for i in range(work.GetNumAtoms()):
        conf.SetAtomPosition(i, [float(v) for v in xyz[i]])
    work.AddConformer(conf, assignId=True)
    # RDKit's own reading of the hand from the coordinates; it must equal the requested (supplied) tag at every unit, otherwise that unit gets no sign.
    Chem.AssignAtomChiralTagsFromStructure(work, replaceExistingTags=True)
    return {int(i): work.GetAtomWithIdx(int(i)).GetChiralTag() == mol.GetAtomWithIdx(int(i)).GetChiralTag() for i in atoms}


def attach_unit_signs(g, mol, source, conformer_seed=0):
    # Output: z_full[atom] = +1 / -1 at stereocentres, 0 everywhere else; 'info' counts what happened (for auditing).
    nd = g.nodes['lig'].data
    z_full = torch.zeros(g.num_nodes('lig'), dtype=torch.float32, device=g.device)
    info = {'units': 0, 'signed': 0, 'planar': 0, 'tag_mismatch': 0, 'embedding_failed': 0}
    if 'stereo_centre' not in nd or not bool(nd['stereo_centre'].any()):
        nd['stereo_unit_z'] = z_full
        return info
    centres, nbrs = graph_units(g)
    info['units'] = int(centres.numel())
    if centres.numel() == 0:
        nd['stereo_unit_z'] = z_full
        return info
    # keep[i] = this centre passed all checks (starts True; the conformer tag check can switch it off).
    keep = torch.ones(centres.numel(), dtype=torch.bool, device=g.device)
    # Where the reference geometry comes from: for training the hand is read from the crystal coordinates; for a SUPPLIED identity (inference)
    # it is read from a conformer of the supplied molecule, never from any pose.
    if source in ('training_geometry', 'oracle_reference') and 'x_1_true' in nd:
        xyz = nd['x_1_true'].double()
    else:
        coords = supplied_conformer(mol, seed=conformer_seed)
        if coords is None:
            info['embedding_failed'] = info['units']
            nd['stereo_unit_z'] = z_full
            return info
        # The conformer must reproduce the requested R/S tag at every unit; units where it does not are not signed (z = 0) and are counted.
        ok = conformer_matches_tags(mol, coords, centres.tolist())
        keep = torch.tensor([ok[int(c)] for c in centres.tolist()], dtype=torch.bool, device=g.device)
        info['tag_mismatch'] = int((~keep).sum())
        xyz = torch.as_tensor(coords, dtype=torch.float64, device=g.device)
    # chi of each centre in the reference geometry, using the same formula and neighbour order as the correction feature and the margin loss.
    chi = chi_tensor(xyz, centres, nbrs)
    # Reference centre nearly flat: its sign would be arbitrary, so it is not signed either.
    planar = chi.abs() < PLANAR_MIN
    info['planar'] = int((planar & keep).sum())
    good = keep & ~planar
    # z = +1 / -1 = sign of chi at the centre atom's index; this is the target hand used by the arrows.
    z_full[centres[good]] = torch.sign(chi[good]).float()
    info['signed'] = int(good.sum())
    nd['stereo_unit_z'] = z_full
    return info
