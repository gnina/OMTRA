"""Stereo-unit geometry shared by the diagnostics and the (default-off) margin loss. Not a registered loss.

Implements the signed-volume definitions: 3-substituent centres use the determinant of the
three normalised centre-to-neighbour vectors; 4-substituent centres use the tetrahedron of the four substituents scaled by the
mean squared edge length. Computed in the dtype given (diagnostics use float64, training FP32) with eps = 1e-3 A.
Neighbour order is supplied by the caller and must be the SAME for the pose and the reference (atom index order is used by the
diagnostics; the loss uses the graph's edge order, as in `chirality.py`). Pure functions, no state.
"""
from typing import List, Tuple
import numpy as np
import torch

EPS = 1.0e-3
PLANAR_MIN = 0.05  # |chi| below this at the reference counts as nearly planar and is excluded from sign supervision/analysis


def signed_volume3(x: torch.Tensor, centre: torch.Tensor, nbrs3: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """x (..., N, 3); centre (K,); nbrs3 (K, 3). Returns (..., K)."""
    # 3-NEIGHBOUR CENTRES (97% of centres: three heavy neighbours + one implicit hydrogen that is not a graph node).
    # Step 1: the three bond vectors, from the centre to each of its neighbours a, b, c (in the order given by the caller).
    r = x[..., nbrs3, :] - x[..., centre, :].unsqueeze(-2)           # (..., K, 3, 3)
    # Step 2: make them unit vectors (eps keeps the length finite if two atoms coincide). Only the DIRECTIONS of the bonds matter,
    # so chi does not change if a bond is stretched; this is why each neighbour's correction arrow ends up perpendicular to its own bond.
    r = r / torch.sqrt((r * r).sum(-1, keepdim=True) + eps * eps)
    # Step 3: the triple product (a x b) . c of the three unit vectors = a signed volume. Its SIGN says on which side of the triangle
    # a-b-c the centre sits (= the hand; the hidden hydrogen is on that same side). The size is about +-0.77 for an ideal tetrahedral centre,
    # 0 when the centre lies in the plane of its neighbours (flat). Swapping two neighbours in the list flips the sign.
    return (r[..., 0, :] * torch.cross(r[..., 1, :], r[..., 2, :], dim=-1)).sum(-1)


def signed_volume4(x: torch.Tensor, nbrs4: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """x (..., N, 3); nbrs4 (K, 4) substituent indices (a, b, c, d). Returns (..., K)."""
    # 4-NEIGHBOUR CENTRES (about 2-3% of the carbon centres; in the evaluation sets most 4-neighbour units are phosphorus): all four neighbours a, b, c, d are real atoms.
    # The CENTRE'S OWN POSITION IS NOT USED here, so the centre never receives a correction arrow in this case.
    # p = positions of the four neighbours.
    p = x[..., nbrs4, :]                                              # (..., K, 4, 3)
    # Edges from the fourth neighbour d to a, b, c.
    e = p[..., :3, :] - p[..., 3:4, :]
    # Scalar triple product of those edges = signed volume of the tetrahedron a-b-c-d (up to a constant factor). Sign = hand.
    raw = (e[..., 0, :] * torch.cross(e[..., 1, :], e[..., 2, :], dim=-1)).sum(-1)
    # Squared distance between every pair of neighbours (each of the 6 pairs appears twice in this 4 x 4 table).
    d2 = (p.unsqueeze(-2) - p.unsqueeze(-3)).pow(2).sum(-1)           # (..., K, 4, 4), each unordered pair twice
    # Mean squared edge length: 6 pairs x 2 = 12 entries.
    mean_sq = d2.sum((-1, -2)) / 12.0
    # Divide by (mean squared edge)^1.5, i.e. by length^3, so the result does not depend on the size of the tetrahedron.
    # Ideal regular tetrahedron: +-0.7071. A perfectly regular tetrahedron is the MAXIMUM of |chi|, so the gradient there is exactly zero.
    return raw / (mean_sq + eps * eps).pow(1.5)


def chi_for_units(x: torch.Tensor, centres: List[int], neighbours: List[List[int]], eps: float = EPS) -> torch.Tensor:
    """chi for a mixed list of centres, each with 3 or 4 ordered neighbours. Returns (..., K) in input order."""
    # Mixed list of centres: 3-neighbour centres use signed_volume3, 4-neighbour centres use signed_volume4. Result is returned in input order.
    out = [None] * len(centres)
    three = [i for i, n in enumerate(neighbours) if len(n) == 3]
    four = [i for i, n in enumerate(neighbours) if len(n) == 4]
    shape = x.shape[:-2]
    result = torch.zeros(*shape, len(centres), dtype=x.dtype, device=x.device)
    if three:
        c = torch.tensor([centres[i] for i in three], device=x.device)
        n = torch.tensor([neighbours[i] for i in three], device=x.device)
        result[..., three] = signed_volume3(x, c, n, eps)
    if four:
        n = torch.tensor([neighbours[i] for i in four], device=x.device)
        result[..., four] = signed_volume4(x, n, eps)
    return result


def tetrahedral_units(mol, include_phosphorus: bool = False) -> Tuple[List[int], List[List[int]]]:
    """Reference tetrahedral centres of an RDKit mol with 3D coordinates (RDKit assignment from the mol's own geometry),
    with index-sorted neighbour lists of length 3 or 4 (centres with other degrees are skipped)."""
    from rdkit import Chem
    work = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(work, replaceExistingTags=True)
    # RDKit decides which atoms are tetrahedral stereocentres, from the molecule's OWN 3D geometry (used for the reference/crystal ligand).
    found = Chem.FindMolChiralCenters(work, includeUnassigned=False, useLegacyImplementation=False)
    centres, nbrs = [], []
    for i, _ in found:
        atom = work.GetAtomWithIdx(i)
        # Phosphorus is skipped by default: phosphate stereo tags depend on which equivalent oxygen is drawn as P=O, so they are mostly an artefact.
        if atom.GetSymbol() == 'P' and not include_phosphorus:
            continue
        # Neighbours in ascending atom-index order. The SAME order must be used for the target and for the pose, so the arbitrary +/- convention cancels.
        n = sorted(a.GetIdx() for a in atom.GetNeighbors())
        if len(n) in (3, 4):
            centres.append(i)
            nbrs.append(n)
    return centres, nbrs


def reference_signs(x_true: np.ndarray, centres: List[int], nbrs: List[List[int]]):
    """(sign, chi, usable) of the reference; usable excludes nearly planar references."""
    chi = chi_for_units(torch.as_tensor(np.asarray(x_true), dtype=torch.float64), centres, nbrs).numpy()
    return np.sign(chi), chi, np.abs(chi) >= PLANAR_MIN


# ---------------------------------------------------------------------------------------------------------------------
# Graph-level helpers (batched DGL ligand graphs; neighbour order = ascending atom index, the same order the diagnostics use).
# ---------------------------------------------------------------------------------------------------------------------
def chi_tensor(x: torch.Tensor, centres: torch.Tensor, nbrs4: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """chi for units given as tensors. x (N,3); centres (K,); nbrs4 (K,4) with -1 marking an absent fourth substituent
    (three-substituent centre). Returns (K,)."""
    if centres.numel() == 0:
        return x.new_zeros(0)
    # nbrs4 has 4 columns; a -1 in the last column means 'three-neighbour centre' (the fourth bond is the implicit hydrogen).
    four = nbrs4[:, 3] >= 0
    # Start by computing the 3-neighbour formula for every centre (centre + first three neighbours) ...
    chi = signed_volume3(x, centres, nbrs4[:, :3], eps)
    if bool(four.any()):
        # ... and overwrite the 4-neighbour centres with the tetrahedron formula.
        chi4 = signed_volume4(x, nbrs4[four], eps)
        chi = chi.clone()
        chi[four] = chi4
    return chi


def graph_units(g):
    """(centres, nbrs4) of the true tetrahedral centres of a (batched) ligand graph: covalent degree 3 or 4 centres flagged `stereo_centre`.
    Neighbours ascending by index. Requires node data `stereo_centre`, `x_1_true` (device only) and edge data `e_1_true`."""
    from omtra.aux_losses.chirality import centre_neighbours
    return centre_neighbours(g, include_fourth=True)


def unit_signs_from_coords(g, x: torch.Tensor):
    """Handedness sign (+1/-1, 0 when nearly planar) of every graph unit at coordinates x, plus (centres, nbrs4)."""
    centres, nbrs = graph_units(g)
    chi = chi_tensor(x.double(), centres, nbrs)
    # Hand = sign of chi; nearly flat centres (|chi| < 0.05) get 0 = 'no hand', so they get no arrow and no loss.
    z = torch.where(chi.abs() >= PLANAR_MIN, torch.sign(chi), torch.zeros_like(chi))
    return z, centres, nbrs


def margin_unit_loss(chi: torch.Tensor, z: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """Squared normalised hinge [max(0, m - z*chi)/m]^2 per unit; units with z == 0 (hidden / nearly planar) contribute 0."""
    # chi = measured hand of the pose (signed volume), z = requested hand (+1 / -1), margin m = 0.2.
    #   shortfall = max(0, m - z*chi)    0 when the centre is clearly on the right side (z*chi >= 0.2), 0.2 when flat, up to ~0.97 when fully inverted
    #   penalty   = (shortfall / m)^2    divided by m so that a flat centre costs exactly 1; squared so that big mistakes cost much more
    # Examples: z*chi = +0.77 (ideal, right hand) -> 0;  z*chi = 0 (flat) -> 1;  z*chi = -0.77 (mirror image) -> ((0.2 + 0.77) / 0.2)^2 = 23.5.
    # Units with z == 0 (label hidden, or reference nearly flat) contribute nothing.
    return (torch.clamp(margin - z * chi, min=0.0) / margin) ** 2 * (z != 0).to(chi.dtype)
