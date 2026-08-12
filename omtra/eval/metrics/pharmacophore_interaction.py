from typing import Dict, List, Tuple
import numpy as np
from rdkit import Chem
from scipy.spatial.distance import cdist

from omtra.data.pharmacophores import (
    smarts_patterns,
    matching_types,
    matching_distance,
    interaction_distance,
    interaction_angle,
    get_smarts_matches,
    get_vectors,
    _residue_atom_id,
    _get_receptor_pharmacophores,
    _get_ligand_pharmacophores,
)
from omtra.data.pharmvec import AngleBetween, GetDonorHPositions


# --- feature extraction with vectors (for angle-aware interaction checks) ------

def _get_feature_matches(mol: Chem.Mol, feature: str) -> List[dict]:
    """All matches of `feature` in `mol`, with position + direction vector(s)
    (and, for HydrogenDonor, explicit H coordinates for angle checks).
    `mol` must already have explicit Hs with 3D coords."""
    matches_data = []
    for pattern in smarts_patterns[feature]:
        atom_idxs, atom_positions, feature_positions = get_smarts_matches(mol, pattern)
        if not feature_positions:
            continue
        vectors = get_vectors(mol, feature, atom_idxs, atom_positions, feature_positions)
        for match, pos, vecs in zip(atom_idxs, feature_positions, vectors):
            entry = {"pos": pos, "vecs": vecs, "match": match}
            if feature == "HydrogenDonor":
                entry["h_pos"] = GetDonorHPositions(match[0], mol)
            matches_data.append(entry)
    return matches_data


def _get_ligand_pharmacophore_matches(ligand: Chem.Mol, feature_types: List[str]) -> Dict[str, List[dict]]:
    """Like `_get_ligand_pharmacophores`, but also keeps direction vectors
    (and H positions for donors) needed for angle checks."""
    ligand = Chem.AddHs(Chem.Mol(ligand), addCoords=True)
    return {feature: _get_feature_matches(ligand, feature) for feature in feature_types}


def _get_receptor_pharmacophore_matches(receptor: Chem.Mol, feature_types: List[str]) -> Dict[str, List[dict]]:
    """Like `_get_receptor_pharmacophores`, but also keeps direction vectors
    (and H positions for donors), plus residue_id per match."""
    receptor = Chem.AddHs(Chem.Mol(receptor), addCoords=True)
    result: Dict[str, List[dict]] = {}
    for feature in feature_types:
        matches = _get_feature_matches(receptor, feature)
        for m in matches:
            m["residue_id"] = _residue_atom_id(receptor, m["match"][0])
        result[feature] = matches
    return result


# --- per-pairing angle checks ---------------------------------------------------

def _passes_d_h_a(donor_entry: dict, acceptor_pos: np.ndarray, lo: float, hi: float) -> bool:
    """D-H...A angle, measured at each real hydrogen on the donor. Passes if any H qualifies."""
    for h_pos in donor_entry["h_pos"]:
        v_hd = donor_entry["pos"] - h_pos
        v_ha = acceptor_pos - h_pos
        if lo <= AngleBetween(v_hd, v_ha) <= hi:
            return True
    return False


def _passes_aromatic_aromatic(entry1: dict, entry2: dict, lo: float, hi: float) -> bool:
    best = min(AngleBetween(n1, n2) for n1 in entry1["vecs"] for n2 in entry2["vecs"])
    return lo <= best <= hi


def _passes_cation_pi(aromatic_entry: dict, cation_pos: np.ndarray, lo: float, hi: float) -> bool:
    v = cation_pos - aromatic_entry["pos"]
    best = min(AngleBetween(n, v) for n in aromatic_entry["vecs"])
    return lo <= best <= hi


def _passes_halogen_bond_angles(
    halogen_entry: dict, acceptor_entry: dict, axd_lo: float, axd_hi: float, xar_lo: float, xar_hi: float
) -> bool:
    """Both AXD (at the halogen) and XAR (at the acceptor) must pass.
    XAR is approximated via the acceptor's lone-pair vector as a proxy for
    'away from R' -- we don't track the R substituent explicitly, so this
    isn't a faithful reproduction of ProLIF's XAR."""
    v_xa = acceptor_entry["pos"] - halogen_entry["pos"]
    axd_ok = any(axd_lo <= AngleBetween(-xv, v_xa) <= axd_hi for xv in halogen_entry["vecs"])
    if not axd_ok:
        return False
    v_ax = halogen_entry["pos"] - acceptor_entry["pos"]
    return any(xar_lo <= AngleBetween(lp, v_ax) <= xar_hi for lp in acceptor_entry["vecs"])


def _passes_angle(ligand_feature: str, receptor_feature: str, lig_entry: dict, rec_entry: dict, angle_range) -> bool:
    if angle_range is None or angle_range == (None, None):
        return True
    lo, hi = angle_range

    if ligand_feature == "HydrogenDonor" and receptor_feature == "HydrogenAcceptor":
        return _passes_d_h_a(lig_entry, rec_entry["pos"], lo, hi)
    if ligand_feature == "HydrogenAcceptor" and receptor_feature == "HydrogenDonor":
        return _passes_d_h_a(rec_entry, lig_entry["pos"], lo, hi)
    if ligand_feature == "Aromatic" and receptor_feature == "Aromatic":
        return _passes_aromatic_aromatic(lig_entry, rec_entry, lo, hi)
    if ligand_feature == "Aromatic" and receptor_feature == "PositiveIon":
        return _passes_cation_pi(lig_entry, rec_entry["pos"], lo, hi)
    if ligand_feature == "PositiveIon" and receptor_feature == "Aromatic":
        return _passes_cation_pi(rec_entry, lig_entry["pos"], lo, hi)
    return True  # Halogen<->HydrogenAcceptor handled by the caller (needs both AXD and XAR)


# --- distance-only interaction finding (moved from posecheck.py -- no PoseCheck dependency) --

def _find_interactions(
    ligand_pharmacophores: Dict[str, List[np.ndarray]],
    receptor_pharmacophores: Dict[str, List[Tuple[np.ndarray, tuple]]],
) -> set:
    """Pair ligand pharmacophore features against complementary receptor
    pharmacophore features (distance only) and record which residue each
    contact is on. Returns: set of (ligand_feature_type, residue_id)."""
    found = set()
    for lig_feature, lig_positions in ligand_pharmacophores.items():
        if not lig_positions:
            continue
        lig_positions_arr = np.array(lig_positions)
        paired_features = matching_types[lig_feature]
        cutoffs = matching_distance[lig_feature]

        for rec_feature, cutoff in zip(paired_features, cutoffs):
            rec_matches = receptor_pharmacophores.get(rec_feature, [])
            if not rec_matches:
                continue
            rec_positions_arr = np.array([pos for pos, _ in rec_matches])
            rec_residue_ids = [res_id for _, res_id in rec_matches]

            distances = cdist(lig_positions_arr, rec_positions_arr)
            for i in range(distances.shape[0]):
                hit_idxs = np.where(distances[i] <= cutoff)[0]
                for r_idx in hit_idxs:
                    found.add((lig_feature, rec_residue_ids[r_idx]))
    return found


def _interaction_recovery_smarts(
    gen_ligs: List[Chem.Mol],
    true_lig: Chem.Mol,
    true_prot_file: str,
) -> List[float]:
    """Distance-only SMARTS-based interaction recovery (no angle checks)."""
    recovery_feature_types = ["Aromatic", "PositiveIon", "NegativeIon", "HydrogenAcceptor", "HydrogenDonor", "Halogen"]

    receptor = Chem.MolFromPDBFile(true_prot_file, removeHs=False)
    receptor = Chem.AddHs(receptor, addCoords=True)
    receptor_pharmacophores = _get_receptor_pharmacophores(receptor, recovery_feature_types)

    true_pharmacophores = _get_ligand_pharmacophores(true_lig, recovery_feature_types)
    true_interactions = _find_interactions(true_pharmacophores, receptor_pharmacophores)

    if not true_interactions:
        return [float("nan")] * len(gen_ligs)

    recovery: List[float] = []
    for lig in gen_ligs:
        lig_pharmacophores = _get_ligand_pharmacophores(lig, recovery_feature_types)
        lig_interactions = _find_interactions(lig_pharmacophores, receptor_pharmacophores)
        recovery.append(len(true_interactions & lig_interactions) / len(true_interactions))
    return recovery


def pharmacophore_interaction_recovery(
    ligs: List[Chem.Mol], true_lig: Chem.Mol, true_prot_file: str
) -> List[float]:
    """Public entry point: distance-only SMARTS interaction recovery."""
    return _interaction_recovery_smarts(ligs, true_lig, true_prot_file)


# --- angle-aware (ProLIF-matched) interaction finding ---------------------------

def _find_interactions_strict(ligand_features: Dict[str, List[dict]], receptor_features: Dict[str, List[dict]]) -> set:
    """Like `_find_interactions`, but uses `interaction_distance` (ProLIF-derived
    cutoffs) plus `interaction_angle` geometry checks."""
    found = set()
    for lig_feature, lig_matches in ligand_features.items():
        if not lig_matches:
            continue
        paired_features = matching_types[lig_feature]
        cutoffs = interaction_distance.get(lig_feature, matching_distance[lig_feature])
        angle_ranges = interaction_angle.get(lig_feature, [None] * len(paired_features))

        for rec_feature, cutoff, angle_range in zip(paired_features, cutoffs, angle_ranges):
            rec_matches = receptor_features.get(rec_feature, [])
            if not rec_matches:
                continue
            for lig_entry in lig_matches:
                for rec_entry in rec_matches:
                    dist = np.linalg.norm(lig_entry["pos"] - rec_entry["pos"])
                    if dist > cutoff:
                        continue

                    if {lig_feature, rec_feature} == {"Halogen", "HydrogenAcceptor"}:
                        halogen_entry, acceptor_entry = (
                            (lig_entry, rec_entry) if lig_feature == "Halogen" else (rec_entry, lig_entry)
                        )
                        axd_lo, axd_hi = interaction_angle["Halogen"][1]
                        xar_lo, xar_hi = interaction_angle["HydrogenAcceptor"][1]
                        if not _passes_halogen_bond_angles(halogen_entry, acceptor_entry, axd_lo, axd_hi, xar_lo, xar_hi):
                            continue
                    elif not _passes_angle(lig_feature, rec_feature, lig_entry, rec_entry, angle_range):
                        continue

                    found.add((lig_feature, rec_entry["residue_id"]))
    return found


def _interaction_recovery_smarts_strict(
    gen_ligs: List[Chem.Mol], true_lig: Chem.Mol, true_prot_file: str
) -> List[float]:
    """Stricter SMARTS-based interaction recovery: ProLIF-derived distance
    cutoffs AND geometric angle constraints, without depending on PoseCheck."""
    recovery_feature_types = ["Aromatic", "PositiveIon", "NegativeIon", "HydrogenAcceptor", "HydrogenDonor", "Halogen"]

    receptor = Chem.MolFromPDBFile(true_prot_file, removeHs=False)
    receptor_features = _get_receptor_pharmacophore_matches(receptor, recovery_feature_types)

    true_features = _get_ligand_pharmacophore_matches(true_lig, recovery_feature_types)
    true_interactions = _find_interactions_strict(true_features, receptor_features)

    if not true_interactions:
        return [float("nan")] * len(gen_ligs)

    recovery: List[float] = []
    for lig in gen_ligs:
        lig_features = _get_ligand_pharmacophore_matches(lig, recovery_feature_types)
        lig_interactions = _find_interactions_strict(lig_features, receptor_features)
        recovery.append(len(true_interactions & lig_interactions) / len(true_interactions))
    return recovery


def pharmacophore_interaction_recovery_strict(
    ligs: List[Chem.Mol], true_lig: Chem.Mol, true_prot_file: str
) -> List[float]:
    """Public entry point: angle-aware, ProLIF-matched SMARTS interaction recovery."""
    return _interaction_recovery_smarts_strict(ligs, true_lig, true_prot_file)