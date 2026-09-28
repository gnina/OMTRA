"""Pocket RMSD: Kabsch-aligned RMSD over matched pocket heavy atoms."""

from typing import Tuple

import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io
import numpy as np


def _atom_key(chain_id, res_id, atom_name) -> Tuple[str, int, str]:
    return (str(chain_id), int(res_id), str(atom_name))


def _rmsd_from_coord_maps(true_map: dict, gen_map: dict) -> float:
    common = sorted(set(true_map) & set(gen_map))
    if not common:
        return float("nan")

    true_coords = np.stack([true_map[k] for k in common])
    gen_coords = np.stack([gen_map[k] for k in common])
    gen_aligned = _kabsch_align(gen_coords, true_coords)
    return float(np.sqrt(((gen_aligned - true_coords) ** 2).sum(axis=1).mean()))


def _heavy_coords_by_key(prot_file: str) -> dict:
    pdb = pdb_io.PDBFile.read(str(prot_file))
    arr = pdb.get_structure(model=1)
    heavy = arr[(~arr.hetero) & (arr.element != "H")]
    return {
        _atom_key(a.chain_id, a.res_id, a.atom_name): a.coord
        for a in heavy
    }


def _kabsch_align(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align mobile onto target (same atom ordering)."""
    mobile_mean = mobile.mean(axis=0)
    target_mean = target.mean(axis=0)
    mobile_c = mobile - mobile_mean
    target_c = target - target_mean
    h = mobile_c.T @ target_c
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = target_mean - r @ mobile_mean
    return (r @ mobile.T).T + t


def pocket_rmsd_from_atom_arrays(
    gen_arr: struc.AtomArray,
    true_arr: struc.AtomArray,
) -> float:
    """RMSD from biotite AtomArrays (matched heavy protein atoms)."""
    gen_heavy = gen_arr[(~gen_arr.hetero) & (gen_arr.element != "H")]
    true_heavy = true_arr[(~true_arr.hetero) & (true_arr.element != "H")]
    gen_map = {
        _atom_key(a.chain_id, a.res_id, a.atom_name): a.coord for a in gen_heavy
    }
    true_map = {
        _atom_key(a.chain_id, a.res_id, a.atom_name): a.coord for a in true_heavy
    }
    return _rmsd_from_coord_maps(true_map, gen_map)


def pocket_rmsd(gen_prot_file: str, true_prot_file: str) -> float:
    """RMSD over matched pocket heavy atoms after aligning gen onto holo."""
    true_map = _heavy_coords_by_key(true_prot_file)
    gen_map = _heavy_coords_by_key(gen_prot_file)
    return _rmsd_from_coord_maps(true_map, gen_map)
