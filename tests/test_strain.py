"""Strain energy: the phosphorus hydrogen-placement artefact is removed, ordinary values are unchanged, results are reproducible, and strain does not need a protein."""
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from omtra.eval.metrics.strain import (STATUS_BAD_H, STATUS_OK, prepare_mol, strain_energies, strain_energy_checked)

DATA = Path(__file__).parent / "data"


def _load(name):
    return next(iter(Chem.SDMolSupplier(str(DATA / name), removeHs=False)))


def _longest_h_bond(mol):
    mh = Chem.AddHs(mol, addCoords=True)
    conf = mh.GetConformer()
    best = 0.0
    for b in mh.GetBonds():
        if b.GetBeginAtom().GetAtomicNum() == 1 or b.GetEndAtom().GetAtomicNum() == 1:
            best = max(best, conf.GetAtomPosition(b.GetBeginAtomIdx()).Distance(conf.GetAtomPosition(b.GetEndAtomIdx())))
    return best


def test_artefact_pose_old_hydrogen_placement_is_degenerate_and_prepared_is_not():
    mol = _load("strain_p_artefact.sdf")
    # the raw molecule gets an implicit H on a four-coordinate phosphorus; its placement is absurd (observed 96 A)
    assert _longest_h_bond(mol) > 5.0
    prepared = prepare_mol(mol)
    assert _longest_h_bond(prepared) < 1.8


def test_artefact_pose_gives_ordinary_strain():
    r = strain_energy_checked(_load("strain_p_artefact.sdf"), num_confs=10)
    assert r["status"] == STATUS_OK
    # previously 2.13e6 kcal/mol; the recomputed ordinary value was ~108 (embeddings are now seeded)
    assert 0.0 < r["strain"] < 500.0


def test_ordinary_pose_unchanged_against_original_value():
    r = strain_energy_checked(_load("strain_normal.sdf"), num_confs=50)
    assert r["status"] == STATUS_OK
    assert abs(r["strain"] - 52.0006) < 15.0   # original unseeded value 52.0; unseeded noise is ~10 kcal/mol


def test_seeded_embeddings_are_reproducible():
    mol = _load("strain_normal.sdf")
    a = strain_energy_checked(mol, num_confs=8, seed=1)["strain"]
    b = strain_energy_checked(mol, num_confs=8, seed=1)["strain"]
    assert a == b


def test_failure_is_nan_not_a_number():
    assert np.isnan(strain_energy_checked(None)["strain"])
    assert np.isnan(strain_energies([None])[0])


def test_posecheck_all_keeps_strain_when_protein_step_fails(tmp_path, monkeypatch):
    from omtra.eval.metrics import posecheck as pcmod

    class Boom:
        def load_protein_from_pdb(self, *a, **k):
            raise ValueError("vdw radii for types: MN")

    monkeypatch.setattr(pcmod, "PoseCheck", Boom)
    monkeypatch.setattr(pcmod, "strain_energies", lambda ligs, **k: [12.5] * len(ligs))
    out = pcmod.posecheck_all([_load("strain_normal.sdf")], str(tmp_path / "missing.pdb"))
    assert out["strain"] == [12.5]
    assert np.isnan(out["clashes"][0]) and np.isnan(out["HBDonor"][0])
