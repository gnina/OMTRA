import unittest
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from omtra.aux_losses.stereo_geometry import (signed_volume3, signed_volume4, chi_for_units, tetrahedral_units,
                                              reference_signs)


def _embed(smiles, seed=7, hs=True):
    mol = Chem.MolFromSmiles(smiles)
    if hs:
        mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


class StereoGeometry(unittest.TestCase):
    def test_regular_tetrahedron_magnitudes(self):
        pts = torch.tensor([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]], dtype=torch.float64)
        centre = torch.zeros(1, 3, dtype=torch.float64)
        x = torch.cat([centre, pts])
        v3 = signed_volume3(x, torch.tensor([0]), torch.tensor([[1, 2, 3]]))
        self.assertAlmostEqual(abs(float(v3)), 0.7698, places=3)
        v4 = signed_volume4(x, torch.tensor([[1, 2, 3, 4]]))
        self.assertAlmostEqual(abs(float(v4)), 0.7071, places=3)

    def test_mirror_flips_sign_and_scale_invariance(self):
        torch.manual_seed(0)
        x = torch.randn(5, 3, dtype=torch.float64)
        n4 = torch.tensor([[1, 2, 3, 4]])
        mirrored = x * torch.tensor([-1., 1., 1.], dtype=torch.float64)
        self.assertAlmostEqual(float(signed_volume4(x, n4)), -float(signed_volume4(mirrored, n4)), places=9)
        self.assertAlmostEqual(float(signed_volume4(x, n4)), float(signed_volume4(3.0 * x, n4)), places=4)
        c, n3 = torch.tensor([0]), torch.tensor([[1, 2, 3]])
        self.assertAlmostEqual(float(signed_volume3(x, c, n3)), -float(signed_volume3(mirrored, c, n3)), places=9)

    def test_enantiomers_have_opposite_signs_on_real_molecules(self):
        for smi, hs in (('C[C@H](N)C(=O)O', True), ('C[C@H](N)C(=O)O', False), ('C[C@H](F)[C@@H](O)Cl', True)):
            a = _embed(smi, hs=hs)
            b = _embed(smi.replace('@@', '#').replace('@', '@@').replace('#', '@'), hs=hs)
            ca, na = tetrahedral_units(a)
            cb, nb = tetrahedral_units(b)
            self.assertEqual(ca, cb)
            sa, _, ua = reference_signs(a.GetConformer().GetPositions(), ca, na)
            sb, _, ub = reference_signs(b.GetConformer().GetPositions(), cb, nb)
            self.assertTrue(ua.all() and ub.all())
            self.assertTrue((sa == -sb).all(), (smi, sa, sb))
            # exact reflection of the same conformer flips all signs too
            xa = a.GetConformer().GetPositions()
            sm, _, _ = reference_signs(xa * np.array([1, 1, -1.0]), ca, na)
            self.assertTrue((sm == -sa).all())

    def test_batched_time_axis(self):
        mol = _embed('C[C@H](N)C(=O)O')
        c, n = tetrahedral_units(mol)
        x = torch.as_tensor(mol.GetConformer().GetPositions(), dtype=torch.float64)
        series = torch.stack([x, x * torch.tensor([1., 1., -1.], dtype=torch.float64)])   # (2, N, 3)
        chi = chi_for_units(series, c, n)
        self.assertEqual(tuple(chi.shape), (2, len(c)))
        self.assertTrue(torch.allclose(chi[0], -chi[1]))

    def test_implicit_h_three_neighbour_unit(self):
        mol = _embed('C[C@H](N)C(=O)O', hs=False)
        c, n = tetrahedral_units(mol)
        self.assertEqual([len(v) for v in n], [3])


if __name__ == '__main__':
    unittest.main()
