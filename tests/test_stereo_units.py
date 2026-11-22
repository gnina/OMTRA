"""Unit-sign adapters for the optional correction features: training geometry, supplied identity (ETKDG conformer + independent tag check),
atom-permutation parity, undefined stereo staying undefined, hidden labels, pre-preprocessing pose perturbations."""
from types import SimpleNamespace
from unittest import mock
import unittest
import numpy as np
import torch
from rdkit import Chem

from omtra.aux_losses.stereo_geometry import chi_tensor, graph_units, margin_unit_loss
from omtra.data.requested_stereo import attach_geometry, attach_identity, hide_stereo
from omtra.data.stereo_units import attach_unit_signs
from omtra.models.embeddings.stereo_correction import StereoCorrection
from tests.test_stereo_relational import graph_from_mol, embedded

FIXED = SimpleNamespace(groups_fixed=['ligand_identity_condensed'])
SMILES = ['C[C@H](O)C(=O)O', 'C[C@@H](O)C(=O)O', 'C[C@](F)(Cl)Br', 'C[C@H](O)[C@@H](O)[C@H](O)C', 'C[C@H]1CCCC[C@@H]1O', 'F/C=C/[C@H](O)C']


def z_of(g):
    return g.nodes['lig'].data['stereo_unit_z'].clone()


class TrainingAndSuppliedAgree(unittest.TestCase):
    def test_supplied_conformer_gives_the_crystal_signs(self):
        for smiles in SMILES:
            crystal = embedded(smiles)
            g_train = graph_from_mol(crystal)
            attach_geometry(g_train, FIXED)
            g_sup = graph_from_mol(crystal)
            attach_identity(g_sup, FIXED, Chem.MolFromSmiles(smiles), 'supplied')
            zt, zs = z_of(g_train), z_of(g_sup)
            self.assertGreater(int((zt != 0).sum()), 0, smiles)
            both = (zt != 0) & (zs != 0)
            self.assertTrue(bool((zt[both] == zs[both]).all()), (smiles, zt, zs))
            self.assertEqual(int(both.sum()), int((zt != 0).sum()), smiles)      # every crystal-signed unit is also signed from the supplied molecule
            self.assertEqual(g_sup.stereo_unit_info['tag_mismatch'], 0, smiles)

    def test_enantiomers_get_opposite_signs_and_mirror_geometry_flips_training_signs(self):
        a, b = SMILES[0], SMILES[1]
        ga, gb = graph_from_mol(embedded(a)), graph_from_mol(embedded(a))
        attach_identity(ga, FIXED, Chem.MolFromSmiles(a), 'supplied')
        attach_identity(gb, FIXED, Chem.MolFromSmiles(b), 'supplied')                 # same graph, opposite requested configuration
        self.assertTrue(bool((z_of(ga) == -z_of(gb)).all()))
        self.assertGreater(float(z_of(ga).abs().sum()), 0)
        gt = graph_from_mol(embedded(a)); gm = graph_from_mol(embedded(a))
        gm.nodes['lig'].data['x_1_true'] = gm.nodes['lig'].data['x_1_true'] * torch.tensor([-1., 1., 1.])
        attach_geometry(gt, FIXED); attach_geometry(gm, FIXED)
        self.assertTrue(bool((z_of(gt) == -z_of(gm)).all()))


class Robustness(unittest.TestCase):
    def test_conformer_seed_does_not_change_mapped_signs(self):
        for smiles in SMILES:
            mol = Chem.MolFromSmiles(smiles)
            g = graph_from_mol(embedded(smiles))
            attach_identity(g, FIXED, mol, 'supplied')
            ref = z_of(g)
            for seed in (1, 2, 3):
                attach_unit_signs(g, mol, 'supplied', conformer_seed=seed)
                z = z_of(g)
                both = (ref != 0) & (z != 0)
                self.assertTrue(bool((ref[both] == z[both]).all()), (smiles, seed))
                self.assertEqual(int(both.sum()), int((ref != 0).sum()), (smiles, seed))

    def test_atom_permutation_keeps_the_requested_configuration(self):
        rng = np.random.default_rng(3)
        for smiles in SMILES:
            crystal = embedded(smiles)
            perm = [int(i) for i in rng.permutation(crystal.GetNumAtoms())]
            crystal_p = Chem.RenumberAtoms(crystal, perm)
            ident_p = Chem.RenumberAtoms(Chem.MolFromSmiles(smiles), perm)
            g = graph_from_mol(crystal_p)
            attach_identity(g, FIXED, ident_p, 'supplied')
            centres, nbrs = graph_units(g)
            z = z_of(g)[centres]
            chi = chi_tensor(g.nodes['lig'].data['x_1_true'].double(), centres, nbrs)
            # the crystal realises the supplied configuration: the hinge energy of the signs read from the permuted supplied molecule is zero
            self.assertEqual(float(margin_unit_loss(chi, z, 0.2).sum()), 0.0, smiles)
            self.assertGreater(int((z != 0).sum()), 0)

    def test_undefined_stereo_stays_undefined(self):
        smiles = 'CC(O)C(=O)O'
        g = graph_from_mol(embedded(smiles))
        attach_identity(g, FIXED, Chem.MolFromSmiles(smiles), 'supplied')
        self.assertEqual(float(z_of(g).abs().sum()), 0.0)
        self.assertEqual(g.stereo_unit_info['units'], 0)

    def test_wrong_handed_conformer_is_rejected_not_signed(self):
        smiles = SMILES[0]
        mol = Chem.MolFromSmiles(smiles)
        g = graph_from_mol(embedded(smiles))
        real = embedded(smiles).GetConformer().GetPositions()
        with mock.patch('omtra.data.stereo_units.supplied_conformer', return_value=real * np.array([-1., 1., 1.])):
            attach_identity(g, FIXED, mol, 'supplied')
        self.assertEqual(float(z_of(g).abs().sum()), 0.0)
        self.assertGreater(g.stereo_unit_info['tag_mismatch'], 0)

    def test_reference_pose_perturbation_before_preprocessing_changes_nothing_in_supplied_mode(self):
        smiles = SMILES[3]
        mol = Chem.MolFromSmiles(smiles)
        g1 = graph_from_mol(embedded(smiles)); g2 = graph_from_mol(embedded(smiles))
        g2.nodes['lig'].data['x_1_true'] = torch.randn_like(g2.nodes['lig'].data['x_1_true']) * 3
        attach_identity(g1, FIXED, mol, 'supplied'); attach_identity(g2, FIXED, mol, 'supplied')
        torch.testing.assert_close(z_of(g1), z_of(g2), rtol=0, atol=0)

    def test_hidden_labels_remove_every_correction_signal(self):
        smiles = SMILES[3]
        g = graph_from_mol(embedded(smiles))
        attach_identity(g, FIXED, Chem.MolFromSmiles(smiles), 'supplied')
        self.assertIsNotNone(StereoCorrection.units(g))
        self.assertEqual(hide_stereo(g, p_all=1.0), 2)
        self.assertIsNone(StereoCorrection.units(g))                               # signs are still stored, but no visible label => no force
        gu = graph_from_mol(embedded(smiles))
        attach_identity(gu, FIXED, Chem.MolFromSmiles('CC(O)C(O)C(O)C'), 'supplied')   # unknown mode: chemistry only
        self.assertIsNone(StereoCorrection.units(gu))

    def test_explicit_hydrogen_graph_and_four_heavy_neighbour_centre(self):
        smiles = 'C[C@H](O)C(=O)O'
        crystal_h = Chem.AddHs(Chem.MolFromSmiles(smiles))
        from rdkit.Chem import AllChem
        self.assertEqual(AllChem.EmbedMolecule(crystal_h, randomSeed=7), 0)
        g_h = graph_from_mol(crystal_h)                      # every hydrogen is a graph node
        attach_geometry(g_h, FIXED)
        g_s = graph_from_mol(crystal_h)
        attach_identity(g_s, FIXED, Chem.AddHs(Chem.MolFromSmiles(smiles)), 'supplied')
        zt, zs = z_of(g_h), z_of(g_s)
        self.assertGreater(int((zt != 0).sum()), 0)
        self.assertTrue(bool((zt == zs).all()))
        quat = 'C[C@](F)(Cl)Br'
        g4 = graph_from_mol(embedded(quat))
        attach_identity(g4, FIXED, Chem.MolFromSmiles(quat), 'supplied')
        _, nbrs = graph_units(g4)
        self.assertEqual(int((nbrs[:, 3] >= 0).sum()), 1)
        self.assertEqual(int((z_of(g4) != 0).sum()), 1)


if __name__ == '__main__':
    unittest.main()
