"""CPU tests for the relational (RitS-style) stereo labels."""
from types import SimpleNamespace
import unittest
import numpy as np
import torch
import dgl
from rdkit import Chem
from rdkit.Chem import AllChem

from omtra.constants import lig_atom_type_map, charge_map
from omtra.data.graph.utils import build_lig_edge_idxs
from omtra.data.requested_stereo import attach_identity, attach_geometry, hide_stereo, labels_from_identity
from omtra.data.stereo_relational import relational_structure, canonical_ranks, N_CHANNELS

FIXED = SimpleNamespace(groups_fixed=['ligand_identity_condensed'])
GENERATED = SimpleNamespace(groups_fixed=[])


def graph_from_mol(mol):
    src, dst = build_lig_edge_idxs(mol.GetNumAtoms())
    g = dgl.heterograph({('lig', 'lig_to_lig', 'lig'): (src, dst)})
    nd = g.nodes['lig'].data
    nd['a_1_true'] = torch.tensor([lig_atom_type_map.index(a.GetSymbol()) for a in mol.GetAtoms()])
    nd['c_1_true'] = torch.tensor([charge_map.index(a.GetFormalCharge()) for a in mol.GetAtoms()])
    nd['x_1_true'] = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
    orders = [0 if mol.GetBondBetweenAtoms(int(a), int(b)) is None
              else int(mol.GetBondBetweenAtoms(int(a), int(b)).GetBondTypeAsDouble()) for a, b in zip(src, dst)]
    g.edges['lig_to_lig'].data['e_1_true'] = torch.tensor(orders)
    g.edges['lig_to_lig'].data['e_t'] = torch.tensor(orders)
    return g


def embedded(smiles, seed=7):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    return Chem.RemoveHs(mol)


def rel(g):
    return g.edges['lig_to_lig'].data['stereo_rel']


class RelationalLabelTests(unittest.TestCase):
    SMILES = ['C[C@H](O)C(=O)O', 'C[C@@H](O)C(=O)O', 'C[C@](F)(Cl)Br', 'F/C=C/F', 'F/C=C\\F',
              'C[C@H](O)[C@@H](O)[C@H](O)C', 'C[C@H]1CCCC[C@@H]1O', 'F/C=C/[C@H](O)C']

    def test_training_geometry_and_supplied_identity_agree(self):
        for smiles in self.SMILES:
            crystal = embedded(smiles)
            g_train = graph_from_mol(crystal)
            attach_geometry(g_train, FIXED)
            g_user = graph_from_mol(crystal)
            attach_identity(g_user, FIXED, Chem.MolFromSmiles(smiles), source='supplied')
            torch.testing.assert_close(rel(g_train), rel(g_user), msg=smiles)
            self.assertGreater(float(rel(g_train).sum()), 0, smiles)

    def test_enantiomers_swap_direction_channels(self):
        a = graph_from_mol(embedded('C[C@H](O)C(=O)O'))
        b = graph_from_mol(embedded('C[C@@H](O)C(=O)O'))
        attach_geometry(a, FIXED)
        attach_geometry(b, FIXED)
        torch.testing.assert_close(rel(a)[:, 0], rel(b)[:, 1])
        torch.testing.assert_close(rel(a)[:, 1], rel(b)[:, 0])
        self.assertFalse(torch.equal(rel(a), rel(b)))

    def test_double_bond_cis_trans_use_distinct_channels(self):
        out = {}
        for name, smiles in (('trans', 'F/C=C/F'), ('cis', 'F/C=C\\F')):
            g = graph_from_mol(embedded(smiles))
            attach_identity(g, FIXED, Chem.MolFromSmiles(smiles), source='supplied')
            out[name] = rel(g)
        self.assertGreater(float(out['trans'][:, 4].sum()), 0)
        self.assertEqual(float(out['trans'][:, 3].sum()), 0)
        self.assertGreater(float(out['cis'][:, 3].sum()), 0)
        self.assertEqual(float(out['cis'][:, 4].sum()), 0)

    def test_atom_permutation_invariance(self):
        smiles = 'C[C@H](O)[C@@H](O)[C@H](O)C'
        crystal = embedded(smiles)
        order = list(reversed(range(crystal.GetNumAtoms())))
        permuted = Chem.RenumberAtoms(crystal, order)
        g0, g1 = graph_from_mol(crystal), graph_from_mol(permuted)
        attach_geometry(g0, FIXED)
        attach_geometry(g1, FIXED)
        new_index = {old: order.index(old) for old in range(crystal.GetNumAtoms())}
        src, dst = g0.edges(etype='lig_to_lig')
        mapped = g1.edge_ids(torch.tensor([new_index[int(u)] for u in src]),
                             torch.tensor([new_index[int(v)] for v in dst]), etype='lig_to_lig')
        torch.testing.assert_close(rel(g0), rel(g1)[mapped])

    def test_no_pose_leakage_on_supplied_path(self):
        smiles = 'C[C@H]1CCCC[C@@H]1O'
        crystal = embedded(smiles)
        g1 = graph_from_mol(crystal)
        attach_identity(g1, FIXED, Chem.MolFromSmiles(smiles), source='supplied')
        g2 = graph_from_mol(crystal)
        g2.nodes['lig'].data['x_1_true'].fill_(float('nan'))
        attach_identity(g2, FIXED, Chem.MolFromSmiles(smiles), source='supplied')
        torch.testing.assert_close(rel(g1), rel(g2))

    def test_hiding_removes_labels_consistently(self):
        g = graph_from_mol(embedded('C[C@H](O)[C@@H](O)[C@H](O)C'))
        attach_geometry(g, FIXED)
        full = rel(g).clone()
        self.assertGreater(float(full.sum()), 0)
        gen = torch.Generator().manual_seed(1)
        self.assertEqual(hide_stereo(g, p_all=1.0, p_partial=0.0, generator=gen), 2)
        self.assertEqual(float(rel(g).sum()), 0)
        # partial hiding leaves a strict subset of the full labels
        seen_partial = False
        for _ in range(40):
            g2 = graph_from_mol(embedded('C[C@H](O)[C@@H](O)[C@H](O)C'))
            attach_geometry(g2, FIXED)
            if hide_stereo(g2, p_all=0.0, p_partial=1.0, partial_rate=0.5, generator=gen) == 1:
                self.assertTrue(bool((rel(g2) <= full + 1e-6).all()))
                seen_partial = seen_partial or float(rel(g2).sum()) < float(full.sum())
        self.assertTrue(seen_partial)

    def test_generative_and_achiral_are_zero(self):
        g = graph_from_mol(embedded('C[C@H](O)C(=O)O'))
        attach_identity(g, GENERATED, Chem.MolFromSmiles('C[C@H](O)C(=O)O'))
        self.assertEqual(float(rel(g).sum()), 0)
        g = graph_from_mol(embedded('CCO'))
        attach_geometry(g, FIXED)
        self.assertEqual(float(rel(g).sum()), 0)
        self.assertEqual(tuple(rel(g).shape[1:]), (N_CHANNELS,))

    def test_symmetric_ring_ties_are_skipped_not_guessed(self):
        mol = embedded('C[C@H]1CC[C@@H](C)CC1')
        atoms, bonds = labels_from_identity(Chem.MolFromSmiles('C[C@H]1CC[C@@H](C)CC1'))
        centres, _, skipped = relational_structure(mol, atoms, bonds, mol.GetConformer().GetPositions())
        self.assertEqual(centres, {})
        self.assertGreater(skipped['tie'], 0)

    def test_canonical_ranks_do_not_depend_on_stereo_tags(self):
        a = canonical_ranks(Chem.MolFromSmiles('C[C@H](O)C(=O)O'))
        b = canonical_ranks(Chem.MolFromSmiles('C[C@@H](O)C(=O)O'))
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
