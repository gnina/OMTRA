"""Masked, gated implicit-hydrogen direction features (S4): chemistry mask, geometry, batching, leakage, gates, bootstrap."""
from copy import deepcopy
import unittest
import dgl
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from omtra.data.graph.utils import build_lig_edge_idxs
from omtra.models.embeddings.implicit_h import h_site_metadata, h_directions
from omtra.tasks.register import task_name_to_class
from tests.test_model_integration import graph, predict
from tests.test_stereo_model import make_field


def lig_graph(mol_heavy, xyz_heavy, eligible, is_h):
    n = mol_heavy.GetNumAtoms()
    src, dst = build_lig_edge_idxs(n)
    orders = torch.tensor([0 if mol_heavy.GetBondBetweenAtoms(int(a), int(b)) is None else 1 for a, b in zip(src, dst)])
    g = dgl.heterograph({('lig', 'lig_to_lig', 'lig'): (src, dst)})
    g.edges['lig_to_lig'].data['e_t'] = orders
    g.nodes['lig'].data['h_eligible'] = torch.tensor(eligible)
    g.nodes['lig'].data['is_hydrogen'] = torch.tensor(is_h)
    return g


def embedded(smiles, with_h):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=3) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    xyz = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
    if with_h:
        return mol, xyz
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    return Chem.RemoveHs(mol), xyz[heavy]


class MaskTests(unittest.TestCase):
    def eligible(self, smiles):
        e, h = h_site_metadata(Chem.MolFromSmiles(smiles))
        return e, h

    def test_mask_follows_total_hydrogen_semantics(self):
        e, _ = self.eligible('C[C@H](O)C(=O)O')          # bracket [C@H]: GetNumImplicitHs()==0 but total H == 1
        self.assertEqual(e, [0., 1., 0., 0., 0., 0.])
        self.assertEqual(self.eligible('CC(C)(C)O')[0], [0.] * 5)          # four heavy neighbours: no missing H
        self.assertEqual(sum(self.eligible('c1ccccc1')[0]), 0)             # aromatic
        self.assertEqual(sum(self.eligible('CN(C)C')[0]), 0)               # amine N: no H
        self.assertEqual(self.eligible('C[NH+](C)C')[0][1], 1.)            # protonated amine N+-H with three heavy neighbours
        self.assertEqual(sum(self.eligible('CC(=O)C')[0]), 0)              # sp2 carbon
        self.assertEqual(sum(self.eligible('CC=CC')[0]), 0)

    def test_mask_is_independent_of_stereo_tags_and_explicit_h(self):
        self.assertEqual(self.eligible('C[C@H](O)C(=O)O')[0], self.eligible('C[C@@H](O)C(=O)O')[0])
        self.assertEqual(self.eligible('CC(O)C(=O)O')[0], self.eligible('C[C@H](O)C(=O)O')[0])
        mol, _ = embedded('CC(O)C(=O)O', True)
        e, h = h_site_metadata(mol)
        self.assertEqual(sum(e), 1.)                                       # same single site with explicit H atoms present
        self.assertEqual(sum(h), sum(a.GetAtomicNum() == 1 for a in mol.GetAtoms()))


class DirectionTests(unittest.TestCase):
    def test_virtual_and_actual_h_agree_with_the_real_hydrogen(self):
        for with_h in (False, True):
            mol, x = embedded('C[C@H](O)C(=O)O', with_h)
            e, h = h_site_metadata(mol)
            g = lig_graph(mol, x, e, h)
            out = h_directions(g, x)
            site = [i for i, v in enumerate(e) if v][0]
            full, xyz = embedded('C[C@H](O)C(=O)O', True)
            hyd = [n.GetIdx() for n in full.GetAtomWithIdx(site).GetNeighbors() if n.GetAtomicNum() == 1][0]
            cos = torch.nn.functional.cosine_similarity(out[site], xyz[hyd] - xyz[site], dim=0)
            self.assertGreater(float(cos), 0.9, with_h)
            self.assertLessEqual(float(out.norm(dim=-1).max()), 1.0 + 1e-5)
            self.assertTrue(bool((out[[i for i, v in enumerate(e) if not v]] == 0).all()))   # zero off-mask

    def test_equivariance_polar_vector_and_finiteness(self):
        mol, x = embedded('C[C@H](O)C(=O)O', False)
        e, h = h_site_metadata(mol); g = lig_graph(mol, x, e, h); out = h_directions(g, x)
        q, _ = torch.linalg.qr(torch.randn(3, 3))
        if torch.det(q) < 0:
            q[:, 0] *= -1
        torch.testing.assert_close(h_directions(g, x @ q.T + 4.), out @ q.T, rtol=1e-4, atol=1e-5)
        flip = torch.tensor([-1., 1., 1.])
        torch.testing.assert_close(h_directions(g, x * flip), out * flip, rtol=1e-4, atol=1e-5)
        with self.assertRaises(ValueError):
            h_directions(g, x * float('nan'))
        # degenerate (coincident neighbours): bounded and finite, never amplified
        z = h_directions(g, torch.zeros_like(x))
        self.assertTrue(torch.isfinite(z).all()); self.assertLessEqual(float(z.norm(dim=-1).max()), 1.0 + 1e-5)

    def test_batching_uses_global_node_indices_and_no_label_or_target_dependence(self):
        mol, x = embedded('C[C@H](O)C(=O)O', False); e, h = h_site_metadata(mol)
        mol2, x2 = embedded('CC(C)N', False); e2, h2 = h_site_metadata(mol2)
        g1, g2 = lig_graph(mol, x, e, h), lig_graph(mol2, x2, e2, h2)
        batched = dgl.batch([g1, g2])
        torch.testing.assert_close(h_directions(batched, torch.cat([x, x2])), torch.cat([h_directions(g1, x), h_directions(g2, x2)]), rtol=0, atol=1e-6)
        g1.nodes['lig'].data['x_1_true'] = torch.randn_like(x); g1.nodes['lig'].data['stereo_atom'] = torch.full((len(e),), 3)
        torch.testing.assert_close(h_directions(g1, x), h_directions(lig_graph(mol, x, e, h), x), rtol=0, atol=0)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.plain = make_field(); torch.manual_seed(5); self.hvec = make_field(implicit_h_vectors=True)
        self.hvec.load_state_dict(self.plain.state_dict(), strict=False)
        self.plain.eval(); self.hvec.eval()
        self.task = task_name_to_class('rigid_docking_condensed')
        self.g = graph(self.task, self.plain)
        self.g.nodes['lig'].data['h_eligible'] = torch.tensor([0., 1., 1., 1., 1., 0.]); self.g.nodes['lig'].data['is_hydrogen'] = torch.zeros(6)

    def test_gates_start_at_zero_and_receive_gradients(self):
        self.assertEqual(float(self.hvec.h_gate_current), 0.); self.assertEqual(float(self.hvec.h_gate_predicted), 0.)
        self.assertNotIn('h_gate_current', self.plain.state_dict())
        self.hvec.train(); self.hvec.zero_grad()
        g = deepcopy(self.g)
        # t = 0.5, forced self-conditioning bootstrap (training draws it with p=1/2): retry a few draws
        for seed in range(8):
            torch.manual_seed(seed)
            self.hvec.zero_grad()
            out = predict(self.hvec, deepcopy(g), self.task)
            sum(v.square().mean() for v in out.values()).backward()
            if self.hvec.h_gate_predicted.grad is not None and float(self.hvec.h_gate_predicted.grad.abs()) > 0:
                break
        self.assertTrue(torch.isfinite(self.hvec.h_gate_current.grad).all()); self.assertGreater(float(self.hvec.h_gate_current.grad.abs()), 0)
        self.assertGreater(float(self.hvec.h_gate_predicted.grad.abs()), 0)

    def test_predicted_feature_is_reachable_through_the_inference_bootstrap(self):
        from omtra.data.graph.utils import get_batch_idxs, get_upper_edge_mask
        def run(model):
            g = deepcopy(self.g)
            return model(g, self.task, torch.zeros(1), get_batch_idxs(g)[0], {'lig_to_lig': get_upper_edge_mask(g, 'lig_to_lig')})['lig_x']
        base = run(self.plain)
        torch.testing.assert_close(run(self.hvec), base, rtol=0, atol=0)               # both gates zero
        with torch.no_grad():
            self.hvec.h_gate_predicted.fill_(.7)                                       # only the predicted-endpoint channel
        self.assertGreater(float((run(self.hvec) - base).abs().max()), 1e-8)           # t == 0 bootstrap produced a prediction

    def test_feature_survives_hidden_labels(self):
        with torch.no_grad():
            self.hvec.h_gate_current.fill_(.5)
        a = predict(self.hvec, deepcopy(self.g), self.task)['lig_x']
        g = deepcopy(self.g); g.nodes['lig'].data['stereo_atom'] = torch.ones(6, dtype=torch.long)       # all labels 'unspecified'
        b = predict(self.hvec, g, self.task)['lig_x']
        # the labels change the output (stereo module) but H stays enabled: compare to H gate off under the same hidden labels
        with torch.no_grad():
            self.hvec.h_gate_current.zero_()
        c = predict(self.hvec, deepcopy(g), self.task)['lig_x']
        self.assertGreater(float((b - c).abs().max()), 1e-8)


class ReviewTests(unittest.TestCase):
    """Added after the independent S4 implementation review (2026-10-19)."""

    def setUp(self):
        torch.manual_seed(5)
        self.hvec = make_field(implicit_h_vectors=True).eval()
        with torch.no_grad():
            self.hvec.h_gate_current.fill_(.6); self.hvec.h_gate_predicted.fill_(.4)     # open gates: the feature is live
        self.task = task_name_to_class('rigid_docking_condensed')
        self.g = graph(self.task, self.hvec)
        nd = self.g.nodes['lig'].data
        nd['h_eligible'] = torch.tensor([0., 1., 1., 1., 1., 0.]); nd['is_hydrogen'] = torch.zeros(6)

    def test_open_gates_rotation_and_permutation_equivariance_with_valid_metadata(self):
        original = predict(self.hvec, deepcopy(self.g), self.task)['lig_x']
        order = torch.arange(5, -1, -1)
        permuted = dgl.node_subgraph(self.g, {nt: order if nt == 'lig' else torch.arange(self.g.num_nodes(nt)) for nt in self.g.ntypes})
        torch.testing.assert_close(predict(self.hvec, permuted, self.task)['lig_x'], original[order], rtol=2e-5, atol=2e-6)
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        rotated = deepcopy(self.g)
        for nt in rotated.ntypes:
            for key, value in list(rotated.nodes[nt].data.items()):
                if key.startswith('x_'):
                    rotated.nodes[nt].data[key] = value @ rotation + 7
                elif key.startswith('v_'):
                    rotated.nodes[nt].data[key] = value @ rotation
        torch.testing.assert_close(predict(self.hvec, rotated, self.task)['lig_x'], original @ rotation + 7, rtol=2e-5, atol=2e-6)
        # the open gates must actually matter in this configuration, otherwise the test would pass trivially
        with torch.no_grad():
            self.hvec.h_gate_current.zero_(); self.hvec.h_gate_predicted.zero_()
        self.assertGreater(float((predict(self.hvec, deepcopy(self.g), self.task)['lig_x'] - original).abs().max()), 1e-8)

    def test_missing_metadata_is_an_error_on_identity_fixed_tasks_only(self):
        g = deepcopy(self.g)
        del g.nodes['lig'].data['h_eligible']
        with self.assertRaises(ValueError):
            predict(self.hvec, g, self.task)
        generated = task_name_to_class('fixed_protein_ligand_denovo_condensed')
        gd = graph(generated, self.hvec)
        predict(self.hvec, gd, generated)          # no metadata and no error: the feature is absent on de novo tasks



if __name__ == '__main__':
    unittest.main()
