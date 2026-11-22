"""Margin handedness loss and geometric stereo-correction vector features (default-off mechanisms)."""
from copy import deepcopy
import unittest
import dgl
import torch

from omtra.aux_losses.chirality_margin import ChiralityMarginLoss
from omtra.aux_losses.stereo_geometry import (graph_units, chi_tensor, margin_unit_loss, unit_signs_from_coords)
from omtra.models.embeddings.stereo_correction import StereoCorrection
from omtra.tasks.register import task_name_to_class
from tests.test_model_integration import graph, predict
from tests.test_stereo_model import make_field

CENTRE = 1
NBRS = [0, 2, 3, 4]


def branched_graph(task, vf, seed=0, shown=True):
    """The 6-atom test graph rewired to a degree-4 centre (atom 1; neighbours 0,2,3,4) with crystal handedness attached."""
    g = graph(task, vf)
    torch.manual_seed(seed)
    src, dst = g.edges(etype='lig_to_lig')
    bonded = (((src == CENTRE) & torch.isin(dst, torch.tensor(NBRS))) | ((dst == CENTRE) & torch.isin(src, torch.tensor(NBRS)))).long()
    g.edges['lig_to_lig'].data['e_1_true'] = bonded.clone()
    g.edges['lig_to_lig'].data['e_t'] = bonded.clone()
    nd = g.nodes['lig'].data
    tet = torch.tensor([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]]) * 1.5
    x = torch.zeros(6, 3)
    x[CENTRE] = 0.
    for atom, p in zip(NBRS, tet):
        x[atom] = p
    x[5] = torch.tensor([6., 6., 6.])
    nd['x_1_true'] = x.clone()
    nd['x_t'] = (x + 0.3 * torch.randn(6, 3)).clone()
    nd['stereo_centre'] = torch.zeros(6, dtype=torch.long)
    nd['stereo_centre'][CENTRE] = 1
    nd['stereo_atom'] = torch.zeros(6, dtype=torch.long)
    nd['stereo_atom'][CENTRE] = 2 if shown else 1
    z, centres, _ = unit_signs_from_coords(g, x)
    nd['stereo_unit_z'] = torch.zeros(6)
    nd['stereo_unit_z'][centres] = z.float()
    return g


class MarginLossTests(unittest.TestCase):
    def test_hinge_values(self):
        chi = torch.tensor([0.5, 0.2, 0.1, 0.0, -0.3])
        z = torch.ones(5)
        loss = margin_unit_loss(chi, z, 0.2)
        torch.testing.assert_close(loss, torch.tensor([0., 0., 0.25, 1., 6.25]))      # ((0.2-chi)/0.2)^2 for chi < 0.2
        self.assertEqual(float(margin_unit_loss(chi, torch.zeros(5), 0.2).sum()), 0.)  # hidden / planar units contribute nothing

    def test_loss_is_zero_for_correct_large_volume_and_positive_for_mirror(self):
        task = task_name_to_class('ligand_conformer_condensed')
        g = branched_graph(task, make_field())
        loss_fn = ChiralityMarginLoss(weight=1.0)
        from omtra.data.graph.utils import get_batch_idxs
        node_idx = get_batch_idxs(g)[0]
        x = g.nodes['lig'].data['x_1_true']
        right = loss_fn(g, {'lig_x': x.clone()}, task, node_idx, None, None)
        mirror = loss_fn(g, {'lig_x': x * torch.tensor([-1., 1., 1.])}, task, node_idx, None, None)
        self.assertAlmostEqual(float(right), 0.0, places=6)
        self.assertGreater(float(mirror), 1.0)

    def test_per_molecule_normalisation_and_gradients(self):
        task = task_name_to_class('ligand_conformer_condensed')
        a = branched_graph(task, make_field(), seed=1); b = branched_graph(task, make_field(), seed=2)
        batched = dgl.batch([a, b])
        from omtra.data.graph.utils import get_batch_idxs
        idx = get_batch_idxs(batched)[0]
        x = batched.nodes['lig'].data['x_1_true'].clone()
        pred = x.clone(); pred[6:] = pred[6:] * torch.tensor([-1., 1., 1.])           # second molecule mirrored, first correct
        pred.requires_grad_(True)
        loss = ChiralityMarginLoss(weight=1.0)(batched, {'lig_x': pred}, task, idx, None, None)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all()); self.assertGreater(float(pred.grad.abs().sum()), 0)
        alone = ChiralityMarginLoss(weight=1.0)(b, {'lig_x': x[6:] * torch.tensor([-1., 1., 1.])}, task, get_batch_idxs(b)[0], None, None)
        self.assertAlmostEqual(float(loss), 0.5 * float(alone), places=5)             # mean over the two molecules (one has loss 0)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.mod = StereoCorrection(8, 4)
        task = task_name_to_class('ligand_conformer_condensed')
        self.g = branched_graph(task, make_field())
        self.ctx = StereoCorrection.units(self.g)

    def test_force_decreases_the_energy_and_is_bounded(self):
        x = self.g.nodes['lig'].data['x_1_true'] * torch.tensor([-1., 1., 1.]) + 0.05 * torch.randn(6, 3)   # mirrored: wrong side
        f = self.mod.feature(x, self.ctx)
        self.assertLessEqual(float(f.norm(dim=-1).max()), 1.0 + 1e-5)
        energy = lambda y: margin_unit_loss(chi_tensor(y, self.ctx['centres'], self.ctx['nbrs']), self.ctx['z'], 0.2).sum()
        self.assertLess(float(energy(x + 1e-3 * f)), float(energy(x)))

    def test_rotation_and_reflection_equivariance(self):
        x = self.g.nodes['lig'].data['x_1_true'] * torch.tensor([-1., 1., 1.]) + 0.1 * torch.randn(6, 3)
        f = self.mod.feature(x, self.ctx)
        q, _ = torch.linalg.qr(torch.randn(3, 3))
        if torch.det(q) < 0:
            q[:, 0] *= -1
        torch.testing.assert_close(self.mod.feature(x @ q.T + 3., self.ctx), f @ q.T, rtol=1e-3, atol=1e-4)
        flip = torch.tensor([-1., 1., 1.])
        mirrored_ctx = dict(self.ctx, z=-self.ctx['z'])
        torch.testing.assert_close(self.mod.feature(x * flip, mirrored_ctx), f * flip, rtol=1e-3, atol=1e-4)

    def test_exactly_mirrored_ideal_tetrahedron_is_a_stationary_point(self):
        """Documented property of the plan's four-substituent chi (volume / mean-edge^1.5): |chi| is maximal at the regular
        tetrahedron, so the hinge force vanishes at an exactly inverted ideal centre and grows with distortion."""
        mirrored = self.g.nodes['lig'].data['x_1_true'] * torch.tensor([-1., 1., 1.])
        at_ideal = self.mod.feature(mirrored, self.ctx)
        distorted = self.mod.feature(mirrored + 0.15 * torch.randn(6, 3, generator=torch.Generator().manual_seed(4)), self.ctx)
        self.assertLess(float(at_ideal.norm(dim=-1).max()), 1e-4)
        self.assertGreater(float(distorted.norm(dim=-1).max()), 1e-2)

    def test_hidden_labels_or_no_units_give_no_context(self):
        task = task_name_to_class('ligand_conformer_condensed')
        hidden = branched_graph(task, make_field(), shown=False)
        self.assertIsNone(StereoCorrection.units(hidden))
        zero = self.mod.feature(torch.randn(6, 3), None)
        self.assertEqual(float(zero.abs().sum()), 0.)

    def test_missing_unit_signs_raise(self):
        task = task_name_to_class('ligand_conformer_condensed')
        g = branched_graph(task, make_field())
        del g.nodes['lig'].data['stereo_unit_z']
        with self.assertRaises(ValueError):
            StereoCorrection.units(g)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.plain = make_field()
        torch.manual_seed(5)
        self.corr = make_field(stereo_correction=True)
        self.corr.load_state_dict(self.plain.state_dict(), strict=False)
        self.plain.eval(); self.corr.eval()
        self.task = task_name_to_class('rigid_docking_condensed')
        self.g = branched_graph(self.task, self.plain)
        # wrong-handed current coordinates: with correct, well-formed handedness the hinge force is exactly zero
        self.g.nodes['lig'].data['x_t'] = self.g.nodes['lig'].data['x_1_true'] * torch.tensor([-1., 1., 1.]) + 0.05 * torch.randn(6, 3)

    def test_state_dict_only_gains_the_two_zero_tables_and_function_is_preserved(self):
        extra = set(self.corr.state_dict()) - set(self.plain.state_dict())
        self.assertEqual(extra, {'stereo_correction.w_current', 'stereo_correction.w_pred'})
        for k in extra:
            self.assertEqual(float(self.corr.state_dict()[k].abs().sum()), 0.)
        torch.manual_seed(1); a = predict(self.plain, deepcopy(self.g), self.task)
        torch.manual_seed(1); b = predict(self.corr, deepcopy(self.g), self.task)
        for k in a:
            torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)

    def _with_mirrored_prev(self, model, g):
        """Forward with an explicit previous endpoint prediction that is wrong-handed (exercises the predicted-endpoint channel)."""
        from omtra.data.graph.utils import get_batch_idxs, get_upper_edge_mask
        idx, ue = get_batch_idxs(g)[0], {'lig_to_lig': get_upper_edge_mask(g, 'lig_to_lig')}
        with torch.no_grad():
            prev = {k: v.detach().clone() for k, v in model(deepcopy(g), self.task, torch.full((1,), .5), idx, ue).items()}
        # noisy, wrong-handed endpoint (an exactly mirrored regular tetrahedron is a stationary point of the hinge energy, see FeatureTests)
        prev['lig_x'] = g.nodes['lig'].data['x_1_true'] * torch.tensor([-1., 1., 1.]) + 0.15 * torch.randn(6, 3, generator=torch.Generator().manual_seed(4))
        return model(g, self.task, torch.full((1,), .5), idx, ue, prev_dst_dict=prev)

    def test_both_tables_get_nonzero_gradients_immediately(self):
        self.corr.train()
        self.corr.zero_grad()
        out = self._with_mirrored_prev(self.corr, deepcopy(self.g))
        sum(v.square().mean() for v in out.values()).backward()
        for name in ('w_current', 'w_pred'):
            p = getattr(self.corr.stereo_correction, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(float(p.grad.abs().sum()), 0, name)

    def test_open_tables_change_outputs_only_when_a_label_is_visible_and_stay_equivariant(self):
        with torch.no_grad():
            self.corr.stereo_correction.w_current.normal_(0, 0.3); self.corr.stereo_correction.w_pred.normal_(0, 0.3)
        torch.manual_seed(3); base = predict(self.plain, deepcopy(self.g), self.task)['lig_x']
        torch.manual_seed(3); shown = predict(self.corr, deepcopy(self.g), self.task)['lig_x']
        pred_only = deepcopy(self.corr)
        with torch.no_grad():
            pred_only.stereo_correction.w_current.zero_()
        base_prev = self._with_mirrored_prev(self.plain, deepcopy(self.g))['lig_x']
        self.assertGreater(float((self._with_mirrored_prev(pred_only.eval(), deepcopy(self.g))['lig_x'] - base_prev).abs().max()), 1e-6)   # predicted-endpoint channel alone matters
        self.assertGreater(float((shown - base).abs().max()), 1e-6)
        hidden = deepcopy(self.g); hidden.nodes['lig'].data['stereo_atom'][CENTRE] = 1
        torch.manual_seed(3); hidden_plain = predict(self.plain, deepcopy(hidden), self.task)['lig_x']
        torch.manual_seed(3); hidden_corr = predict(self.corr, deepcopy(hidden), self.task)['lig_x']
        torch.testing.assert_close(hidden_plain, hidden_corr, rtol=0, atol=0)          # hidden label: the feature is absent
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        rotated = deepcopy(self.g)
        for nt in rotated.ntypes:
            for key, value in list(rotated.nodes[nt].data.items()):
                if key.startswith('x_'):
                    rotated.nodes[nt].data[key] = value @ rotation + 7
                elif key.startswith('v_'):
                    rotated.nodes[nt].data[key] = value @ rotation
        torch.manual_seed(3); out_rot = predict(self.corr, rotated, self.task)['lig_x']
        torch.testing.assert_close(out_rot, shown @ rotation + 7, rtol=1e-3, atol=1e-4)

    def test_missing_unit_signs_on_identity_fixed_task_raise(self):
        g = deepcopy(self.g)
        del g.nodes['lig'].data['stereo_unit_z']
        with self.assertRaises(ValueError):
            predict(self.corr, g, self.task)


if __name__ == '__main__':
    unittest.main()
