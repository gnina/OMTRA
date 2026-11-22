"""Classifier-free stereo guidance (evaluation diagnostic): equivalences, state handling and tensor restoration."""
from copy import deepcopy
import unittest
import torch

from omtra.data.graph.utils import get_batch_idxs, get_upper_edge_mask
from omtra.data.inference_stereo import attach_guidance_alternative
from omtra.tasks.register import task_name_to_class
from tests.test_model_integration import graph
from tests.test_stereo_model import make_field

TASK = 'rigid_docking_condensed'


def prepared(vf, task):
    g = graph(task, vf)                       # stereo_atom[1] = 2 (R) is the requested label
    nd, ed = g.nodes['lig'].data, g.edges['lig_to_lig'].data
    nd['stereo_atom_unk'] = torch.zeros_like(nd['stereo_atom'])
    nd['stereo_atom_unk'][1] = 1              # same centre, hidden ("unspecified")
    ed['stereo_bond_unk'] = torch.zeros_like(ed['stereo_bond'])
    ed['stereo_rel_unk'] = torch.zeros_like(ed['stereo_rel'])
    return g


def run(vf, g, task, w, steps=4):
    vf.stereo_guidance_w = w
    g = deepcopy(g)
    upper = {'lig_to_lig': get_upper_edge_mask(g, 'lig_to_lig')}
    with torch.no_grad():
        out = vf.integrate(g, task, upper, n_timesteps=steps, stochasticity=0.0)
    vf.stereo_guidance_w = None
    return out


class GuidanceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.vf = make_field().eval()
        self.task = task_name_to_class(TASK)
        self.g = prepared(self.vf, self.task)

    def test_w1_is_given_and_w0_is_unknown(self):
        given = run(self.vf, self.g, self.task, None).nodes['lig'].data['x_t']
        w1 = run(self.vf, self.g, self.task, 1.0).nodes['lig'].data['x_t']
        torch.testing.assert_close(w1, given, rtol=1e-5, atol=1e-5)
        unknown_graph = deepcopy(self.g)
        unknown_graph.nodes['lig'].data['stereo_atom'] = unknown_graph.nodes['lig'].data['stereo_atom_unk'].clone()
        unknown = run(self.vf, unknown_graph, self.task, None).nodes['lig'].data['x_t']
        w0 = run(self.vf, self.g, self.task, 0.0).nodes['lig'].data['x_t']
        torch.testing.assert_close(w0, unknown, rtol=1e-5, atol=1e-5)

    def test_labels_matter_and_guidance_extrapolates(self):
        w1 = run(self.vf, self.g, self.task, 1.0).nodes['lig'].data['x_t']
        w0 = run(self.vf, self.g, self.task, 0.0).nodes['lig'].data['x_t']
        w3 = run(self.vf, self.g, self.task, 3.0).nodes['lig'].data['x_t']
        self.assertGreater(float((w1 - w0).abs().max()), 1e-7)
        self.assertGreater(float((w3 - w1).abs().max()), 1e-7)

    def test_inputs_restored_and_missing_alternative_fails_loudly(self):
        before = {k: v.clone() for k, v in self.g.nodes['lig'].data.items() if k.startswith('stereo')}
        out = run(self.vf, self.g, self.task, 2.0)
        for k, v in before.items():
            torch.testing.assert_close(self.g.nodes['lig'].data[k], v, rtol=0, atol=0)
        broken = deepcopy(self.g)
        del broken.nodes['lig'].data['stereo_atom_unk']
        with self.assertRaises(ValueError):
            run(self.vf, broken, self.task, 2.0)

    def test_default_off_is_unchanged(self):
        a = run(self.vf, self.g, self.task, None).nodes['lig'].data['x_t']
        plain = deepcopy(self.g)
        for store in (plain.nodes['lig'].data, plain.edges['lig_to_lig'].data):
            for key in [k for k in store.keys() if k.endswith('_unk')]:
                del store[key]
        b = run(self.vf, plain, self.task, None).nodes['lig'].data['x_t']
        torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
