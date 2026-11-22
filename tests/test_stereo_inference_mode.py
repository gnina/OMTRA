"""Audit finding 7: the stereo-correction feature must work under torch.no_grad AND torch.inference_mode, identically, without touching network gradients."""
import unittest

import torch

from omtra.models.embeddings.stereo_correction import StereoCorrection
from omtra.tasks.register import task_name_to_class
from tests.test_stereo_correction import branched_graph
from tests.test_stereo_model import make_field


class T(unittest.TestCase):
    def setUp(self):
        self.task = task_name_to_class('rigid_docking_condensed')
        self.g = branched_graph(self.task, make_field())
        self.sc = StereoCorrection(8, 3)
        x = self.g.nodes['lig'].data['x_1_true'] * torch.tensor([-1., 1., 1.])        # mirror image: wrong hand ...
        self.x = x + 0.15 * torch.randn(6, 3, generator=torch.Generator().manual_seed(4))   # ... and distorted (the exact mirrored ideal tetrahedron is a stationary point)

    def test_feature_identical_in_grad_no_grad_and_inference_mode(self):
        ctx = StereoCorrection.units(self.g)
        f_grad = self.sc.feature(self.x, ctx)
        with torch.no_grad():
            f_ng = self.sc.feature(self.x, ctx)
        with torch.inference_mode():
            x_inf = self.x.clone(); ctx_inf = StereoCorrection.units(self.g)
            f_inf = self.sc.feature(x_inf, ctx_inf)
        torch.testing.assert_close(f_ng, f_grad, rtol=0, atol=0); torch.testing.assert_close(f_inf.clone(), f_grad, rtol=0, atol=0)
        self.assertGreater(float(f_grad.abs().sum()), 0.0)                      # the unit is visible and the noisy coordinates violate it: a real arrow
        self.assertFalse(f_grad.requires_grad); self.assertFalse(f_inf.requires_grad)

    def test_no_parameter_gradient_accumulates_and_training_gradient_still_flows_to_weights(self):
        ctx = StereoCorrection.units(self.g)
        with torch.inference_mode():
            self.sc.feature(self.x.clone(), StereoCorrection.units(self.g))
        self.assertIsNone(self.sc.w_current.grad)
        f = self.sc.feature(self.x, ctx)
        out = self.sc.inject(torch.zeros(6, 8, 3), 0, f, f)
        self.sc.w_current.data.fill_(0.1)
        out = self.sc.inject(torch.zeros(6, 8, 3), 0, f, f); out.sum().backward()
        self.assertIsNotNone(self.sc.w_current.grad); self.assertGreater(float(self.sc.w_current.grad.abs().sum()), 0.0)


if __name__ == '__main__':
    unittest.main()
