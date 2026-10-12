"""Regression tests for defects in the predicted-apo flexible docking path. CPU only, no data required."""
import dgl
import torch
from omtra.tasks.register import task_name_to_class
from omtra.priors.sample import sample_priors

TASK = 'predapo_conditioned_ligand_docking_condensed'


def _prot_graph(n=200):
    g = dgl.heterograph({('prot_atom', 'prot_atom_to_prot_atom', 'prot_atom'): ([], [])}, num_nodes_dict={'prot_atom': n})
    clean = torch.randn(n, 3) * 5
    g.nodes['prot_atom'].data['x_0_true'] = clean.clone()
    g.nodes['prot_atom'].data['x_0'] = clean + torch.randn(n, 3) * 3.0      # an earlier, noisy draw
    g.nodes['prot_atom'].data['x_1_true'] = clean + torch.randn(n, 3) * 0.5
    return g, clean


def _prior(std):
    return {'prot_atom_x': ('apo_pred', lambda x0: x0.clone() + torch.randn_like(x0) * std)}


def test_apo_prior_starts_from_clean_input_not_previous_noisy_draw():
    g, clean = _prot_graph()
    task = task_name_to_class(TASK)
    # zero-noise prior must return exactly the clean input, whatever x_0 held before
    g0 = sample_priors(g, task, _prior(0.0), training=True)
    torch.testing.assert_close(g0.nodes['prot_atom'].data['x_0'], clean, atol=0, rtol=0)


def test_apo_prior_noise_is_drawn_once():
    g, clean = _prot_graph(n=20000)
    task = task_name_to_class(TASK)
    g1 = sample_priors(g, task, _prior(1.0), training=True)
    rms = (g1.nodes['prot_atom'].data['x_0'] - clean).pow(2).mean().sqrt().item()
    assert abs(rms - 1.0) < 0.05, rms          # sqrt(1 + 3^2) = 3.16 if it started from the earlier noisy draw


def test_apo_prior_falls_back_to_x_0_without_clean_copy():
    g, _ = _prot_graph()
    del g.nodes['prot_atom'].data['x_0_true']
    start = g.nodes['prot_atom'].data['x_0'].clone()
    task = task_name_to_class(TASK)
    g0 = sample_priors(g, task, _prior(0.0), training=True)
    torch.testing.assert_close(g0.nodes['prot_atom'].data['x_0'], start, atol=0, rtol=0)
