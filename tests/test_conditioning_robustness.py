import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import dgl

from omtra.models.conditioning import prepare_coordinate_condition, prepare_partial_condition
from omtra.models.embeddings.fixed_cond_embed import FixedConditionEmbedder
from omtra.dataset.partial_masking import sample_partial_mask
from omtra.data.graph.utils import build_lig_edge_idxs, get_upper_edge_mask
from omtra.tasks.register import task_name_to_class
from omtra.eval.pharmacophore import compute_pharmacophore_match


def graph(n=6):
    src, dst = build_lig_edge_idxs(n)
    g = dgl.heterograph({('lig', 'lig_to_lig', 'lig'): (src, dst),
                         ('pharm', 'pharm_to_pharm', 'pharm'): ([], [])},
                        num_nodes_dict={'lig': n, 'pharm': 3})
    data = g.nodes['lig'].data
    data['x_1_true'] = torch.randn(n, 3)
    data['x_t'] = torch.randn(n, 3)
    data['cond_a_1_true'] = torch.arange(n) % 3
    data['atom_mask_1_true'] = (torch.arange(n) < n//2).long()
    g.edges['lig_to_lig'].data['e_1_true'] = torch.zeros(len(src), dtype=torch.long)
    g.edges['lig_to_lig'].data['edge_mask_1_true'] = (data['atom_mask_1_true'][src] & data['atom_mask_1_true'][dst])
    g.nodes['pharm'].data['x_1_true'] = torch.zeros(3, 3)
    return g


class ConditioningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_pharmacophore_input_moves_clean_reference_does_not(self):
        g = graph()
        clean = g.nodes['pharm'].data['x_1_true'].clone()
        prepare_coordinate_condition(g, 'pharm', 0.8, resample=True)
        data = g.nodes['pharm'].data
        self.assertTrue(torch.equal(data['x_1_true'], clean))
        self.assertFalse(torch.equal(data['x_t'], clean))
        self.assertTrue(torch.equal(data['x_t'], data['x_cond']))
        self.assertTrue(((data['pharm_pos_std'] >= 0) & (data['pharm_pos_std'] <= .8)).all())
        before = data['x_cond'].clone()
        prepare_coordinate_condition(g, 'pharm')
        self.assertTrue(torch.equal(data['x_cond'], before))

    def test_provided_observation_is_preserved(self):
        g = graph()
        g.nodes['pharm'].data['x_cond'] = torch.ones(3, 3)
        prepare_coordinate_condition(g, 'pharm', uncertainty=.4)
        self.assertTrue(torch.equal(g.nodes['pharm'].data['x_t'], torch.ones(3, 3)))
        self.assertTrue(torch.equal(g.nodes['pharm'].data['x_1_true'], torch.zeros(3, 3)))

    def test_partial_noise_stable_and_hidden_targets_do_not_leak(self):
        task = task_name_to_class('partial_denovo_ligand_condensed')
        g = graph()
        clean = g.nodes['lig'].data['x_1_true'].clone()
        prepare_partial_condition(g, task, .5, .2, resample=True)
        upper = get_upper_edge_mask(g, 'lig_to_lig')
        self.assertTrue(torch.equal(g.edges['lig_to_lig'].data['e_cond'][upper], g.edges['lig_to_lig'].data['e_cond'][~upper]))
        emb = FixedConditionEmbedder(task.partial_modalities_fixed, 8, 8, 4)
        first = emb(g, task)
        unknown = ~g.nodes['lig'].data['atom_mask_1_true'].bool()
        self.assertTrue(torch.equal(g.nodes['lig'].data['x_1_true'], clean))
        g.nodes['lig'].data['x_1_true'][unknown] += 100
        g.nodes['lig'].data['cond_a_1_true'][unknown] = 999
        second = emb(g, task)
        for a,b in zip(first,second):
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        sum(t.sum() for dic in first for t in dic.values()).backward()
        self.assertTrue(any(p.grad is not None for p in emb.parameters()))

    def test_partial_embedding_rotation_translation_equivariance(self):
        task = task_name_to_class('partial_ligand_conformer_condensed')
        g = graph()
        prepare_partial_condition(g, task, .5, resample=True)
        emb = FixedConditionEmbedder(task.partial_modalities_fixed, 8, 8, 4)
        a = emb(g, task)
        rotation = torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        for field in ['x_t','x_cond']:
            g.nodes['lig'].data[field] = g.nodes['lig'].data[field] @ rotation + 13
        b = emb(g, task)
        torch.testing.assert_close(a[0]['lig'], b[0]['lig'], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(a[2]['lig'] @ rotation, b[2]['lig'], atol=2e-5, rtol=2e-5)

    def test_mask_modes_and_single_fragment_fallback(self):
        np.random.seed(9)
        for mode in ['whole_fragments','fragment_bernoulli_atom','mixed']:
            for _ in range(20):
                mask = sample_partial_mask(np.array([0,0,1,1,2,2]), mode)
                self.assertTrue(np.isin(mask,[0,1]).all())
                if mode == 'whole_fragments':
                    self.assertTrue((mask[::2] == mask[1::2]).all())
        self.assertTrue(sample_partial_mask(np.zeros(5), 'fragment_bernoulli_atom', 1).all())
        with self.assertRaises(ValueError):
            sample_partial_mask([0,1], 'typo')

    def test_batching_keeps_one_noisy_input_per_system(self):
        from omtra.models.omtra import OMTRA
        captured = []
        def sampler(g_list, n_replicates, **kwargs):
            result = []
            for g in g_list:
                for _ in range(n_replicates):
                    captured.append(g.nodes['pharm'].data['x_cond'].clone())
                    result.append(object())
            return result
        model = SimpleNamespace(vector_field=SimpleNamespace(soft_partial_conditioning=True), sample=sampler)
        g = graph()
        OMTRA.sample_in_batches(model, 'ligand_conformer_from_pharmacophore_condensed',
            g_list=[g], n_replicates=3, max_batch_size=1, pharm_input_noise_std=.8)
        self.assertEqual(len(captured), 3)
        for value in captured[1:]:
            torch.testing.assert_close(value, captured[0], rtol=0, atol=0)
        self.assertTrue(torch.equal(g.nodes['pharm'].data['x_1_true'], torch.zeros(3,3)))
        self.assertNotIn('x_cond', g.nodes['pharm'].data)

    def test_empty_pharmacophore_input_is_not_a_match(self):
        from omtra.eval.system import SampledSystem
        g = graph()
        g.remove_nodes(torch.arange(3), ntype='pharm')
        sample = SampledSystem.__new__(SampledSystem)
        sample.g = g
        for reference in ['clean', 'input']:
            self.assertEqual(compute_pharmacophore_match([sample], reference=reference)['frac_pharm_samples_matching'], 0)

    def test_metric_uses_clean_A_unless_input_B_requested(self):
        sample = SimpleNamespace(get_rdkit_ligand=lambda: object(),
            get_pharmacophore_from_graph=lambda kind: {'coords': [[0.,0.,0.]] if kind=='gt' else [[2.,0.,0.]], 'types_idx':[1]})
        with patch('omtra.eval.pharmacophore.get_pharmacophores', return_value=([[0.,0.,0.]], [1], None, None)):
            self.assertEqual(compute_pharmacophore_match([sample])['frac_pharm_samples_matching'], 1)
            self.assertEqual(compute_pharmacophore_match([sample], reference='input')['frac_pharm_samples_matching'], 0)
        with patch('omtra.eval.pharmacophore.get_pharmacophores', return_value=([], [], None, None)):
            self.assertEqual(compute_pharmacophore_match([sample])['frac_true_centers_matched'], 0)
        self.assertEqual(compute_pharmacophore_match([])['frac_pharm_samples_matching'], 0)


class TrainingLossTests(unittest.TestCase):
    def test_weights_apply_without_mutating_labels(self):
        from torch import nn
        from omtra.models.omtra import OMTRA
        from omtra.tasks.modalities import name_to_modality
        from unittest.mock import patch
        task = SimpleNamespace(modalities_generated=[name_to_modality('lig_x'), name_to_modality('lig_cond_a')],
                               groups_present=['protein_identity'], groups_fixed=[], partial_modalities_fixed=[])
        g = graph()
        categories = name_to_modality('lig_cond_a').n_categories
        g.nodes['lig'].data['cond_a_t'] = torch.full((6,), categories, dtype=torch.long)
        original = g.nodes['lig'].data['cond_a_1_true'].clone()
        class StubField(nn.Module):
            soft_partial_conditioning = False
            def forward(self, graph, *args, **kwargs):
                return {'lig_x': graph.nodes['lig'].data['x_1_true'] + 1,
                        'lig_cond_a': torch.zeros((6, categories), requires_grad=True)}
        model = OMTRA.__new__(OMTRA)
        nn.Module.__init__(model)
        model.vector_field = StubField()
        model.train_t_sampler = lambda batch_size, device: torch.full((batch_size,), .5, device=device)
        model.sample_conditional_path = lambda graph,*a,**kw: graph
        model.lig_com_offset_std = model.distort_p = model.prot_pos_std = model.fake_atom_p = 0
        model.time_scaled_loss = True
        model.cat_loss_weight = .8
        model.loss_fn_dict = nn.ModuleDict({'lig_cond_a': nn.CrossEntropyLoss(ignore_index=-100)})
        model.aux_loss_fn_dict = nn.ModuleDict()
        with patch('omtra.models.omtra.task_name_to_class', return_value=task):
            losses = model(g, 'test')
        self.assertAlmostEqual(float(losses['lig_x']), 4.0)
        self.assertAlmostEqual(float(losses['lig_cond_a']), float(torch.log(torch.tensor(float(categories)))*.8*4), places=4)
        self.assertTrue(torch.equal(g.nodes['lig'].data['cond_a_1_true'], original))
        g.nodes['lig'].data['cond_a_t'] = original.clone()
        with patch('omtra.models.omtra.task_name_to_class', return_value=task):
            losses = model(g, 'test')
        self.assertEqual(float(losses['lig_cond_a']), 0)
        self.assertTrue(torch.equal(g.nodes['lig'].data['cond_a_1_true'], original))

if __name__ == '__main__':
    unittest.main()
