"""Vector-field level checks for the relational stereo encoding and the open-valence vector channel."""
from copy import deepcopy
from pathlib import Path
import unittest
import dgl
import torch
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem import AllChem

from omtra.data.requested_stereo import clear_stereo
from omtra.load.conf import build_td_coupling
from omtra.models.interpolant_scheduler import InterpolantScheduler
from omtra.models.vector_field import VectorField
from omtra.models.embeddings.implicit_h import open_valence_vectors
from omtra.models.embeddings.stereo import StereoEmbedding
from omtra.tasks.register import task_name_to_class
from tests.test_model_integration import NAMES, graph, predict

ROOT = Path(__file__).resolve().parents[1]


def make_field(task_names=NAMES, **extra):
    coupling = build_td_coupling([dict(duration_batches=None, tasks=[dict(name=n, probability=1 / len(task_names)) for n in task_names])],
                                 {n: [['plinder', 1.]] for n in task_names})
    return VectorField(InterpolantScheduler(), coupling, OmegaConf.load(ROOT / 'configs/graph/more_prot.yaml'),
                       n_hidden_scalars=24, n_hidden_edge_feats=16, n_vec_channels=8, n_cp_feats=2, token_dim=8,
                       time_embedding_dim=8, rbf_dim=8, n_molecule_updates=2, convs_per_update=1, n_message_gvps=1,
                       n_update_gvps=1, n_expansion_gvps=1, self_conditioning=True, stereo_features=True,
                       soft_partial_conditioning=True, pharm_pos_std_flag=True, **extra)


def with_relational(g, seed=0):
    gen = torch.Generator().manual_seed(seed)
    ed = g.edges['lig_to_lig'].data
    rel = torch.zeros(g.num_edges('lig_to_lig'), 5)
    rel[:, :3] = (torch.rand(rel.shape[0], 3, generator=gen) < 0.2).float()
    ed['stereo_rel'] = rel
    return g


class EncodingTests(unittest.TestCase):
    def test_categorical_state_dict_and_outputs_unchanged(self):
        torch.manual_seed(5); a = make_field()
        torch.manual_seed(5); b = make_field(stereo_encoding='categorical', implicit_h_vectors=False)
        for k, v in a.state_dict().items():
            torch.testing.assert_close(v, b.state_dict()[k], rtol=0, atol=0)
        task = task_name_to_class('rigid_docking_condensed'); g = graph(task, a)
        torch.manual_seed(1); x = predict(a, deepcopy(g), task)
        torch.manual_seed(1); y = predict(b, deepcopy(g), task)
        for k in x:
            torch.testing.assert_close(x[k], y[k], rtol=0, atol=0)

    def test_new_modes_only_add_the_relational_projection(self):
        keys = lambda m: set(StereoEmbedding(8, 6, mode=m).state_dict())
        self.assertEqual(keys('categorical'), {'atom.weight', 'bond.weight'})
        self.assertEqual(keys('relational'), {'atom.weight', 'bond.weight', 'rel.weight'})
        with self.assertRaises(ValueError):
            StereoEmbedding(8, 6, mode='nope')

    def test_relational_labels_change_outputs_and_get_gradients_but_not_on_generative_tasks(self):
        torch.manual_seed(9); vf = make_field(stereo_encoding='relational').eval()
        task = task_name_to_class('rigid_docking_condensed'); g = graph(task, vf)
        a = predict(vf, deepcopy(g), task)
        b = predict(vf, deepcopy(with_relational(deepcopy(g))), task)
        self.assertGreater(float((a['lig_x'] - b['lig_x']).abs().max()), 1e-8)
        vf.zero_grad()
        sum(v.square().mean() for v in b.values()).backward()
        grad = vf.stereo_embedder.rel.weight.grad
        self.assertTrue(torch.isfinite(grad).all()); self.assertGreater(float(grad.abs().sum()), 0)
        denovo = task_name_to_class('fixed_protein_ligand_denovo_condensed'); gd = graph(denovo, vf)
        c = predict(vf, deepcopy(gd), denovo)
        d = predict(vf, with_relational(deepcopy(gd)), denovo)
        for k in c:
            torch.testing.assert_close(c[k], d[k], rtol=0, atol=0)

    def test_open_valence_channel_changes_identity_fixed_tasks_only(self):
        # S4 (2026-10-19): masked, gated H direction. Gates start at zero (exact function preservation); with open gates the
        # feature changes identity-fixed tasks only. Eligibility metadata is attached to the graph by attach_identity.
        torch.manual_seed(11); plain = make_field(); torch.manual_seed(11); hvec = make_field(implicit_h_vectors=True)
        hvec.load_state_dict(plain.state_dict(), strict=False); plain.eval(); hvec.eval()
        fixed = task_name_to_class('rigid_docking_condensed'); g = graph(fixed, plain)
        g.nodes['lig'].data['h_eligible'] = torch.tensor([0., 1., 1., 1., 1., 0.]); g.nodes['lig'].data['is_hydrogen'] = torch.zeros(6)
        a, b = predict(plain, deepcopy(g), fixed), predict(hvec, deepcopy(g), fixed)
        torch.testing.assert_close(a['lig_x'], b['lig_x'], rtol=0, atol=0)          # zero gates: unchanged
        with torch.no_grad():
            hvec.h_gate_current.fill_(.5)
        b = predict(hvec, deepcopy(g), fixed)
        self.assertGreater(float((a['lig_x'] - b['lig_x']).abs().max()), 1e-8)
        generated = task_name_to_class('fixed_protein_ligand_denovo_condensed'); gd = graph(generated, plain)
        gd.nodes['lig'].data['h_eligible'] = torch.tensor([0., 1., 1., 1., 1., 0.]); gd.nodes['lig'].data['is_hydrogen'] = torch.zeros(6)
        c, d = predict(plain, deepcopy(gd), generated), predict(hvec, deepcopy(gd), generated)
        for k in c:
            torch.testing.assert_close(c[k], d[k], rtol=0, atol=0)

    def test_rotation_and_permutation_equivariance_with_all_new_features(self):
        torch.manual_seed(113); vf = make_field(stereo_encoding='both', implicit_h_vectors=True).eval()
        task = task_name_to_class('ligand_conformer_condensed'); g = with_relational(graph(task, vf))
        original = predict(vf, deepcopy(g), task)['lig_x']
        order = torch.arange(5, -1, -1)
        permuted = dgl.node_subgraph(g, {nt: order if nt == 'lig' else torch.arange(g.num_nodes(nt)) for nt in g.ntypes})
        torch.testing.assert_close(predict(vf, permuted, task)['lig_x'], original[order], rtol=2e-5, atol=2e-6)
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        rotated = deepcopy(g)
        for nt in rotated.ntypes:
            for key, value in list(rotated.nodes[nt].data.items()):
                if key.startswith('x_'):
                    rotated.nodes[nt].data[key] = value @ rotation + 7
                elif key.startswith('v_'):
                    rotated.nodes[nt].data[key] = value @ rotation
        torch.testing.assert_close(predict(vf, rotated, task)['lig_x'], original @ rotation + 7, rtol=2e-5, atol=2e-6)


class OpenValenceTests(unittest.TestCase):
    def test_points_toward_the_real_hydrogen_and_is_equivariant(self):
        mol = Chem.AddHs(Chem.MolFromSmiles('C[C@H](O)C(=O)O'))
        self.assertEqual(AllChem.EmbedMolecule(mol, randomSeed=3), 0)
        AllChem.MMFFOptimizeMolecule(mol)
        xyz = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
        heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        from omtra.data.graph.utils import build_lig_edge_idxs
        index = {old: new for new, old in enumerate(heavy)}
        x = xyz[heavy]
        src, dst = build_lig_edge_idxs(len(heavy))
        orders = torch.tensor([0 if mol.GetBondBetweenAtoms(heavy[int(a)], heavy[int(b)]) is None else 1 for a, b in zip(src, dst)])
        g = dgl.heterograph({('lig', 'lig_to_lig', 'lig'): (src, dst)})
        g.edges['lig_to_lig'].data['e_t'] = orders
        out = open_valence_vectors(g, x)
        stereo_c = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6
                    and sum(n.GetAtomicNum() == 1 for n in a.GetNeighbors()) == 1
                    and sum(n.GetAtomicNum() > 1 for n in a.GetNeighbors()) == 3][0]
        h = [n.GetIdx() for n in mol.GetAtomWithIdx(stereo_c).GetNeighbors() if n.GetAtomicNum() == 1][0]
        true_dir = xyz[h] - xyz[stereo_c]
        cos = torch.nn.functional.cosine_similarity(out[index[stereo_c]], true_dir, dim=0)
        self.assertGreater(float(cos), 0.9)
        q, _ = torch.linalg.qr(torch.randn(3, 3))
        if torch.det(q) < 0:
            q[:, 0] *= -1
        moved = open_valence_vectors(g, x @ q.T + 5.0)
        torch.testing.assert_close(moved, out @ q.T, rtol=1e-4, atol=1e-5)
        # a mirror image reflects the vector: it is a true vector, not a pseudovector
        mirrored = open_valence_vectors(g, x * torch.tensor([-1., 1., 1.]))
        torch.testing.assert_close(mirrored, out * torch.tensor([-1., 1., 1.]), rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
