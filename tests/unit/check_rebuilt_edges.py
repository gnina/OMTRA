"""Synthetic graph regression: real edge builders, denoising loop and GVPs.

usage: python tests/unit/check_rebuilt_edges.py --source .   (CPU, no data needed)

This checks software behavior, not trained-model performance.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--report', type=Path)
args = p.parse_args()
sys.path.insert(0, str(args.source.resolve()))

import dgl
import torch
from omegaconf import OmegaConf
import omtra.tasks.register  # import order: avoids a circular import in omtra.data.graph
from omtra.models.vector_field import VectorField
from omtra.models.gvp import HeteroGVPConv
from omtra.tasks.utils import build_edges
from omtra.data.graph import to_canonical_etype
import omtra.data.graph.edge_builders  # register the actual radius/KNN builders


class Shift(torch.nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, scalars, coords, vectors):
        # Keep a differentiable dependence on the actual convolution output.
        return coords + coords.new_tensor([self.displacement, 0, 0]) + vectors[:, 0] * .001


class CheckedConv(HeteroGVPConv):
    def forward(self, graph, **kwargs):
        xd, dist = kwargs['x_diff'], kwargs['d']
        for relation in self.edge_types:
            count = graph.num_edges(relation)
            if count:
                assert xd[relation].shape == (count, 3)
                assert dist[relation].shape == (count, self.rbf_dim)
                u, v = graph.edges(etype=relation)
                src, _, dst = graph.to_canonical_etype(relation)
                delta = kwargs['coord_feats'][src][u] - kwargs['coord_feats'][dst][v]
                norm = delta.square().sum(-1, keepdim=True).clamp_min(1e-8).sqrt() + 1e-8
                torch.testing.assert_close(xd[relation], delta / norm)
        self.counts.append(graph.num_edges('prot_atom_to_lig'))
        return super().forward(graph, **kwargs)


rows = []
for name, start, shift in [('empty_to_present', 20., -20.),
                           ('present_to_empty', 0., 20.),
                           ('present_changed_neighbors', 3., -6.)]:
    for attention in (False, True):
        torch.manual_seed(42)
        etypes = ['lig_to_lig', 'prot_atom_to_prot_atom',
                  'prot_atom_to_lig', 'lig_to_prot_atom']
        cfg = OmegaConf.create({'edges': {
            'lig_to_lig': {'type': 'complete'},
            'prot_atom_to_prot_atom': {'type': 'radius', 'params': {'r': 10}},
            'prot_atom_to_lig': {'type': 'radius', 'params': {'r': 4}},
            'lig_to_prot_atom': {'type': 'knn', 'params': {'k': 1}},
        }, 'symmetric_etypes': [], 'cov_prot_edges': False})
        generated = SimpleNamespace(entity_name='lig', graph_entity='node',
            data_key='x', is_node=True, is_categorical=False, name='lig_x')
        fixed = SimpleNamespace(entity_name='prot_atom', is_node=True)
        task = type('FixtureTask', (), {'modalities_generated': [generated],
            'modalities_present': [generated, fixed], 'partial_modalities_fixed': []})
        g = dgl.heterograph({to_canonical_etype(e): ([], []) for e in etypes},
            num_nodes_dict={'lig': 2, 'prot_atom': 2})
        # Predetermined ligand edges are preserved by the real rebuild code.
        g.add_edges(torch.tensor([0, 1]), torch.tensor([1, 0]), etype='lig_to_lig')
        positions = {'lig': torch.tensor([[start, 0., 0.], [start, 1., 0.]], requires_grad=True),
                     'prot_atom': torch.tensor([[-3., 0., 0.], [3., 0., 0.]])}
        batch = {n: torch.zeros(2, dtype=torch.long) for n in g.ntypes}
        for n in g.ntypes:
            g.nodes[n].data['x_t'] = positions[n]
        g = build_edges(g, task, batch, cfg)
        model = VectorField.__new__(VectorField)
        torch.nn.Module.__init__(model)
        model.node_types = ['lig', 'prot_atom']
        model.edge_types = etypes
        model.rbf_dmax = 18
        model.rbf_dim = 8
        model.n_recycles = 1
        model.convs_per_update = 1
        model.separate_mol_updaters = False
        model.rebuild_edges = True
        model.graph_config = cfg
        convs = [CheckedConv(model.node_types, etypes, scalar_size=8,
                 vector_size=2, rbf_dim=8, rbf_dmax=18, attention=attention) for _ in range(2)]
        for conv in convs:
            conv.counts = []
        model.conv_layers = torch.nn.ModuleList(convs)
        model.node_position_updaters = torch.nn.ModuleDict({'lig': torch.nn.ModuleList([Shift(shift)])})
        output = model.denoise_graph(g, task,
            {n: torch.randn(2, 8) for n in g.ntypes},
            {n: torch.randn(2, 2, 3) for n in g.ntypes}, positions, {}, batch, {})
        assert torch.isfinite(output['lig_x']).all()
        output['lig_x'].square().sum().backward()
        gradients = [x.grad for x in model.parameters() if x.grad is not None]
        assert gradients and all(torch.isfinite(x).all() for x in gradients)
        counts = [conv.counts[0] for conv in convs]
        if name == 'empty_to_present':
            assert counts[0] == 0 and counts[1] > 0
        elif name == 'present_to_empty':
            assert counts[0] > 0 and counts[1] == 0
        rows.append({'case': name, 'attention': attention, 'radius_edge_counts': counts,
                     'finite_forward_backward': True})
report = {'scope': 'synthetic graph functionality, not model performance', 'passed': True, 'cases': rows,
          'vector_field_sha256': hashlib.sha256((args.source / 'omtra/models/vector_field.py').read_bytes()).hexdigest(),
          'check_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if args.report:
    args.report.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
