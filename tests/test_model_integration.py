"""Actual vector-field forward/backward checks; no optimizer steps or GPUs."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
import dgl
import torch
from omegaconf import OmegaConf
from rdkit import Chem

from omtra.data.graph import build_complex_graph
from omtra.data.graph import edge_builders
from omtra.data.graph.utils import build_lig_edge_idxs, get_batch_idxs, get_upper_edge_mask
from omtra.data.requested_stereo import attach_geometry, clear_stereo
from omtra.load.conf import build_td_coupling
from omtra.models.interpolant_scheduler import InterpolantScheduler
from omtra.models.vector_field import VectorField
from omtra.tasks.register import task_name_to_class
from omtra.aux_losses.pair import LigPairLoss, ProtLigDist

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['rigid_docking_condensed','fixed_protein_ligand_denovo_condensed',
         'partial_rigid_docking_condensed','rigid_docking_pharmacophore_condensed',
         'flexible_docking_condensed','ligand_conformer_condensed']


def field(self_conditioning=True, stereo=True):
    coupling = build_td_coupling([dict(duration_batches=None,tasks=[dict(name=n,probability=1/len(NAMES)) for n in NAMES])],
                                {n:[['plinder',1.]] for n in NAMES})
    return VectorField(InterpolantScheduler(),coupling,OmegaConf.load(ROOT/'configs/graph/more_prot.yaml'),
        n_hidden_scalars=24,n_hidden_edge_feats=16,n_vec_channels=8,n_cp_feats=2,token_dim=8,
        time_embedding_dim=8,rbf_dim=8,n_molecule_updates=2,convs_per_update=1,n_message_gvps=1,
        n_update_gvps=1,n_expansion_gvps=1,self_conditioning=self_conditioning,
        stereo_features=stereo,soft_partial_conditioning=True,pharm_pos_std_flag=True)


def graph(task, vf, npnde=0):
    nodes={}
    counts={'lig':6,'prot_atom':16,'npnde':npnde,'pharm':3}
    for m in task.modalities_present:
        if not m.is_node:continue
        n=counts[m.entity_name];data=nodes.setdefault(m.entity_name,{})
        if m.is_categorical:v=torch.zeros(n,dtype=torch.long)
        elif m.data_key=='x':v=torch.randn(n,3)
        elif m.data_key=='v':v=torch.zeros(n,4,3)
        elif m.data_key=='pos_enc':v=torch.zeros(n,64)
        else:raise ValueError(m)
        data[m.data_key+'_1_true']=v.clone();data[m.data_key+'_t']=v.clone()
    # A supported all-carbon chain for semantic geometry reconstruction.
    nodes['lig']['a_1_true']=torch.zeros(6,dtype=torch.long)
    nodes['lig']['c_1_true']=torch.full((6,),2,dtype=torch.long)
    nodes['lig']['atom_mask_1_true']=torch.tensor([1,1,1,0,0,0])
    # Condensed carbon token is enough for model mechanics; stereo labels are supplied below.
    src,dst=build_lig_edge_idxs(6)
    edges={'lig_to_lig':torch.stack([src,dst])}
    orders=((src-dst).abs()==1).long()
    edge_data={'lig_to_lig':{'e_1_true':orders,'e_t':orders.clone(),
                           'edge_mask_1_true':((src<3)&(dst<3)).long()}}
    if 'npnde' in nodes:
        a,b=build_lig_edge_idxs(npnde)
        edges['npnde_to_npnde']=torch.stack([a,b])
        edge_data['npnde_to_npnde']={'e_1_true':torch.ones(len(a),dtype=torch.long),'e_t':torch.ones(len(a),dtype=torch.long)}
    if 'pharm' in nodes:nodes['pharm']['pharm_pos_std']=torch.zeros(3,1)
    g=build_complex_graph(nodes,edges,edge_data,task=task,graph_config=vf.graph_config)
    clear_stereo(g)
    g.nodes['lig'].data['stereo_atom'][1]=2
    from omtra.models.conditioning import prepare_partial_condition
    prepare_partial_condition(g,task)
    return g


def predict(vf,g,task):
    return vf(g,task,torch.full((g.batch_size,),.5),get_batch_idxs(g)[0],
              {'lig_to_lig':get_upper_edge_mask(g,'lig_to_lig')})


class VectorFieldTests(unittest.TestCase):
    def test_rotation_and_atom_permutation_equivariance(self):
        torch.manual_seed(113);vf=field().eval();task=task_name_to_class('ligand_conformer_condensed');g=graph(task,vf)
        original=predict(vf,deepcopy(g),task)['lig_x']
        order=torch.arange(5,-1,-1)
        permuted=dgl.node_subgraph(g,{nt:order if nt=='lig' else torch.arange(g.num_nodes(nt)) for nt in g.ntypes})
        predicted=predict(vf,permuted,task)['lig_x']
        torch.testing.assert_close(predicted,original[order],rtol=2e-5,atol=2e-6)
        rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        rotated=deepcopy(g)
        for nt in rotated.ntypes:
            for key,value in list(rotated.nodes[nt].data.items()):
                if key.startswith('x_'):rotated.nodes[nt].data[key]=value@rotation+7
                elif key.startswith('v_'):rotated.nodes[nt].data[key]=value@rotation
        predicted=predict(vf,rotated,task)['lig_x']
        torch.testing.assert_close(predicted,original@rotation+7,rtol=2e-5,atol=2e-6)

    def test_disabled_channel_preserves_legacy_initialization_and_outputs(self):
        torch.manual_seed(55);old=field(stereo=False)
        torch.manual_seed(55);new=field();new.stereo_enabled=False
        for k,v in old.state_dict().items():torch.testing.assert_close(v,new.state_dict()[k],rtol=0,atol=0)
        task=task_name_to_class('rigid_docking_condensed');g=graph(task,new)
        torch.manual_seed(1);a=predict(old,deepcopy(g),task)
        torch.manual_seed(1);b=predict(new,deepcopy(g),task)
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)

    def test_labels_change_actual_outputs_and_receive_gradients(self):
        torch.manual_seed(57);vf=field().eval();task=task_name_to_class('rigid_docking_condensed');g=graph(task,vf)
        a=predict(vf,deepcopy(g),task)
        g.nodes['lig'].data['stereo_atom'][1]=3
        b=predict(vf,deepcopy(g),task)
        self.assertGreater(float((a['lig_x']-b['lig_x']).abs().max()),1e-8)
        sum(v.square().mean() for v in b.values()).backward()
        grad=vf.stereo_embedder.atom.weight.grad
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().sum()),0)

    def test_stale_rebuild_output_and_gradient_parity(self):
        for sc in [False,True]:
            torch.manual_seed(59);vf=field(sc)
            for name in NAMES:
                for n in [0,2]:
                    with self.subTest(task=name,self_conditioning=sc,npnde=n):
                        task=task_name_to_class(name);g=graph(task,vf,n)
                        g=dgl.batch([deepcopy(g),deepcopy(g)])
                        vf.rebuild_edges=False;vf.zero_grad();torch.manual_seed(1);a=predict(vf,deepcopy(g),task)
                        sum(v.square().mean() for v in a.values()).backward()
                        grads={k:p.grad.clone() for k,p in vf.named_parameters() if p.grad is not None}
                        vf.rebuild_edges=True;vf.zero_grad();torch.manual_seed(1);b=predict(vf,deepcopy(g),task)
                        sum(v.square().mean() for v in b.values()).backward()
                        for k in a:torch.testing.assert_close(a[k],b[k],rtol=2e-5,atol=2e-6)
                        for k,p in vf.named_parameters():
                            if k in grads:torch.testing.assert_close(grads[k],p.grad,rtol=2e-4,atol=2e-6)


class EmptyPairTests(unittest.TestCase):
    def test_empty_pairs_have_zero_finite_backward(self):
        task=task_name_to_class('flexible_docking_condensed');vf=field();g=graph(task,vf)
        g.nodes['prot_atom'].data['x_1_true'].fill_(100.)
        g.nodes['lig'].data['x_1_true']=torch.arange(6).float()[:,None].expand(-1,3)*1000
        for weight in [None,torch.tensor([3.])]:
            for loss in [LigPairLoss(),ProtLigDist()]:
                x=torch.randn(6,3,requires_grad=True);p=torch.randn(16,3,requires_grad=True)
                result=loss(g,{'lig_x':x,'prot_atom_x':p},task,get_batch_idxs(g)[0],get_upper_edge_mask(g,'lig_to_lig'),weight)
                self.assertEqual(float(result),0.);result.backward()
                self.assertTrue(torch.isfinite(x.grad).all());self.assertEqual(float(x.grad.abs().sum()),0.)




if __name__=='__main__':unittest.main()
