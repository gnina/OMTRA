"""Source identity loading, quarantine sampling, completeness and worker boundaries."""
import json
import pickle
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
import torch
import dgl
from rdkit import Chem
from rdkit.Chem import AllChem
from omtra.constants import lig_atom_type_map,charge_map
from omtra.data.graph.utils import build_lig_edge_idxs
from omtra.data.identity_catalog import build_catalog,IdentityCatalog,view_fingerprint,file_hash
from omtra.dataset.chunk_tracker import GraphChunkTracker
from omtra.dataset.multitask import MultitaskDataSet
from omtra.tasks.register import task_name_to_class


def graph():
    mol=Chem.AddHs(Chem.MolFromSmiles('C[C@H](O)C(=O)O'))
    assert AllChem.EmbedMolecule(mol,randomSeed=41)==0
    mol=Chem.RemoveHs(mol);src,dst=build_lig_edge_idxs(mol.GetNumAtoms())
    g=dgl.heterograph({('lig','lig_to_lig','lig'):(src,dst)})
    g.nodes['lig'].data.update(x_1_true=torch.tensor(mol.GetConformer().GetPositions(),dtype=torch.float32),
        a_1_true=torch.tensor([lig_atom_type_map.index(a.GetSymbol()) for a in mol.GetAtoms()]),
        c_1_true=torch.tensor([charge_map.index(a.GetFormalCharge()) for a in mol.GetAtoms()]))
    g.edges['lig_to_lig'].data['e_1_true']=torch.tensor([0 if mol.GetBondBetweenAtoms(int(a),int(b)) is None else int(mol.GetBondBetweenAtoms(int(a),int(b)).GetBondTypeAsDouble()) for a,b in zip(src,dst)])
    return g


class Dataset:
    weighted_sampling=False
    graphs_per_chunk=2
    def __init__(self,root):
        self.store_path=Path(root);self.store_path.mkdir();(self.store_path/'zarr.json').write_text('{}')
        self.system_lookup=pd.DataFrame({'system_id':['s0','s1','s2'],'ligand_id':['l']*3,
            'lig_atom_start':[0,6,12],'lig_atom_end':[6,12,18],
            'lig_bond_start':[0,5,10],'lig_bond_end':[5,10,15]})
    def __len__(self):return len(self.system_lookup)
    def __getitem__(self,index):return graph()
    def retrieve_graph_chunks(self,**kwargs):return torch.tensor([[0,1],[1,3]])
    def get_num_edges(self,task,start,end):return torch.full((end-start,),30)


class CatalogTests(unittest.TestCase):
    @contextmanager
    def temporary_directory(self):
        self.catalogs=[]
        with TemporaryDirectory() as tmp:
            try:yield tmp
            finally:
                for catalog in self.catalogs:catalog.close()

    def setup_catalog(self,tmp,smiles='CC(O)C(=O)O',seal=True):
        root=Path(tmp);dataset=Dataset(root/'store')
        source=root/'source.json';source.write_text(json.dumps({'smiles':smiles}))
        manifest={'version':1,'views':{'plinder/no_links/train':{'count':3,'sha256':view_fingerprint(dataset)}},
                  'evidence':{'source':{'path':str(source),'sha256':file_hash(source)}}}
        records=[{'view':'plinder/no_links/train','index':0,'status':'quarantine','reason':'specified target conflict','evidence':'source'}]
        records += [{'view':'plinder/no_links/train','index':i,'status':'approved','source_kind':'source_database',
                     'independent':True,'evidence':'source','smiles':smiles,'graph_to_identity':list(range(6))} for i in (1,2)]
        path=root/'catalog.sqlite';build_catalog(path,manifest,records,seal=seal)
        catalog=IdentityCatalog(path,require_sealed=seal);self.catalogs.append(catalog)
        return dataset,catalog,source,manifest,records

    def test_actual_multitask_loader_preserves_unspecified_and_generative_channel_zero(self):
        with self.temporary_directory() as tmp:
            dataset,catalog,_,_,_=self.setup_catalog(tmp)
            dataset.identity_policy=catalog.bind('plinder/no_links/train',dataset)
            multi=object.__new__(MultitaskDataSet);multi.stereo_features=True
            multi.task_space=['ligand_conformer_condensed','denovo_ligand_condensed']
            multi.dataset_space=['pharmit'];multi.datasets={'pharmit':dataset}
            g,_,_=multi[(0,0,1)]
            self.assertEqual(int((g.nodes['lig'].data['stereo_atom']==1).sum()),1)
            self.assertTrue(torch.all(g.nodes['lig'].data['stereo_source']==1))
            with self.assertRaisesRegex(ValueError,'quarantine'):multi[(0,0,0)]
            g,_,_=multi[(1,0,0)]
            self.assertEqual(int(g.nodes['lig'].data['stereo_atom'].sum()),0)
            self.assertEqual(int(g.nodes['lig'].data['stereo_source'].sum()),0)

    def test_contradictory_target_rejected_instead_of_relabelled(self):
        with self.temporary_directory() as tmp:
            dataset,catalog,_,_,_=self.setup_catalog(tmp,'C[C@@H](O)C(=O)O')
            policy=catalog.bind('plinder/no_links/train',dataset);g=graph()
            with self.assertRaisesRegex(ValueError,'conflicts with training target'):
                policy.attach(g,task_name_to_class('ligand_conformer_condensed'),1)
            self.assertEqual(int(g.nodes['lig'].data['stereo_source'].sum()),0)

    def test_sampler_skips_quarantine_chunks_but_retains_generative_membership(self):
        with self.temporary_directory() as tmp:
            dataset,catalog,_,_,_=self.setup_catalog(tmp)
            dataset.identity_policy=catalog.bind('plinder/no_links/train',dataset)
            tracker=GraphChunkTracker(dataset,100,0,1)
            tracker.chunk_queue=torch.tensor([0,1])
            selected=tracker.get_batch_idxs(task_name_to_class('ligand_conformer_condensed'))
            self.assertEqual(set(selected),{1,2})
            tracker.chunk_queue=torch.tensor([0,1]);tracker.chunk_queue_idx=0;tracker.n_samples_served_this_chunk=0
            self.assertEqual(tracker.get_batch_idxs(task_name_to_class('denovo_ligand_condensed')),[0])
            tracker.chunk_index=torch.tensor([[0,1]]);tracker.n_chunks=1;tracker.reset_queue()
            with self.assertRaisesRegex(ValueError,'rank shard'):tracker.get_batch_idxs(task_name_to_class('ligand_conformer_condensed'))

    def test_source_change_pickle_and_membership_mismatch(self):
        with self.temporary_directory() as tmp:
            dataset,catalog,source,_,_=self.setup_catalog(tmp)
            restored=pickle.loads(pickle.dumps(catalog));self.catalogs.append(restored)
            self.assertEqual(restored.record('plinder/no_links/train',1)['status'],'approved')
            dataset.system_lookup=dataset.system_lookup.iloc[::-1].reset_index(drop=True)
            with self.assertRaisesRegex(ValueError,'membership'):catalog.bind('plinder/no_links/train',dataset)
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError,'evidence changed'):catalog.record('plinder/no_links/train',1)

    def test_incomplete_and_candidate_only_catalogs_cannot_serve_training(self):
        with self.temporary_directory() as tmp:
            dataset,catalog,_,manifest,records=self.setup_catalog(tmp,seal=False)
            with self.assertRaisesRegex(ValueError,'not sealed'):IdentityCatalog(catalog.path)
            with self.assertRaisesRegex(ValueError,'Unsealed'):catalog.bind('plinder/no_links/train',dataset)
            output=Path(tmp)/'incomplete.sqlite'
            with self.assertRaisesRegex(ValueError,'Cannot seal'):build_catalog(output,manifest,records[:1],seal=True)
            self.assertFalse(output.exists())
            records[1]['source_kind']='annotation_candidate'
            with self.assertRaisesRegex(ValueError,'independently reviewed'):build_catalog(output,manifest,records,seal=True)



if __name__=='__main__':unittest.main()
