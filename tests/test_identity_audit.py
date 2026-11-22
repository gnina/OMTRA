"""The distinction between unspecified input and contradictory stereo targets."""
import unittest
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import nullcontext
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from omtra.data.identity_audit import assess_stereo_compatibility
from omtra.constants import lig_atom_type_map


def coordinates(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=41) == 0
    return Chem.RemoveHs(mol).GetConformer().GetPositions()


class IdentityCompatibilityTests(unittest.TestCase):
    def test_unspecified_request_accepts_either_enantiomer_without_changing_request(self):
        request = Chem.MolFromSmiles('CC(O)C(=O)O')
        before = Chem.MolToSmiles(request)
        for stereo in ('C[C@H](O)C(=O)O', 'C[C@@H](O)C(=O)O'):
            result = assess_stereo_compatibility(request, coordinates(stereo))
            self.assertTrue(result['compatible'], result)
            self.assertEqual(result['unspecified_atom_centers'], 1)
        self.assertEqual(Chem.MolToSmiles(request), before)

    def test_mirrored_target_conflicts_with_specified_request(self):
        request = Chem.MolFromSmiles('C[C@H](O)C(=O)O')
        xyz = coordinates('C[C@H](O)C(=O)O')
        self.assertTrue(assess_stereo_compatibility(request, xyz)['compatible'])
        xyz[:, 0] *= -1
        result = assess_stereo_compatibility(request, xyz)
        self.assertFalse(result['compatible'])
        self.assertEqual(len(result['atom_conflicts']), 1)

    def test_double_bond_requested_vs_unspecified(self):
        xyz = coordinates('F/C=C/F')
        self.assertTrue(assess_stereo_compatibility(Chem.MolFromSmiles('FC=CF'), xyz)['compatible'])
        self.assertTrue(assess_stereo_compatibility(Chem.MolFromSmiles('F/C=C/F'), xyz)['compatible'])
        result = assess_stereo_compatibility(Chem.MolFromSmiles('F/C=C\\F'), xyz)
        self.assertFalse(result['compatible'])
        self.assertEqual(len(result['bond_conflicts']), 1)

    def test_bad_geometry_and_unsupported_identity_cannot_pass(self):
        request = Chem.MolFromSmiles('CCO')
        for xyz in (np.zeros((2, 3)), np.full((3, 3), np.nan)):
            result = assess_stereo_compatibility(request, xyz)
            self.assertFalse(result['compatible'])
            self.assertIn('error', result)
        request = Chem.MolFromSmiles('[13CH3]CO')
        self.assertFalse(assess_stereo_compatibility(request, np.zeros((3, 3)))['compatible'])


class PharmitSourceLinkageTests(unittest.TestCase):
    def test_database_cursor_retains_ids_and_sdf_reader_retains_record_index(self):
        from omtra_pipelines.pharmit_dataset.guts.phase1 import DBCrawler, read_mol_from_conf_file
        crawler = object.__new__(DBCrawler)
        crawler.max_num_queries = 1
        crawler.query_size = 1
        crawler.conn = SimpleNamespace(cursor=lambda: nullcontext(object()))
        crawler._fetch_next_batch = lambda cursor, last_id: [(51, 'CCO', '/source/51.sdf.gz')]
        self.assertEqual(list(crawler)[0], ((51,), ('CCO',), ('/source/51.sdf.gz',)))
        mol = Chem.MolFromSmiles('CCO')
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'records.sdf.gz'
            with gzip.open(path, 'wt') as stream:
                stream.write('broken\n\n\ninvalid counts\nM  END\n$$$$\n'+Chem.MolToMolBlock(mol)+'\n$$$$\n')
            result, index = read_mol_from_conf_file(path, return_record_index=True)
            self.assertEqual(index, 1)
            self.assertEqual(Chem.MolToSmiles(result), 'CCO')

    def test_original_ids_survive_all_filters_and_phase2_global_offsets(self):
        import omtra_pipelines.pharmit_dataset.run_phase1 as phase1
        import omtra_pipelines.pharmit_dataset.run_phase2 as phase2
        from omtra_pipelines.pharmit_dataset.guts.provenance import decode_records, publish_records
        smiles = ['CCO', 'CCO', 'CC', '[SiH3]CC', 'CCCC', 'C[C@H](O)C(=O)O', 'F/C=C/F']
        with TemporaryDirectory() as tmp:
            directory = Path(tmp);paths = []
            for index, value in enumerate(smiles):
                path = directory/f'{index}.sdf.gz';paths.append(str(path))
                if index == 1:continue  # Source file missing.
                mol = Chem.AddHs(Chem.MolFromSmiles(value))
                self.assertEqual(AllChem.EmbedMolecule(mol, randomSeed=41), 0)
                mol = Chem.RemoveHs(mol);mol.SetProp('_Name', str(index))
                with gzip.open(path, 'wt') as stream:stream.write(Chem.MolToMolBlock(mol)+'\n$$$$\n')
            def pharmacophores(mols):
                kept = [m for m in mols if m.GetProp('_Name') != '4']
                failed = [i for i, m in enumerate(mols) if m.GetProp('_Name') == '4']
                return ([np.zeros((1, 3)) for m in kept], [np.zeros(1, dtype=int) for m in kept],
                        [np.zeros((1, 4, 3)) for m in kept], failed)
            finder = SimpleNamespace(query_name_from_smiles=lambda values: ([['PubChem'] for _ in values], [0]))
            with patch.object(phase1, 'name_finder', finder), patch.object(phase1, 'get_pharmacophore_data', pharmacophores):
                tensors = phase1.process_batch((list(range(100, 107)), smiles, paths), lig_atom_type_map, {}, ['PubChem'], 120, 3)
            self.assertEqual([r['source_id'] for r in tensors['source_records']], [105, 106])
            self.assertEqual([r['source_smiles'] for r in tensors['source_records']], smiles[5:])
            saver = phase1.ChunkSaver(directory/'phase1', dev_mode=True)
            saver.save_chunk_to_disk(tensors, 0)
            chunk = saver.chunk_data_dir/'chunk_data_0.npz'
            with np.load(chunk, allow_pickle=False) as data:
                records = decode_records(data)
                na, ne, np_ = len(data['lig_x']), len(data['lig_e']), len(data['pharm_x'])
                root = {'lig/node': {k: np.zeros((40+na, *data[v].shape[1:]), dtype=data[v].dtype)
                        for k, v in [('x','lig_x'), ('a','lig_a'), ('c','lig_c')]},
                    'lig/edge': {k: np.zeros((20+ne, *data[v].shape[1:]), dtype=data[v].dtype)
                        for k, v in [('e','lig_e'), ('edge_index','lig_edge_idx')]},
                    'pharm/node': {k: np.zeros((5+np_, *data[v].shape[1:]), dtype=data[v].dtype)
                        for k, v in [('x','pharm_x'), ('a','pharm_a'), ('v','pharm_v')]},
                    'db/db': np.zeros((9, data['database'].shape[1]), dtype=bool)}
                for name in ('lig/node', 'lig/edge', 'pharm/node'):root[name]['graph_lookup'] = np.zeros((9, 2), dtype=int)
            info = {'lig_node_start':40, 'lig_node_end':40+na, 'lig_node_offset':40,
                'lig_edge_start':20, 'lig_edge_end':20+ne, 'lig_edge_offset':20,
                'pharm_node_start':5, 'pharm_node_end':5+np_, 'pharm_node_offset':5,
                'lig_node_graph_lookup_start':7, 'lig_node_graph_lookup_end':9,
                'lig_edge_graph_lookup_start':7, 'lig_edge_graph_lookup_end':9,
                'pharm_node_graph_lookup_start':7, 'pharm_node_graph_lookup_end':9,
                'db_start':7, 'db_end':9}
            store = directory/'store';store.mkdir()
            fake_store = SimpleNamespace(root=root, store_path=store, get_chunks_touched=lambda *args: [])
            with patch.object(phase2, 'global_zstore', fake_store):phase2.write_data_to_store(chunk, info)
            sidecar = store/'identity/000000000007_000000000009.jsonl'
            rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
            self.assertEqual([r['source_id'] for r in rows], [105, 106])
            self.assertEqual([r['graph_index'] for r in rows], [7, 8])
            self.assertEqual(rows[0]['atom_start'], 40)
            with np.load(chunk, allow_pickle=False) as data:
                publish_records(store, records, 7, 40, data['node_lookup'], chunk)
                records[0]['source_id'] = 999
                with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
                    publish_records(store, records, 7, 40, data['node_lookup'], chunk)

    def test_legacy_source_less_chunks_fail_before_array_writes(self):
        import omtra_pipelines.pharmit_dataset.run_phase2 as phase2
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'legacy.npz'
            np.savez(path, node_lookup=np.array([[0, 3]]))
            with patch.object(phase2, 'global_zstore', None):
                with self.assertRaisesRegex(ValueError, 'no source identity'):
                    phase2.write_data_to_store(path, {})


if __name__ == '__main__':
    unittest.main()
