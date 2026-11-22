"""Coordinates cannot select the mapping or complete a supplied stereo request."""
import unittest
import hashlib
import json
from pathlib import Path
import tempfile
import os
import subprocess
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
from rdkit import Chem
from omtra.data.supplied_identity import map_supplied_identity, request_signature
from omtra.data.inference_stereo import attach_inference_stereo
from omtra.tasks.register import task_name_to_class
from tests.test_stereo_relational import embedded, graph_from_mol


class SuppliedMappingTests(unittest.TestCase):
    def test_atom_permutation_and_poisoned_coordinates_preserve_requested_identity(self):
        mol = embedded('C[C@H](O)C(=O)O')
        order = [4, 1, 3, 0, 5, 2]
        graph = Chem.RenumberAtoms(mol, order)
        result = map_supplied_identity(graph, 'C[C@H](O)C(=O)O')
        self.assertEqual(result['status'], 'mapped')
        source = Chem.MolFromSmiles(result['smiles'])
        mapped = Chem.RenumberAtoms(source, result['graph_to_identity'])
        self.assertEqual(request_signature(graph), request_signature(mapped))
        for i in range(graph.GetNumAtoms()):
            graph.GetConformer().SetAtomPosition(i, [float('nan')] * 3)
        self.assertEqual(result, map_supplied_identity(graph, 'C[C@H](O)C(=O)O'))

    def test_unspecified_remains_unspecified_and_chemistry_is_not_neutralized(self):
        mol = embedded('C[C@H](O)C(=O)O')
        result = map_supplied_identity(mol, 'CC(O)C(=O)O')
        self.assertEqual(result['status'], 'mapped')
        mapped = Chem.RenumberAtoms(Chem.MolFromSmiles(result['smiles']), result['graph_to_identity'])
        atoms, _ = request_signature(mapped)
        self.assertNotIn(2, atoms)
        self.assertNotIn(3, atoms)
        self.assertIn(1, atoms)
        result = map_supplied_identity(Chem.MolFromSmiles('NCC(=O)O'), '[NH3+]CC(=O)[O-]')
        self.assertEqual(result['status'], 'chemistry_mismatch')

    def test_stereo_ambiguous_automorphisms_and_capped_search_are_not_guessed(self):
        results = [map_supplied_identity(Chem.MolFromSmiles(s), s) for s in
                   ('C[C@H](O)[C@@H](O)C', 'C[C@H](O)[C@H](O)C')]
        self.assertTrue(any(r['status'] == 'ambiguous_stereo_mapping' for r in results), results)
        result = map_supplied_identity(Chem.MolFromSmiles('CC(C)(C)C'), 'CC(C)(C)C', max_matches=1)
        self.assertEqual(result['status'], 'mapping_budget_exceeded')
        self.assertNotIn('graph_to_identity', result)

    def test_invalid_and_isotopic_identities_are_explicit(self):
        mol = Chem.MolFromSmiles('CCO')
        self.assertEqual(map_supplied_identity(mol, 'not_a_smiles')['status'], 'invalid_smiles')
        self.assertEqual(map_supplied_identity(mol, '[13CH3]CO')['status'], 'unsupported_isotope')

    def test_unreviewed_candidates_cannot_be_used_as_supplied_inference(self):
        mol = embedded('C[C@H](O)C(=O)O')
        g = graph_from_mol(mol)
        record = map_supplied_identity(mol, 'C[C@H](O)C(=O)O')
        record['review_status'] = 'unreviewed'
        with self.assertRaisesRegex(ValueError, 'unreviewed'):
            attach_inference_stereo([g], task_name_to_class('ligand_conformer_condensed'),
                                    pd.DataFrame([{'system_id': 's', 'ligand_id': 'l'}]), {'s|l': record})


    def test_meso_isomer_identity_is_not_atom_index_centre_agreement(self):
        from omtra.data.identity_audit import molecular_stereo_compatibility, assess_stereo_compatibility
        for smiles, equivalent in [('C[C@H](O)[C@H](O)C', True),
                                   ('C[C@H](O)[C@@H](O)C', False),
                                   ('C[C@H](O)[C@@H](O)[C@H](O)C', True)]:
            mol = embedded(smiles)
            mirror = mol.GetConformer().GetPositions() * np.array([-1., 1., 1.])
            self.assertFalse(assess_stereo_compatibility(mol, mirror)['compatible'])
            result = molecular_stereo_compatibility(mol, mirror)
            self.assertEqual(result['compatible'], equivalent, smiles)
            self.assertNotIn('error', result)

    def test_molecular_check_enforces_ez_and_preserves_unspecified_requests(self):
        from omtra.data.identity_audit import molecular_stereo_compatibility
        trans = embedded('F/C=C/F'); cis = embedded('F/C=C\\F')
        self.assertFalse(molecular_stereo_compatibility(trans, cis.GetConformer().GetPositions())['compatible'])
        self.assertTrue(molecular_stereo_compatibility(Chem.MolFromSmiles('FC=CF'), cis.GetConformer().GetPositions())['compatible'])
        self.assertFalse(molecular_stereo_compatibility(trans, np.zeros((4, 3)))['compatible'])


    def test_explicit_fixed_crop_is_honoured_on_validation_and_test_only(self):
        from omtra.dataset.plinder import PlinderDataset
        from omtra.dataset.crossdocked import CrossdockedDataset
        from omtra.dataset.zarr_dataset import ZarrDataset
        def fake_init(dataset, *args, **kwargs):
            dataset.root = SimpleNamespace(attrs={'system_lookup': [
                {'system_id': 's', 'ligand_id': 'l', 'lig_atom_start': 0, 'lig_atom_end': 6}],
                'npnde_lookup': []})
        with patch.object(ZarrDataset, '__init__', fake_init):
            for split, expected in [('train', None), ('val', 10.), ('test', 10.)]:
                for cls in (PlinderDataset, CrossdockedDataset):
                    kwargs = {'link_version': 'no_links'} if cls is PlinderDataset else {}
                    ds = cls(split=split, processed_data_dir='unused', additional_prot_crop=10., **kwargs)
                    self.assertEqual(ds.additional_prot_crop, expected)


if __name__ == '__main__':
    unittest.main()
