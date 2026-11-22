"""matched_geometry training labels: pose-derived labels kept only for verified centres / E-Z bonds; the rest train as 'unspecified' while the true-centre mask is unchanged."""
import json, tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from omtra.data.requested_stereo import attach_geometry, attach_identity, hide_stereo, hide_unverified
from omtra.data.matched_stereo import BoundMatchedView, MatchedMasks, VERSION
from omtra.data.identity_catalog import file_hash, view_fingerprint
from tests.test_stereo_relational import graph_from_mol, embedded

FIXED = SimpleNamespace(groups_fixed=['ligand_identity_condensed'])
FREE = SimpleNamespace(groups_fixed=[])
SMILES = 'C[C@H](O)[C@@H](O)[C@H](O)C'


def labelled(smiles=SMILES):
    g = graph_from_mol(embedded(smiles))
    attach_geometry(g, FIXED)
    return g


class HideUnverified(unittest.TestCase):
    def test_only_verified_centres_keep_their_label(self):
        g = labelled()
        centres = g.nodes['lig'].data['stereo_centre'].nonzero().flatten().tolist()
        self.assertEqual(len(centres), 3)
        before = g.nodes['lig'].data['stereo_atom'].clone()
        n_atoms, n_bonds = hide_unverified(g, [centres[0]], [])
        nd = g.nodes['lig'].data
        self.assertEqual((n_atoms, n_bonds), (2, 0))
        self.assertEqual(int(nd['stereo_atom'][centres[0]]), int(before[centres[0]]))
        for c in centres[1:]:
            self.assertEqual(int(nd['stereo_atom'][c]), 1)                      # 'unspecified'
        self.assertEqual(nd['stereo_centre'].nonzero().flatten().tolist(), centres)    # the true-centre mask (chirality loss) is untouched
        self.assertEqual(nd['stereo_hidden'].nonzero().flatten().tolist(), centres[1:])

    def test_no_verified_sites_hides_everything_and_all_verified_changes_nothing(self):
        g = labelled(); centres = g.nodes['lig'].data['stereo_centre'].nonzero().flatten().tolist()
        before = g.nodes['lig'].data['stereo_atom'].clone()
        hide_unverified(g, centres, [])
        self.assertTrue(torch.equal(g.nodes['lig'].data['stereo_atom'], before))
        hide_unverified(g, [], [])
        self.assertTrue(bool((g.nodes['lig'].data['stereo_atom'][centres] == 1).all()))

    def test_ez_bonds_are_verified_per_bond_in_both_directions(self):
        g = labelled('F/C=C/[C@H](O)C')
        ed = g.edges['lig_to_lig'].data
        spec = ((ed['stereo_bond'] >= 2) & (ed['stereo_bond'] <= 3)).nonzero().flatten()
        self.assertGreater(len(spec), 0)
        src, dst = g.edges(etype='lig_to_lig')
        pair = (int(src[spec[0]]), int(dst[spec[0]]))
        hide_unverified(g, [], [pair])
        self.assertTrue(bool(((ed['stereo_bond'][spec] >= 2) & (ed['stereo_bond'][spec] <= 3)).all()))      # kept in both directions
        hide_unverified(g, [], [])
        self.assertTrue(bool((ed['stereo_bond'][spec] == 1).all()))

    def test_later_random_hiding_does_not_erase_the_matched_hidden_mask(self):
        g = labelled(); centres = g.nodes['lig'].data['stereo_centre'].nonzero().flatten().tolist()
        hide_unverified(g, [centres[0]], [])
        hide_stereo(g, p_all=0.0, p_partial=1.0, partial_rate=0.0)
        self.assertEqual(g.nodes['lig'].data['stereo_hidden'].nonzero().flatten().tolist(), centres[1:])
        hide_stereo(g, p_all=1.0)
        self.assertEqual(g.nodes['lig'].data['stereo_hidden'].nonzero().flatten().tolist(), centres)


class FakeDataset:
    def __init__(self, store_path, system_lookup):
        self.store_path, self.system_lookup = store_path, system_lookup

    def __len__(self):
        return len(self.system_lookup)


def fake_masks(tmp, per_record, lookup):
    """per_record: list of (verified atoms, verified bonds) or None (no independent identity)."""
    store = Path(tmp) / 'store.zarr'; store.mkdir()
    (store / 'zarr.json').write_text('{}')
    ds = FakeDataset(store, lookup)
    ao, bo, atoms, bonds, has = [0], [0], [], [], []
    for item in per_record:
        has.append(item is not None)
        if item is not None:
            atoms += item[0]; bonds += item[1]
        ao.append(len(atoms)); bo.append(len(bonds))
    root = Path(tmp) / 'masks'; root.mkdir()
    np.savez(root / 'masks.npz', plinder_atom_offsets=np.asarray(ao), plinder_atoms=np.asarray(atoms, dtype=np.int32), plinder_bond_offsets=np.asarray(bo),
             plinder_bonds=np.asarray(bonds, dtype=np.int32).reshape(-1, 2), plinder_has_identity=np.asarray(has))
    manifest = {'version': VERSION, 'masks_sha256': file_hash(root / 'masks.npz'),
                'views': {'plinder/no_links/train': {'key': 'plinder', 'count': len(lookup), 'fingerprint': view_fingerprint(ds)}}}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root, ds


LOOKUP = pd.DataFrame({'system_id': ['a', 'b'], 'ligand_id': ['1.A', '1.B'], 'lig_atom_start': [0, 10], 'lig_atom_end': [10, 20], 'lig_bond_start': [0, 9], 'lig_bond_end': [9, 18]})


class Policy(unittest.TestCase):
    def test_binding_attach_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            g0 = labelled(); centres = g0.nodes['lig'].data['stereo_centre'].nonzero().flatten().tolist()
            root, ds = fake_masks(tmp, [([centres[1]], []), None], LOOKUP)
            masks = MatchedMasks(root)
            view = masks.bind('plinder/no_links/train', ds)
            g = graph_from_mol(embedded(SMILES)); view.attach(g, FIXED, 0)
            nd = g.nodes['lig'].data
            self.assertEqual(nd['stereo_atom'].gt(1).nonzero().flatten().tolist(), [centres[1]])
            g = graph_from_mol(embedded(SMILES)); view.attach(g, FIXED, 1)                       # record without an independent identity: nothing contradicts the pose, every label kept
            self.assertEqual(g.nodes['lig'].data['stereo_atom'].gt(1).nonzero().flatten().tolist(), centres)
            self.assertEqual(int(g.nodes['lig'].data['stereo_hidden'].sum()), 0)
            self.assertEqual(g.nodes['lig'].data['stereo_centre'].nonzero().flatten().tolist(), centres)
            g = graph_from_mol(embedded(SMILES)); view.attach(g, FREE, 0)                        # task without a fixed ligand identity: cleared
            self.assertEqual(int(g.nodes['lig'].data['stereo_atom'].sum()), 0)
            self.assertTrue(bool(view.allowed_mask(0, 2).all()))
            moved = LOOKUP.copy(); moved.loc[0, 'ligand_id'] = '9.Z'
            other = FakeDataset(ds.store_path, moved)
            with self.assertRaises(ValueError):
                masks.bind('plinder/no_links/train', other)                                     # membership changed -> refuse
            with self.assertRaises(ValueError):
                masks.bind('plinder/no_links/val', ds)                                          # unknown view
            (root / 'masks.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                MatchedMasks(root)                                                              # arrays changed after the manifest


PHOSPHORUS = 'C[C@H](O)C[P@](=O)(OC)c1ccccc1'      # one carbon centre and one phosphorus centre
ONLY_PHOSPHORUS = 'C[P@](=O)(OC)c1ccccc1'


class PhosphorusIsNeverACentre(unittest.TestCase):
    """Phosphorus tags are drawing conventions (equivalent terminal oxygens): no label, arrow input, relational edge or margin-loss centre, in training and at inference."""

    def check(self, g, smiles):
        nd = g.nodes['lig'].data
        p = [i for i, a in enumerate(Chem.MolFromSmiles(smiles).GetAtoms()) if a.GetSymbol() == 'P']
        self.assertEqual(len(p), 1)
        self.assertEqual(int(nd['stereo_atom'][p[0]]), 0)                                      # 'not a centre', not 'unspecified'
        self.assertEqual(int(nd['stereo_centre'][p[0]]), 0)                                    # the margin loss and the arrows never see it
        self.assertEqual(float(nd['stereo_unit_z'][p[0]]), 0.0)
        self.assertEqual(int(nd['stereo_hidden'].sum()), 0)

    def test_training_geometry_labels_skip_phosphorus_and_keep_carbon(self):
        g = graph_from_mol(embedded(PHOSPHORUS)); attach_geometry(g, FIXED)
        self.check(g, PHOSPHORUS)
        self.assertEqual(int(g.nodes['lig'].data['stereo_centre'].sum()), 1)                   # the carbon centre survives
        self.assertEqual(int(g.nodes['lig'].data['stereo_atom'].gt(1).sum()), 1)

    def test_supplied_identity_skips_phosphorus_and_keeps_carbon(self):
        g = graph_from_mol(embedded(PHOSPHORUS)); attach_identity(g, FIXED, Chem.MolFromSmiles(PHOSPHORUS), source='supplied')
        self.check(g, PHOSPHORUS)
        self.assertEqual(int(g.nodes['lig'].data['stereo_centre'].sum()), 1)
        self.assertEqual(int((g.nodes['lig'].data['stereo_unit_z'] != 0).sum()), 1)

    def test_molecule_with_only_a_phosphorus_tag_has_no_units(self):
        from omtra.aux_losses.stereo_geometry import graph_units
        for attach in (lambda g: attach_geometry(g, FIXED), lambda g: attach_identity(g, FIXED, Chem.MolFromSmiles(ONLY_PHOSPHORUS), source='supplied')):
            g = graph_from_mol(embedded(ONLY_PHOSPHORUS)); attach(g)
            self.assertEqual(int(g.nodes['lig'].data['stereo_centre'].sum()), 0)
            self.assertEqual(graph_units(g)[0].numel(), 0)

    def test_matched_policy_and_random_hiding_do_not_bring_phosphorus_back(self):
        g = graph_from_mol(embedded(PHOSPHORUS)); attach_geometry(g, FIXED)
        hide_unverified(g, [], [])
        hide_stereo(g, p_all=0.0, p_partial=1.0, partial_rate=0.0)
        self.assertEqual(int(g.nodes['lig'].data['stereo_atom'][[i for i, a in enumerate(Chem.MolFromSmiles(PHOSPHORUS).GetAtoms()) if a.GetSymbol() == 'P'][0]]), 0)


if __name__ == '__main__':
    unittest.main()
