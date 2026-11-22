"""Training stereo labels taken from the training pose, kept only where an independent source identity agrees (per centre / E-Z bond).

Sidecar (built offline from an identity-match audit): `manifest.json` + `masks.npz`. For every dataset view it stores the
ordered-membership fingerprint (omtra.data.identity_catalog.view_fingerprint), so indices cannot be silently applied to a different filtered dataset, and for every record the verified
atoms and E/Z bonds. A record WITH an independent identity keeps a label only where that identity agrees with the pose; a centre that was checked and disagrees, or where the identity leaves the centre unspecified, keeps NO label (only `ia == ga` is kept): it trains as
'unspecified'. A record with NO independent identity (nothing can contradict its pose) keeps every pose-derived label: whenever there is a stereo label the model must use it
(same rule for datasets without any identity source, e.g. Pharmit, see dataset/multitask.py). Phosphorus is never a stereocentre (attach_identity): no label, arrow or loss.
Nothing is repaired or dropped; the pose is still the generative target everywhere.
"""
import json
from pathlib import Path
import numpy as np
from omtra.data.identity_catalog import file_hash, view_fingerprint

VERSION = 1


class MatchedMasks:
    def __init__(self, path):
        root = Path(path)
        self.manifest = json.loads((root / 'manifest.json').read_text())
        if self.manifest.get('version') != VERSION:
            raise ValueError('Unknown matched-mask version')
        if file_hash(root / 'masks.npz') != self.manifest['masks_sha256']:
            raise ValueError('Matched-mask arrays changed after the manifest was written')
        self.arrays = {k: v for k, v in np.load(root / 'masks.npz').items()}

    def bind(self, view, dataset):
        spec = self.manifest['views'].get(view)
        if spec is None or spec['count'] != len(dataset) or spec['fingerprint'] != view_fingerprint(dataset):
            raise ValueError(f'Matched masks do not match dataset membership/schema: {view}')
        prefix = spec['key']
        return BoundMatchedView(self, view, *(self.arrays[f'{prefix}_{name}'] for name in ('atom_offsets', 'atoms', 'bond_offsets', 'bonds', 'has_identity')))


class BoundMatchedView:
    def __init__(self, masks, view, atom_offsets, atoms, bond_offsets, bonds, has_identity):
        self.masks, self.view = masks, view
        self.atom_offsets, self.atoms, self.bond_offsets, self.bonds, self.has_identity = atom_offsets, atoms, bond_offsets, bonds, has_identity

    def allowed_mask(self, start, end):
        import numpy as np
        return np.ones(end - start, dtype=bool)          # nothing is excluded: unverified records stay in training, unlabelled

    def attach(self, graph, task, index):
        from omtra.data.requested_stereo import eligible, clear_stereo, attach_geometry, hide_unverified
        if not eligible(task):
            clear_stereo(graph)
            return graph
        attach_geometry(graph, task, 'training_geometry')
        i = int(index)
        if not bool(self.has_identity[i]):
            # No independent identity exists to contradict the pose: whenever there is a stereo label the model must use it, so every pose-derived label is kept
            # (phosphorus is never a centre, see attach_identity). Labels are withheld only where an independent source was checked and DISAGREES with the pose or leaves the centre UNSPECIFIED (below).
            return graph
        keep_atoms = self.atoms[self.atom_offsets[i]:self.atom_offsets[i + 1]].tolist()
        keep_bonds = self.bonds[self.bond_offsets[i]:self.bond_offsets[i + 1]].tolist()
        hide_unverified(graph, keep_atoms, keep_bonds)
        return graph
