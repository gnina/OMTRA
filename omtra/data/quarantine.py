"""Dataset membership exclusions, independent of stereo feature/identity sourcing."""
import json
from pathlib import Path
import numpy as np

from omtra.data.identity_catalog import view_fingerprint, file_hash


class MembershipQuarantine:
    def __init__(self, path):
        self.manifest = json.loads(Path(path).read_text())
        if self.manifest.get('version') != 1:
            raise ValueError('Unknown quarantine manifest version')
        for evidence in self.manifest.get('evidence', {}).values():
            if file_hash(evidence['path']) != evidence['sha256']:
                raise ValueError('Quarantine evidence changed')

    def bind(self, view, dataset):
        spec = self.manifest['views'].get(view)
        if spec is None:
            return None
        if spec['count'] != len(dataset) or spec['sha256'] != view_fingerprint(dataset):
            raise ValueError(f'Quarantine does not match dataset membership/schema: {view}')
        indices = spec['indices']
        if (any(type(i) is not int or not 0 <= i < len(dataset) for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('Invalid quarantine indices')
        return BoundQuarantine(indices)


class BoundQuarantine:
    def __init__(self, indices):
        self.indices = np.array(sorted(indices), dtype=np.int64)

    def allowed_mask(self, start, end):
        mask = np.ones(end - start, dtype=bool)
        indices = self.indices[(self.indices >= start) & (self.indices < end)]
        mask[indices - start] = False
        return mask

    def check(self, task, index):
        from omtra.data.requested_stereo import eligible
        if eligible(task) and index in self.indices:
            raise ValueError(f'Quarantined identity-fixed training example: {index}')
