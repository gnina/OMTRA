"""Auditable partial initialization: exact key sets and no swallowed shape errors."""
import hashlib
import json
from pathlib import Path
import torch


def state_digest(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def nested_state_digest(value):
    """Stable content digest for optimizer/scheduler state; independent of storage IDs."""
    digest=hashlib.sha256()
    def visit(item):
        digest.update(type(item).__name__.encode())
        if torch.is_tensor(item):digest.update(state_digest({'tensor':item}).encode())
        elif isinstance(item,dict):
            for key in sorted(item,key=lambda k:(type(k).__name__,repr(k))):visit(key);visit(item[key])
        elif isinstance(item,(list,tuple)):
            digest.update(str(len(item)).encode())
            for child in item:visit(child)
        else:digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def load_manifest(target, source, excluded=('vector_field.task_embedding.weight',)):
    excluded = set(excluded)
    loaded, mismatched = [], {}
    for key in sorted(source):
        if key in excluded or key not in target:
            continue
        if source[key].shape != target[key].shape:
            mismatched[key] = {'source': list(source[key].shape), 'target': list(target[key].shape)}
        else:
            loaded.append(key)
    return {'version': 1, 'loaded': loaded, 'excluded': sorted(excluded & set(source)),
            'source_only': sorted(set(source)-set(target)-excluded),
            'new': sorted(set(target)-set(loaded)), 'shape_mismatch': mismatched}


def initialize_weights(model, source, expected=None, allow_mismatch=(), excluded=('vector_field.task_embedding.weight',)):
    manifest = load_manifest(model.state_dict(), source, excluded=excluded)
    if set(manifest['shape_mismatch'])-set(allow_mismatch):
        raise ValueError(f"Partial checkpoint shape mismatch: {manifest['shape_mismatch']}")
    if expected is not None and manifest != expected:
        raise ValueError('Partial checkpoint key manifest differs from the reviewed initialization manifest')
    model.load_state_dict({k: source[k] for k in manifest['loaded']}, strict=False)
    manifest_with_hashes = dict(manifest)
    initialized = model.state_dict()
    manifest_with_hashes['loaded_tensor_sha256'] = state_digest({k: initialized[k] for k in manifest['loaded']})
    manifest_with_hashes['new_initialization_sha256'] = state_digest({k: initialized[k] for k in manifest['new']})
    model.initialization_manifest = manifest_with_hashes
    return manifest_with_hashes
