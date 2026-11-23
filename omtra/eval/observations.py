"""Persist model-independent conditioning for paired evaluation across checkpoints."""
import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import torch
from omtra.models.conditioning import prepare_coordinate_condition, prepare_partial_condition


def reference_digest(g):
    digest=hashlib.sha256()
    for kind,types in [('node',g.ntypes),('edge',g.etypes)]:
        for entity in sorted(types):
            data=(g.nodes[entity] if kind=='node' else g.edges[entity]).data
            for key in sorted(k for k in data if k.endswith('_1_true') and 'mask' not in k):
                tensor=data[key].detach().cpu().contiguous()
                digest.update(f'{kind}/{entity}/{key}/{tuple(tensor.shape)}/{tensor.dtype}'.encode())
                digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def prepare_shared_observation(g,task,center,cache_dir,key,contract):
    """Exclusive creation; a competing writer fails and can retry reading the completed file.

    Cache contains conditioning and crop/reference hashes, never generated results or
    requested stereo labels. Identity provenance remains an independent caller contract.
    """
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    path=cache_dir/(hashlib.sha256(str(key).encode()).hexdigest()+'.npz')
    metadata={'version':1,'key':str(key),'task':task.name,'contract':contract,'reference_sha256':reference_digest(g)}
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:
            if json.loads(str(saved['metadata']))!=metadata:raise ValueError(f'Shared observation input/contract mismatch: {path}')
            for name in saved.files:
                if name in ('metadata','center'):continue
                kind,entity,field=name.split('/')
                data=(g.nodes[entity] if kind=='node' else g.edges[entity]).data
                data[field]=torch.as_tensor(saved[name].copy(),device=g.device)
            center=torch.as_tensor(saved['center'].copy(),device=g.device) if saved['center'].size else None
    else:
        if 'protein_structure' in task.groups_fixed:
            prepare_coordinate_condition(g,'prot_atom',contract['prot_input_noise_max_std'],noise_mode=contract['prot_input_noise_mode'])
        if 'pharmacophore' in task.groups_fixed:
            prepare_coordinate_condition(g,'pharm',contract['pharm_input_noise_std'])
        prepare_partial_condition(g,task,contract['fixed_coord_max_std'],contract['fixed_token_max_prob'])
        arrays={'metadata':np.asarray(json.dumps(metadata,sort_keys=True)),
                'center':np.asarray([]) if center is None else center.detach().cpu().numpy()}
        for kind,types in [('node',g.ntypes),('edge',g.etypes)]:
            for entity in types:
                data=(g.nodes[entity] if kind=='node' else g.edges[entity]).data
                for field,value in data.items():
                    if '_cond' in field or field in ('atom_mask_1_true','edge_mask_1_true'):
                        arrays[f'{kind}/{entity}/{field}']=value.detach().cpu().numpy()
        # Publish only a complete file. Atomic hard-link creation preserves the
        # first observation when paired model workers reach the cache together.
        with tempfile.NamedTemporaryFile(dir=cache_dir,prefix='.observation_',delete=False) as stream:
            temporary=Path(stream.name)
            np.savez_compressed(stream,**arrays)
        try:
            try:os.link(temporary,path)
            except FileExistsError:
                return prepare_shared_observation(g,task,center,cache_dir,key,contract)
        finally:temporary.unlink()
    return center,path
