"""Immutable training-membership/crop-matched empirical ligand-size priors."""
from functools import lru_cache
from pathlib import Path
import json
import hashlib
import numpy as np
import torch


def write_prior(path, counts, membership, provenance):
    """counts columns: real ligand atoms, retained protein atoms, total pharm points."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 2 or counts.shape[1] != 3 or len(counts) != len(membership):
        raise ValueError('One count triple is required for each training-member identity')
    if len(counts) == 0 or (counts[:, :2] < 1).any() or (counts[:, 2] < 0).any():
        raise ValueError('Invalid size-prior counts')
    if len(set(map(tuple, membership))) != len(membership):
        raise ValueError('Duplicate training membership')
    manifest = dict(provenance, version=1, n_records=len(counts), split='train',
        membership_sha256=hashlib.sha256(json.dumps(membership,sort_keys=True).encode()).hexdigest(),
        counts_sha256=hashlib.sha256(counts.tobytes()).hexdigest(),
        policy={'protein_bandwidth_atoms': 32., 'ligand_bandwidth_atoms': 3.,
                'pharmacophore_condition': 'total_count_at_least_supplied',
                'unsupported_protein': 'nearest_observed_count',
                'zero_mass': 'marginal_training_distribution'})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, counts=counts, membership=np.asarray(membership,dtype=str),
                            manifest=np.asarray(json.dumps(manifest,sort_keys=True)))
    return manifest


@lru_cache(maxsize=8)
def load_prior(path):
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as data:
        counts, membership = data['counts'].copy(), data['membership'].tolist()
        manifest = json.loads(str(data['manifest']))
    if manifest.get('version') != 1 or manifest.get('split') != 'train' or not manifest.get('complete', False):
        raise ValueError('Size prior must certify complete, training-only membership')
    if hashlib.sha256(counts.tobytes()).hexdigest() != manifest['counts_sha256']:
        raise ValueError('Size-prior count hash mismatch')
    if hashlib.sha256(json.dumps(membership,sort_keys=True).encode()).hexdigest() != manifest['membership_sha256']:
        raise ValueError('Size-prior membership hash mismatch')
    if len(counts) != manifest['n_records'] or len(membership) != len(counts):
        raise ValueError('Size-prior membership/count length mismatch')
    return counts, manifest


def sample_sizes(path, n_prot_atoms=None, n_pharms=None, n_samples=None, generator=None, return_audit=False):
    counts, manifest = load_prior(str(Path(path).resolve()))
    if n_samples is not None and (n_prot_atoms is not None or n_pharms is not None):
        raise ValueError('Specify conditions OR unconditional sample count')
    if n_samples is None:
        sizes = [len(x) for x in (n_prot_atoms,n_pharms) if x is not None]
        if not sizes or len(set(sizes)) != 1:
            raise ValueError('Size-prior conditions must have equal, nonempty lengths')
        n_samples = sizes[0]
    lo, hi = max(4,int(counts[:,0].min())), max(4,int(counts[:,0].max()))
    support = torch.arange(lo,hi+1,dtype=torch.float64)
    ligand = torch.as_tensor(counts[:,0],dtype=torch.float64)
    prot = torch.as_tensor(counts[:,1],dtype=torch.float64)
    pharm = torch.as_tensor(counts[:,2],dtype=torch.long)
    # Aggregate weighted membership to a histogram before smoothing; avoid a dense joint cube.
    kernel = torch.exp(-.5*((support[:,None]-support[None,:])/3.)**2)
    audit, samples = [], []
    # Replicates usually share the same condition. Compute each histogram once.
    distributions = {}
    for i in range(n_samples):
        condition = (None if n_prot_atoms is None else float(n_prot_atoms[i]),
                     None if n_pharms is None else float(n_pharms[i]))
        if condition in distributions:
            probabilities,row=distributions[condition]
            samples.append(lo+int(torch.multinomial(probabilities,1,generator=generator)))
            audit.append(dict(row));continue
        weights = torch.ones(len(counts),dtype=torch.float64)
        row = {'protein_fallback':False,'marginal_fallback':False}
        if n_prot_atoms is not None:
            value = float(n_prot_atoms[i])
            if not np.isfinite(value) or value < 0:
                raise ValueError('Protein count must be finite and nonnegative')
            value_clipped = min(max(value,float(prot.min())),float(prot.max()))
            row['protein_fallback'] = value != value_clipped
            weights *= torch.exp(-.5*((prot-value_clipped)/32.)**2)
        if n_pharms is not None:
            raw=float(n_pharms[i])
            if not np.isfinite(raw) or raw < 0 or not raw.is_integer():
                raise ValueError('Pharmacophore count must be a finite nonnegative integer')
            value = int(raw)
            weights *= (pharm >= value)
        if weights.sum() <= 1e-300:
            weights.fill_(1.)
            row['marginal_fallback'] = True
        hist = torch.zeros(len(support),dtype=torch.float64)
        hist.scatter_add_(0,(ligand.long().clamp(lo,hi)-lo),weights)
        probabilities = kernel @ hist
        if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
            raise ValueError('Nonfinite or zero-mass size prior')
        distributions[condition]=(probabilities,dict(row))
        samples.append(lo+int(torch.multinomial(probabilities,1,generator=generator)))
        audit.append(row)
    result = torch.tensor(samples,dtype=torch.long)
    return (result,audit) if return_audit else result
