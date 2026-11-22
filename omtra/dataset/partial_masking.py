"""Partial reveals shared by PLINDER and Pharmit, with no noise on labels."""
import numpy as np


def sample_partial_mask(fragment_ids, mode='whole_fragments', bernoulli_p=None):
    if mode not in {'whole_fragments', 'fragment_bernoulli_atom', 'bernoulli_atoms', 'mixed'}:
        raise ValueError(f'Unknown partial_mode: {mode}')
    if bernoulli_p is not None and not 0 <= bernoulli_p <= 1:
        raise ValueError('partial_bernoulli_p must be in [0, 1]')
    frags = np.asarray(fragment_ids)
    # Locked 50/50 mixture of the two previously trained masking modes.
    if mode == 'mixed':
        mode = 'whole_fragments' if np.random.rand() < 0.5 else 'fragment_bernoulli_atom'
    if bernoulli_p is None:
        bernoulli_p = np.random.rand()
    if mode == 'bernoulli_atoms':
        return (np.random.rand(len(frags)) < bernoulli_p).astype(np.int64)
    ids = np.unique(frags)
    selected = np.random.choice(ids, np.random.randint(1, len(ids)), replace=False) if len(ids) > 1 else []
    if len(ids) <= 1 and mode == 'fragment_bernoulli_atom':
        return (np.random.rand(len(frags)) < bernoulli_p).astype(np.int64)
    mask = np.isin(frags, selected)
    if mode == 'fragment_bernoulli_atom':
        mask &= np.random.rand(len(frags)) >= bernoulli_p
    return mask.astype(np.int64)
