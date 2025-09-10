"""This module contains access to empirical distributions from the plinder dataset."""

from omtra.utils import omtra_root
from omtra.data.distributions.utils import smooth_joint_distribution
import functools
import torch

import numpy as np
import pandas as pd
from pathlib import Path

def plinder_n_nodes_dist_raw() -> dict:
    plinder_dists_file = Path(omtra_root()) / "omtra_pipelines/plinder_dataset/plinder_filtered.parquet"

    # read dataframe
    df = pd.read_parquet(plinder_dists_file)

    # filter out npndes
    df = df[ df['ligand_type'] == 'ligand' ]

    # get the observed counts of (n_ligand_atoms, n_pharmacophores, n_protein_atoms)
    raw_cols = ['num_heavy_atoms', 'num_pharmacophores', 'num_pocket_atoms']
    observed = df[raw_cols].values.astype(int)
    lpp_unique, lpp_counts = np.unique(observed, axis=0, return_counts=True)

    # get the support (the set of unique values observed for dimension in the joint distribution)
    supports = {
        'n_ligand_atoms': np.arange(lpp_unique[:, 0].min(), lpp_unique[:, 0].max() + 1),
        'n_pharms': np.arange(lpp_unique[:, 1].min(), lpp_unique[:, 1].max() + 1),
        'n_protein_atoms': np.arange(lpp_unique[:, 2].min(), lpp_unique[:, 2].max() + 1),
    }
    var_order = ['n_ligand_atoms', 'n_pharms', 'n_protein_atoms']

    # convert counts to the full joint distribution p(n_ligand_atoms, n_pharmacophores, n_protein_atoms)
    p_lpp = np.zeros((len(supports['n_ligand_atoms']), len(supports['n_pharms']), len(supports['n_protein_atoms'])), dtype=float)
    for lpp_observed, lpp_count in zip(lpp_unique, lpp_counts):
        lig_idx = np.where(supports['n_ligand_atoms'] == lpp_observed[0])[0][0]
        pharm_idx = np.where(supports['n_pharms'] == lpp_observed[1])[0][0]
        prot_idx = np.where(supports['n_protein_atoms'] == lpp_observed[2])[0][0]
        p_lpp[lig_idx, pharm_idx, prot_idx] = lpp_count

    p_lpp = p_lpp / np.sum(p_lpp)

    output = {
        'density': p_lpp,
        'supports': supports,
        'var_order': var_order,
    }
    return output

@functools.lru_cache(1024 * 1024)
def plinder_n_nodes_dist_smoothed(sigma=(3.0, 3.0, 3.0), mode='constant'):
    """
    Get the empirical plinder distribution, and return a dictionary with a smoothed version 
    of the density along with its supports.
    
    This function calls the raw distribution generator `plinder_n_nodes_dist_raw` 
    and applies smoothing via a 3D Gaussian convolution.
    
    Parameters
    ----------
    sigma : tuple of floats, optional
        Standard deviations for the Gaussian kernel in each of the three dimensions.
    mode : str, optional
        The boundary mode for the convolution. Defaults to 'constant'.
    
    Returns
    -------
    output : dict
        A dictionary with keys:
          - 'density': The original (raw) joint distribution.
          - 'supports': The supports for each dimension.
          - 'smoothed_density': The smoothed joint distribution.
    """
    # Get the raw distribution
    raw_output = plinder_n_nodes_dist_raw()
    raw_density = raw_output['density']
    
    # Compute the smoothed distribution using the defined function
    smoothed_density = smooth_joint_distribution(raw_density, sigma=sigma, mode=mode)
    
    # Save the smoothed density in the output dictionary
    raw_output['smoothed_density'] = smoothed_density

    # Convert the raw density and supports to torch tensors
    for k in raw_output['supports']:
        raw_output['supports'][k] = torch.tensor(raw_output['supports'][k])
    raw_output['smoothed_density'] = torch.tensor(smoothed_density)
    raw_output['density'] = torch.tensor(raw_density)
    
    return raw_output

def sample_n_lig_atoms_plinder(n_prot_atoms: torch.Tensor = None, n_pharms: torch.Tensor = None, n_samples: int = None) -> torch.Tensor:

    joint_dist = plinder_n_nodes_dist_smoothed()
    var_order = joint_dist['var_order']
    supports = joint_dist['supports']
    p_joint = joint_dist['smoothed_density'] # has shape (n_ligand_atoms_support, n_pharms_support, n_protein_atoms_support)

    if n_prot_atoms is not None and n_pharms is not None and n_prot_atoms.shape != n_pharms.shape:
        raise ValueError("n_prot_atoms and n_pharms must have the same shape")
    
    n_nodes_supplied = [n_prot_atoms is not None, n_pharms is not None]
    if not any(n_nodes_supplied) and n_samples is None:
        raise ValueError("If n_prot_atoms and n_pharms are not provided, n_samples must be provided.")
    
    if n_samples is not None and any(n_nodes_supplied):
        raise ValueError("If n_samples is provided, n_prot_atoms and n_pharms must be None.")
    
    # n_prot_atoms is a tensor of shape (n_samples,)
    # n_pharms is a tensor of shape (n_samples,)
    
    if n_samples is None:
        if n_prot_atoms is not None:
            # find corresponding indicies for the number of protein atoms provided
            n_prot_atom_idxs = torch.searchsorted(supports['n_protein_atoms'], n_prot_atoms)
            if not torch.all(supports['n_protein_atoms'][n_prot_atom_idxs] == n_prot_atoms):
                raise ValueError("n_prot_atoms must be in the support of the distribution")

            p = p_joint[:, :, n_prot_atom_idxs] # has shape (n_ligand_atoms_support, n_pharms_support, n_samples)
            p = p.permute(2, 0, 1) # has shape (n_samples, n_ligand_atoms_support, n_pharms_support)
        else:
            # prot atoms is not specified so we marginalize over n_prot_atoms
            p = p_joint.sum(axis=2) # has shape (n_ligand_atoms_support, n_pharms_support)
            p = p.unsqueeze(0).expand(n_pharms.shape[0], -1, -1) # has shape (n_samples, n_ligand_atoms_support, n_pharms_support)


        if n_pharms is not None:
            n_pharms_idxs = torch.searchsorted(supports['n_pharms'], n_pharms)
            
            if not torch.all(supports['n_pharms'][n_pharms_idxs] == n_pharms):
                 raise ValueError("n_pharms must be in the support of the distribution")

            # vectorized masking
            mask_cols = torch.arange(p.shape[1]).unsqueeze(0) # (1, n_pharms_support)
            mask = mask_cols >= n_pharms_idxs.unsqueeze(1)  # (n_samples, n_pharms_support)
            mask = mask.unsqueeze(1).expand(-1, p.shape[1], -1)     # expand to match ligand atoms dimension: (n_samples, n_ligand_atoms_support, n_pharms_support)
            
            # marginalize by summing over valid n_pharms indices
            p = (p * mask).sum(dim=-1)  # (n_samples, n_ligand_atoms_support)

            # find corresponding indicies for the number of pharmacophores provided
            # n_pharms_idxs = torch.searchsorted(supports['n_pharms'], n_pharms)
            # if not torch.all(supports['n_pharms'][n_pharms_idxs] == n_pharms):
            #     raise ValueError("n_pharms must be in the support of the distribution")
            
            # p = p[sample_idxs, :, n_pharms_idxs] # has shape (n_samples, n_ligand_atoms_support)

        else:
            # n_pharms is not specified so we marginalize over n_pharms
            p = p.sum(axis=-1) # has shape (n_samples, n_ligand_atoms_support)

    else:
        p = p_joint.sum(dim=-1).sum(dim=-1) # has shape (n_ligand_atoms_support)
        p = p.unsqueeze(0).expand(n_samples, -1) # has shape (n_samples, n_ligand_atoms_support)

    n_ligand_atoms_idxs = torch.multinomial(p, num_samples=1).flatten() # has shape (n_samples,)
    n_ligand_atoms = supports['n_ligand_atoms'][n_ligand_atoms_idxs] # has shape (n_samples,)
    return n_ligand_atoms

def sample_n_pharms_plinder(n_lig_atoms: torch.Tensor = None, n_prot_atoms: torch.Tensor = None):

    arg_none = [n_lig_atoms is None, n_prot_atoms is None]
    if all(arg_none):
        raise ValueError("n_lig_atoms or n_prot_atoms must be provided")
    if not any(arg_none):
        assert n_lig_atoms.shape == n_prot_atoms.shape, "n_lig_atoms and n_prot_atoms must have the same shape"

    n_samples = n_lig_atoms.shape[0] if n_lig_atoms is not None else n_prot_atoms.shape[0]

    joint_dist = plinder_n_nodes_dist_smoothed()
    var_order = joint_dist['var_order']
    supports = joint_dist['supports']
    p_joint = joint_dist['smoothed_density'] # has shape (n_ligand_atoms_support, n_pharms_support, n_protein_atoms_support)

    if n_prot_atoms is not None:
        n_prot_atom_idxs = torch.searchsorted(supports['n_protein_atoms'], n_prot_atoms)
        if not torch.all(supports['n_protein_atoms'][n_prot_atom_idxs] == n_prot_atoms):
            raise ValueError("n_prot_atoms must be in the support of the distribution")
        p = p_joint[:, :, n_prot_atom_idxs] # has shape (n_ligand_atoms_support, n_pharms_support, n_samples)
        p = p.permute(2, 0, 1) # has shape (n_samples, n_ligand_atoms_support, n_pharms_support)
    else:
        p = p_joint.sum(dim=-1).expand(n_samples, -1, -1) # has shape (n_samples, n_ligand_atoms_support, n_pharms_support)

    if n_lig_atoms is not None:
        n_lig_atoms_idxs = torch.searchsorted(supports['n_ligand_atoms'], n_lig_atoms)
        if not torch.all(supports['n_ligand_atoms'][n_lig_atoms_idxs] == n_lig_atoms):
            raise ValueError("n_lig_atoms must be in the support of the distribution")
        sample_idxs = torch.arange(p.shape[0])
        p = p[sample_idxs, n_lig_atoms_idxs, :] # has shape (n_samples, n_pharms_support)
    else:
        p = p.sum(axis=1) # has shape (n_samples, n_pharms_support)
    
    n_pharms_idxs = torch.multinomial(p, num_samples=1).flatten() # has shape (n_samples,)
    n_pharms = supports['n_pharms'][n_pharms_idxs] # has shape (n_samples,)
    return n_pharms