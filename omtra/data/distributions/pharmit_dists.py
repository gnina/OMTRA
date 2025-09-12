import numpy as np
import torch
from omtra.utils import omtra_root
from omtra.data.distributions.utils import smooth_joint_distribution
import functools
from pathlib import Path

def pharmit_n_nodes_dist_raw(dists_file: str = None):
    if dists_file is None:
        dists_file = Path(omtra_root()) / "omtra_pipelines/pharmit_dataset/train_dists.npz"

    data = np.load(dists_file)

    supports = {
        'n_ligand_atoms': data['p_ap_atoms_space'],
        'n_pharms': data['p_ap_pharms_space'],
    }

    # convert supports to all continuous integers in the range
    for k in supports:
        supports[k] = np.arange(supports[k].min(), supports[k].max() + 1)

    outputs = {
        'supports': supports,
    }
    outputs['density'] = data['p_ap']
    outputs['var_order'] = ['n_ligand_atoms', 'n_pharms']
    return outputs

@functools.lru_cache(1024 * 1024)
def pharmit_n_nodes_dist_smoothed(sigma=(3.0, 3.0), mode='constant'):
    """
    Get the empirical Pharmit distribution, and return a dictionary with a smoothed version 
    of the density along with its supports.
    
    This function calls the raw distribution generator `pharmit_n_nodes_dist_raw` 
    and applies smoothing via a 2D Gaussian convolution.
    
    Parameters
    ----------
    sigma : tuple of float
        The standard deviation for the Gaussian kernel in each dimension.
    mode : str
        The mode parameter for the Gaussian convolution. Default is 'constant'.
    
    Returns
    -------
    dict
        A dictionary containing the smoothed density, supports, and variable order.
    """
    raw = pharmit_n_nodes_dist_raw()
    smoothed_density = smooth_joint_distribution(raw['density'], sigma=sigma, mode=mode)
    raw['smoothed_density'] = smoothed_density

    for k in raw['supports']:
        raw['supports'][k] = torch.tensor(raw['supports'][k])
    raw['smoothed_density'] = torch.tensor(smoothed_density)
    raw['density'] = torch.tensor(raw['density'])

    return raw

def sample_n_lig_atoms_pharmit(n_pharms: torch.Tensor = None, n_samples: int = None) -> torch.Tensor:
    inputs_none = [n_pharms is None, n_samples is None]
    if all(inputs_none):
        raise ValueError("n_pharms or n_samples must be provided")
    elif sum(inputs_none) == 2:
        raise ValueError("specify either n_pharms or n_samples, not both")
    
    joint_dist = pharmit_n_nodes_dist_smoothed()
    p_joint = joint_dist['smoothed_density'] # has shape (n_ligand_atoms_support, n_pharms_support)
    supports = joint_dist['supports']

    if n_pharms is not None:        
        n_pharms_idxs = torch.searchsorted(supports['n_pharms'], n_pharms)

        if not torch.all(supports['n_pharms'][n_pharms_idxs] == n_pharms):
             raise ValueError("n_pharms must be in the support")

        # vectorized masking
        mask_cols = torch.arange(p_joint.shape[1]).unsqueeze(0) # (1, n_pharms_support)
        mask = mask_cols >= n_pharms_idxs.unsqueeze(1)           # (n_samples, n_pharms_support)        
        mask = mask.unsqueeze(1).expand(-1, p_joint.shape[0], -1)   # expand to match ligand atoms dimension: (n_samples, n_ligand_atoms_support, n_pharms_support)
        
        p = p_joint.unsqueeze(0).expand(n_pharms.shape[0], -1, -1)   # expand p_joint to batch dimension: (n_samples, n_ligand_atoms_support, n_pharms_support)
        # marginalize by summing over valid n_pharms indices
        p = (p * mask).sum(dim=-1)  # (n_samples, n_ligand_atoms_support)

        # n_pharms_idxs = torch.searchsorted(supports['n_pharms'], n_pharms)
        # if not torch.all(supports['n_pharms'][n_pharms_idxs] == n_pharms):
        #     raise ValueError("n_pharms must be in the support")
        # p = p_joint[:, n_pharms_idxs]
        # p = p.permute(1, 0) # has shape (n_samples, n_ligand_atoms_support)
    else:
        p = p_joint.sum(dim=-1)
        p = p.expand(n_samples, -1) 

    n_lig_atoms_idxs = torch.multinomial(p, num_samples=1).flatten()
    n_lig_atoms = supports['n_ligand_atoms'][n_lig_atoms_idxs]
    return n_lig_atoms

def sample_n_pharms_pharmit(n_lig_atoms: torch.Tensor):
    # implicit n_lig_atoms cannot be done because we never do unconditional sampling of pharmacophores

    # get joint density and convert to tensor
    joint_dist = pharmit_n_nodes_dist_smoothed()
    p_joint = joint_dist['smoothed_density'] # has shape (n_ligand_atoms_support, n_pharms_support)
    supports = joint_dist['supports']

    n_lig_atoms_idxs = torch.searchsorted(supports['n_ligand_atoms'], n_lig_atoms)
    if not torch.all(supports['n_ligand_atoms'][n_lig_atoms_idxs] == n_lig_atoms):
        raise ValueError("n_lig_atoms must be in the support")
    
    p = p_joint[n_lig_atoms_idxs] # has shape (n_samples, n_pharms_support)
    n_pharms_idxs = torch.multinomial(p, num_samples=1).flatten()
    n_pharms = supports['n_pharms'][n_pharms_idxs]
    return n_pharms