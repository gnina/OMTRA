from scipy.ndimage import gaussian_filter
import torch
import math

def smooth_joint_distribution(density, sigma=(1.0, 1.0, 1.0), mode='constant'):
    """
    Smooth a 3D joint distribution by convolving with a 3D Gaussian kernel.
    
    Parameters
    ----------
    density : np.ndarray
        A 3-dimensional array representing the joint distribution. It is assumed
        to be normalized (i.e. sum equals 1).
    sigma : tuple of floats, optional
        The standard deviations of the Gaussian kernel along each dimension.
        Default is (1.0, 1.0, 1.0); you may adjust these based on your data.
    mode : str, optional
        The mode parameter passed to the gaussian_filter function to handle boundaries.
        'constant' (the default) pads with 0, but you might try 'reflect' or other options
        if needed.
    
    Returns
    -------
    smoothed_density : np.ndarray
        The smoothed joint distribution, normalized so that its sum is 1.
    """
    # Apply Gaussian convolution to smooth the distribution
    smoothed_density = gaussian_filter(density, sigma=sigma, mode=mode)
    
    # Renormalize to ensure the distribution sums to 1
    smoothed_density /= smoothed_density.sum()
    
    return smoothed_density
def residue_sinusoidal_encoding(residue_idx: torch.LongTensor,
                                 d_model: int):
    """
    residue_idx: (N_atoms,) each entry in [0..R-1]
    returns:     (N_atoms, d_model) same encoding for same residue_idx
    """
    device = residue_idx.device
    N = residue_idx.size(0)
    # (d_model/2,) — even dims for sin, odd for cos
    inv_freq = torch.exp(
        torch.arange(0, d_model, 2, device=device).float() *
        -(math.log(10000.0) / d_model)
    )  # (d_model/2,)
    # (N, d_model/2)
    angles = residue_idx.float().unsqueeze(1) * inv_freq.unsqueeze(0)
    pe = torch.zeros(N, d_model, device=device)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    
    return pe