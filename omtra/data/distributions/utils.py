from scipy.ndimage import gaussian_filter

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
