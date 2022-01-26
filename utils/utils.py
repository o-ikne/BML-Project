"""
Authors: @Omar Ikne & @Zakaria Boulkhir
26-01-2022
"""

import numpy as np

def sigmoid(z):
    """sigmoid function"""
    
    return 1 / (1 + np.exp(-z))


def epsilon_function(epsilon):
    """To handle both fix and variable step size"""
    
    if type(epsilon) == callable:
        return lambda t: epsilon(t)
    return epsilon
    
def generate_MG_samples(n_samples, theta, sigma, random_state=None):
    """Generate random samples from a Mixture of two Gaussians
    
    Parameters
    ----------
    n_samples: int,
        Number of samples to sample.
    
    theta: array like of shape (2,),
        The model parameters.
        
    sigma: float,
        The standard deviation of the generated samples.
        
            
    random_state: int, DEFAULT=None,
        Random seed for reproducibility.
        
    Returns
    -------
    gaussian_mix: array like of shape (n_samples,),
            The generated data.
    """
    
    ## random generator
    rng = np.random.default_rng(random_state)
    
    ## generated the mixture of Gaussians
    mixture_1 = .5 * rng.normal(loc=theta[0], scale=sigma, size=(n_samples, 2))
    mixture_2 = .5 * rng.normal(loc=theta[0] + theta[1], scale=sigma, size=(n_samples, 2))  
    gaussian_mix = mixture_1 + mixture_2
    
    return gaussian_mix
    
def generate_art_data(n_samples):
    """Generate artificial data for ICA experiment"""
    
    ## generate 3 channels normally distributed
    normal_samples = np.random.normal(size=(n_samples, n_samples, 3))
    
    ## generate 3 channels with high kurtosis distribution
    kurtosis_samples = stats.kurtosis(np.random.normal(size=(n_samples, n_samples, n_samples, 3)), fisher=True)

    ## put all together
    data = np.concatenate((normal_samples, kurtosis_samples), axis=-1)
    
    return data
