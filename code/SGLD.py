"""
Authors: @Omar Ikne & @Zakaria Boulkhir
26-01-2022
"""

## libraries
import numpy as np
import sys
sys.path.append('../utils/')
from utils import *

##

def SGLD(X, log_prior, log_likelihood, batch_size, epsilon, n_iter, y=None, random_state=None):
    
    """Stochastic Gradient Langevin Dynamics
    
    Parameters
    ----------
    X : array like of shape (N, d),
        The training samples.
    
    log_prior: callable,
        The log prior of the model.
        
    log_likelihood: callable,
        The log likelihhod of the model.
        
    batch_size: int,
        The size of the mini-batch.
        
    epsilon: float or callable,
        The step size.
        
    n_iter: int,
        Number of iterations to perform.
        
    theta_zero: array like of shape (d, 1)
        Instantiating of parameters.
        
    random_state: int, DEFAULT=None,
        Random seed for reproducibility.
    
    Returns
    -------
        Thetas: array like of shape (n_iter, d),
            The history update of the parameters.
            
        noise: array-like of shape (n_inter),
            The added noise.
    """
    
    ## number of training samples
    N, d = X.shape
    samples_per_batch = N // batch_size
    
    ## instantiate parameters
    thetas, noises = [], []
    theta = np.zeros(shape=(d,))
    
    ## random generator
    rng = np.random.default_rng(random_state)
    
    ## to compute step zie
    epsilon = epsilon_function(epsilon)
    
    ## training loop
    for t in range(n_iter):
        
        ## pick a random mini batch
        rdm_idx = np.random.choice(N, size=batch_size)
        X_batch = X[rdm_idx]
        
        ## compute epsilon(t)
        eps_t = epsilon(t)
        
        ## compute log prior & log likelihood
        y_batch = None
        if y is not None:
            y_batch = y[rdm_idx]
            
        log_p = log_prior(theta)
        log_lik = log_likelihood(X_batch, theta, y_batch)

        ## Gaussian noise to add
        noise = rng.normal(loc=0, scale=eps_t)
        
        ## parameters update
        theta = theta + .5 * eps_t * (log_p + (N * log_lik)) + noise
        thetas.append(theta)
        noises.append(noise)
    
    return thetas, noises
