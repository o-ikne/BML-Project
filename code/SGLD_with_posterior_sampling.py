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
def posterior_sampling_phase(eps_t, N, batch_size, M, theta_t, threshold):
    """To check whether the SGLD reached the posterior sampling phase
    
    Parameters
    ----------
    eps_t : float,
        The step size at iteration t.

    N: int,
        Number of samples.

    batch_size: int,
        The mini batch size.

    M: array like,
        The preconditioning matrix.

    theta_t: array like of shape (d, 1),
        The parameters at iteration t.

    threshold: float,
        The threshold for posterior sampling phase.
            
    Returns
    -------
    posterior_phase: bool,
        True if the algorithm has reached the posterior phase, otherwise False.
    
    """
    
    ## compute empirical variance of the parameters
    emp_var = np.var(theta_t)
    
    ## M ^ (1/2)
    pow_M = np.power(M, .5)
    
    ## get eignevalues
    eigenvs, _ = np.linalg.eig(np.dot(pow_M, emp_var).dot(pow_M))
    
    ## compute alpha
    alpha = (1 / 4 * batch_size) * eps_t * N ** 2 * np.max(eigenvs)
    
    ## check if posterior phase is rechead
    in_posterior_phase = (alpha < threshold)
    
    return in_posterior_phase      
    
    
 def SGLD_with_posterior_sampling(X, log_prior, log_likelihood, batch_size, epsilon,
                                 n_iter, thereshold, M, random_state=None):
    
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
        

    threshold: float,
        The threshold for posterior sampling phase.
        
    M: array like,
        The preconditioning matrix.
            
    random_state: int, DEFAULT=None,
        Random seed for reproducibility.
    
    Returns
    -------
        Thetas: array like of shape (n_iter, d),
            The history update of the parameters.
    """
    
    ## number of training samples
    N, d = X.shape
    samples_per_batch = N // batch_size
    
    ## instantiate parameters
    theta = np.zeros(shape=(d,))
    
    ## random generator
    rng = np.random.default_rng(random_state)
    
    ## while loop
    in_posterior_phase = False
    while not in_posterior_phase:
        
        ## pick a random mini batch
        mini_batch = X[rng.choice(N, size=samples_per_batch)]
        
        ## compute epsilon(t)
        eps_t = epsilon(t)
        
        ## compute log prior & log likelihood
        log_p = log_prior(theta)
        log_lik = log_likelihood(mini_batch, theta)
        
        ## Gaussian noise to add
        noise = rng.normal(loc=0, scale=eps_t)
        
        ## parameters update
        theta = theta + .5 * eps_t * (log_p + (N / batch_size) * np.sum(log_lik)) + noise
        
        ## posterior sampling phase
        in_posterior_phase = posterior_sampling_phase(eps_t, N, batch_size, M, theta, threshold)

    ## start sampling
    
    
    return thetas 
