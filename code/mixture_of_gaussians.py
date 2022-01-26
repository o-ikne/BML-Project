"""
Authors: @Omar Ikne & @Zakaria Boulkhir
26-01-2022
"""

## libraries
import numpy as np
import matplotlib.pyplot as plt
from SGLD import SGLD
import seaborn as sns
from scipy import stats
import sys
sys.path.append('../utils/')
from utils import *

#-----------< Setting >------------#
## set plots text font size & style
sns.set(font_scale=1.2, style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

##

def log_prior_MG(theta):
    """returns the log prior of Mixture of Gaussians.
    
    Parameters
    ----------
    theta: array-like of shape (d,)
        The parameters.
        
    Returns
    -------
    log_p: float,
        The log prior of the given parameters.
    """
    
    ## compute log prior
    first_term = stats.norm.logpdf(theta[0], loc=0, scale=sigma[0])
    second_term = stats.norm.logpdf(theta[1], loc=0, scale=sigma[1])
    log_p = first_term + second_term

    return log_p

def log_likelihood_MG(x, theta, y=None):
    """returns the log likelihood of a Mixture of Gaussians
    
    Parameters
    ----------
    x: array-like of shape (n_samples, d),
        The data points.
        
    theta: array-like of shape (d,)
        The parameters.
        
    Returns
    -------
    log_lik: float,
        The log likelihood of the given data and parameters.
    """
     
    ## compute log likelihood
    first_term  = .5 * stats.norm.pdf(x, loc=theta[0], scale=sigma_x)
    second_term = .5 * stats.norm.pdf(x, loc=theta[0] + theta[1], scale=sigma_x)
    lik = first_term + second_term
    
    ## to avoid zero values in log
    eps = 10e-8
    log_lik = np.mean(np.log(lik + eps), axis=0)
    
    return log_lik 
    
    
if __name__ == "__main__":

	## parameters
	sigma_x = np.sqrt(2)
	sigma = [np.sqrt(10), 1]
	n_samples = 100
	theta = [0, 1]
	n_iter = 100
	batch_size = 1
	
	print("="*20, 'Mixture of Gaussians', "="*20)
	print(f'> Number of samples: {n_samples}')
	print(f'> batch size: {batch_size}')
	print(f'> Number of iterations: {n_iter}')

	## generate mixture of Gaussians
	X = generate_MG_samples(n_samples=n_samples, theta=theta, sigma=np.sqrt(2), random_state=0)

	## define step size function epislon
	gamma = .55
	b = n_iter / (np.exp(-np.log(10e-3) / gamma) - 1)
	a = 0.01 / b ** (-gamma)
	epsilon = lambda t: a * (b + t) ** (-gamma)

	## run SGLD
	thetas_1, noises_1 = SGLD(X, log_prior_MG, log_likelihood_MG, batch_size, epsilon, n_iter, random_state=0)
	
	## compute parameters noise
	grad_1 = np.abs(np.gradient([x for x, _ in thetas_1])) / n_iter
	grad_2 = np.abs(np.gradient([x for _, x in thetas_1])) / n_iter
	
	## create a subplots
	fig, ax = plt.subplots(figsize=(14, 5))

	## display injected noise
	ax.plot(grad_1, label=r'$\nabla\theta_1$ noise')
	ax.plot(grad_2, label=r'$\nabla\theta_2$ noise')
	ax.plot(noises_1, label='injected noise')
	ax.plot([epsilon(t) for t in range(1, n_iter+1)], label=r'eps($\epsilon$)')
	ax.set_xlabel('iterations')
	ax.set_ylabel('noise')
	ax.legend(bbox_to_anchor=(1.01, 1))
	plt.show()
