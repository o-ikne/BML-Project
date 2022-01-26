"""
Authors: @Omar Ikne & @Zakaria Boulkhir
26-01-2022
"""

## libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('../utils/')
from utils import *
from SGLD import SGLD
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, confusion_matrix, classification_report

## set plots text font size & style
sns.set(font_scale=1.2, style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

##


def log_likelihood_LR(X, beta, y):
    """return the Log Likelihood for Logistic regression
    
    Parameters
    ----------
    X: array-like of shape (n_samples, d),
        The data points.
        
    y: array-like of shape (n_samples, ),
        The target values.
        
    beta: array-like of shape(d,),
        The parameters.
        
    Returns
    -------
    log_lik: float,
        The Log Likelihood.
    """
    
    ## compute scores B^T.X
    scores = X.dot(beta)

    ## compute log likelihood
    log_lik = np.sum(y * scores - np.log(1 + np.exp(scores)))
    
    return log_lik

def grad_log_likelihood_LR(X, beta, y):
    grad_log_lik = []
    for j in range(beta.shape[0]):
        grad = (y - X.dot(beta)) * X[:, j]
        grad_log_lik.append(grad)
    grad_log_lik = np.asarray(grad_log_lik)

    return grad_log_lik[0, :]

def grad_prior_LR(beta):
    """return the gradient of the prior"""
    
    ## grad prior = -sign(beta)
    grad_p = -np.sign(beta)
    
    return grad_p
    
    
if __name__ == "__main__":

	## upload "a9a" dataset
	X_a9a, y_a9a = fetch_openml(name='a9a', version=1, return_X_y=True, as_frame=False)

	## number of samples to pick
	n_samples = 10_000
	X_a9a = X_a9a[:n_samples]
	y_a9a = y_a9a[:n_samples]
	N, d = X_a9a.shape

	print("="*20, "LOGISTIC REGRESSION", "="*20)
	print(f'> Number of samples : {N}')
	print(f'> Number of features: {d}')

	## split data
	X_train, X_test, y_train, y_test = train_test_split(X_a9a, y_a9a, test_size=.2, random_state=0)

	print(f"> Number of training samples: {X_train.shape[0]}")
	print(f"> Number of testing samples : {X_test.shape[0]}")
	
	## parameters
	n_iter = 100
	batch_size = 1
	
	## define step size function epislon
	gamma = .55
	b = n_iter / (np.exp(-np.log(10e-3) / gamma) - 1)
	a = 0.01 / b ** (-gamma)
	epsilon = lambda t: a * (b + t) ** (-gamma)

	## fit the model
	thetas_LR, noises_LR = SGLD(X_train, grad_prior_LR, grad_log_likelihood_LR, batch_size, epsilon, n_iter, y_train, 0)
	
	## make predictions
	y_pred = np.sign(X_test.dot(thetas_LR[-1]))

	## confusion matrix
	conf_mat = confusion_matrix(y_test, y_pred, labels=(-1, 1))

	fig, ax = plt.subplots(figsize=(5, 5))
	sns.heatmap(conf_mat, ax=ax, annot=True, square=True, fmt='.4g')
	ax.set_title('confusion matrix')
	plt.show()
	
	## compute error
	error = (y_pred != y_test).mean()

	print(f'> error: {error}')
