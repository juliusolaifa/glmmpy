import numpy as np 
from scipy.stats import multivariate_normal


def log_mult_norm_prior(theta_n, Sigma):
    """
    computes the log of the multivariate normal density function


    Parameters:
    -----------
    theta_n: array-like
        A vector of values (the parameter) for which the log density is to be computed.

    Sigma: array-like or 2D array
        The variance-covariance matrix of the multivariate normal distribution.

    Returns:
    --------
    log_likelihood: float
        The log of the multivariate normal probability evaluated at theta_n

    Example:
    --------
    >>> theta_n = np.array([1, 2])
    >>> Sigma = np.array([[2, 1], [1, 2]])
    >>> log_multivariate_normal(theta_n, Sigma)
    """

    mean = np.zeros(len(theta_n)) #mean vector is 0
    log_likelihood = multivariate_normal(theta_n, mean=mean, cov=Sigma)
    return log_likelihood