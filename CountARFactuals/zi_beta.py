from scipy.stats import beta
from scipy.optimize import minimize
import numpy as np

def log_likelihood(params, data):
    pi, a, b = params
    if pi < 0 or pi > 1 or a <= 0 or b <= 0:
        return np.inf  # Penalize invalid parameter values to ensure valid bounds
    
    log_likelihoods = np.where(data == 0, 
                                np.log(pi), 
                                np.log(1 - pi) + beta.logpdf(data, a, b))
    return np.sum(log_likelihoods)

def neg_log_likelihood(params, data):
    return -log_likelihood(params, data)

def fit(data, initial_params=[0.5, 2, 2], method='L-BFGS-B'):
    """Fit the model to the given data using maximum likelihood estimation."""
    # Constrain parameters: pi between 0 and 1, a and b positive
    bounds = [(0, 1), (1e-5, None), (1e-5, None)]
    result = minimize(neg_log_likelihood, initial_params, args=(data,), 
                        method=method, bounds=bounds)

    # Store fitted parameters if successful
    if result.success:
        print(f"Model fitted successfully: pi = {result.x[0]:.4f}, a = {result.x[1]:.4f}, b = {result.x[2]:.4f}")
        return result.x
    else:
        raise RuntimeError("Model fitting failed. Try different initial parameters.")

def pdf(x, pi_hat, a_hat, b_hat):
    return pi_hat * (x == 0) + (1 - pi_hat) * beta.pdf(x, a_hat, b_hat)

def cdf(x, pi_hat, a_hat, b_hat):
    # CDF combines the mass at zero and the cumulative beta portion
    cdf_values = np.where(x == 0, pi_hat, pi_hat + (1 - pi_hat) * beta.cdf(x, a_hat, b_hat))
    return cdf_values

def sample(size, pi_hat, a_hat, b_hat):
    """Generate samples from the fitted zero-inflated beta distribution."""
    # First determine how many of the samples should be zero
    num_zeros = np.random.binomial(size, pi_hat)
    num_beta = size - num_zeros

    # Generate zero-inflated samples
    samples = np.zeros(size)
    beta_samples = beta.rvs(a_hat, b_hat, size=num_beta)
    samples[num_zeros:] = beta_samples

    return samples