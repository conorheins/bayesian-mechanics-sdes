import jax.numpy as jnp
from jax.numpy.linalg import eigvals, det, inv
from jax import random, jacfwd

import numpy as np

global_key = random.PRNGKey(50) # default used if no key is passed into `initialize_2d_OU` function

def initialize_2d_OU(rng_key = None):
    """
    Initialize the relevant variables of a 2-way OU process with an optional random seed
    and return them in parameter dictionaries
    Arguments:
    ==========
    `rng_key` [str or jax.random PRNGkey]:
    Returns:
    =========
    `flow_parameters`  [Dict]:
    `stationary_stats` [Dict]: 
    """
    
    n_var = 2

    if rng_key is None:
        rng_key = global_key

    Pi = np.array(random.normal(rng_key, shape=(n_var,n_var))) #random precision matrix
    _, next_key = random.split(rng_key)

    # enforce symmetric
    Pi = (Pi + Pi.T) / 2.0

    # Ensure Pi is positive definite
    while (np.linalg.eigvals(Pi) <= 0).any():
        Pi -= 2 * np.linalg.eigvals(Pi).min() * np.eye(n_var)
    
    Pi = jnp.array(np.round(Pi,1)) # convert back to JAX array at the end

    # arbitrary volatility matrix
    # sigma = jnp.round(jnp.diag(random.normal(next_key, shape=(n_var,))),1) # arbitrary diagonal volatility matrix
    # sigma = jnp.round(random.normal(next_key, shape=(n_var,n_var)),1)
    sigma = jnp.eye(n_var)
    _, next_key = random.split(next_key)

    #diffusion tensor
    D = sigma @ sigma.T / 2  # diffusion tensor

    # see whether noise is degenerate or not
    print(f'det sigma = {det(sigma)}')

    Q = random.normal(next_key, shape=(n_var,n_var))  # arbitrary solenoidal flow
    _, next_key = random.split(next_key)

    Q = jnp.round(5. * (Q - Q.T), 1) # ensure anti-symmetry

    # Compute the stationary covariance
    S = inv(Pi)

    # Drift matrix
    B = (D + Q) @ Pi  # drift matrix

    if (eigvals(B) <= -1e-5).any():
        print(eigvals(B))
        raise TypeError("Drift should have non-negative spectrum")
        
    # 1) We check it solves the Sylvester equation: BS + SB.T = 2D
    assert jnp.allclose(B @ S + S @ B.T, 2 * D, atol = 1e-5), "Sylvester equation not solved!"

    # 2) we check that there are no numerical errors due to ill conditioning
    assert jnp.allclose(inv(S), Pi, atol = 1e-5), "Precision and inverse covariance are different"

    # We check that the stationary covariance is indeed positive definite
    if (eigvals(S) <= 0).any():
        print(eigvals(S))
        raise TypeError("Stationary covariance not positive definite")

    flow_parameters = {'B': B, 'D': D, 'sigma': sigma, 'Q': Q}

    stationary_stats = {'Pi': Pi, 'S': S}

    return flow_parameters, stationary_stats