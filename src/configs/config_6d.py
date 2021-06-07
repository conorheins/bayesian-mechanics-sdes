import jax.numpy as jnp
from jax.numpy.linalg import inv, pinv, det, eigvals
from jax import random
from utilities.analysis_tools import compute_kernel_condition

import numpy as np

global_key = random.PRNGKey(5)

def initialize_6d_OU(rng_key = None):

    '''
    Setting up the steady-state
    '''

    n_var = 6                 # dimensionality of state-space

    eta_dim = [0, 1]          # dimensions of external states
    s_dim =   [2]             # dimensions of sensory states
    a_dim =   [3]             # indices of active states dimensions
    mu_dim =  [4, 5]          # indices of internal states dimensions
    b_dim =   s_dim + a_dim   # indices of blanket state dimensions (union of sensory and active states)
    pi_dim =   b_dim + mu_dim  # indices of particular states dimensions (union of blanket and internal states)

    u_dim = list(set(range(n_var)) - set(s_dim))      # indices of other/un-perturbed dimensions (complement of sensory states)

    dimensions = {'eta': eta_dim,
                    's': s_dim,
                    'b': b_dim,
                    'mu': mu_dim, 
                    'pi': pi_dim,
                    'u': u_dim
    }

    if rng_key is None:
        rng_key = global_key

    # Define precision
    Pi = np.array(random.normal(rng_key, shape = (n_var, n_var)))
    _, next_key = random.split(rng_key)
    
    # enforce Markov blanket condition
    Pi[np.ix_(eta_dim, mu_dim)] = 0.0
    Pi[np.ix_(mu_dim, eta_dim)] = 0.0
    
    # enforce symmetry
    Pi = (Pi + Pi.T) / 2

    # Ensure Pi is positive definite
    while (np.linalg.eigvals(Pi) <= 0).any():
        Pi -= 2 * np.linalg.eigvals(Pi).min() * np.eye(n_var)
    
    Pi = jnp.array(Pi) # convert to JAX array after creating the matrix in numpy

    # Compute the stationary covariance
    S = inv(Pi)

    '''
    Setting up the synchronisation map
    '''

    if compute_kernel_condition(Pi[np.ix_(mu_dim, b_dim)], Pi[np.ix_(eta_dim, b_dim)]):
        raise TypeError("Synchronisation map not well defined")

    # define the linear synchronisation map
    b_mu = S[np.ix_(mu_dim, b_dim)] @ inv(S[np.ix_(b_dim, b_dim)])    # expected internal state
    b_eta = S[np.ix_(eta_dim, b_dim)] @ inv(S[np.ix_(b_dim, b_dim)])  # expected external state
    sync = inv(Pi[np.ix_(eta_dim, eta_dim)]) @ Pi[np.ix_(eta_dim, b_dim)] @ pinv(Pi[np.ix_(mu_dim, b_dim)]) @ Pi[np.ix_(mu_dim, mu_dim)]  # sync map
    
    sigma = random.normal(next_key, shape = (n_var, n_var))  # arbitrary volatility matrix
    _, next_key = random.split(next_key)

    # print whether noise is degenerate or not
    print(f'det sigma = {det(sigma)}')

    # diffusion tensor
    D = (sigma @ sigma.T) / 2.0

    # solenoidal flow
    Q = jnp.triu(1. + random.normal(next_key, shape = (n_var, n_var)))  # arbitrary solenoidal flow
    Q = Q - Q.T

    # Drift matrix
    B = (D + Q) @ Pi  # (negative) drift matrix

    if (eigvals(B) <= -1e-5).any() :
        print(eigvals(B))
        raise TypeError("Drift should have non-negative spectrum")        

    # 1) We check it solves the Sylvester equation: BS + SB.T = 2D
    assert jnp.allclose(B @ S + S @ B.T, 2 * D, atol = 1e-7), "Sylvester equation not solved!"

    # 2) we check that there are no numerical errors due to ill conditioning
    assert jnp.allclose(inv(S), Pi, atol = 1e-7), "Precision and inverse covariance are different"

    # 3) We check that the stationary covariance is indeed positive definite
    if (eigvals(S) <= 0).any():
        print(eigvals(S))
        raise TypeError("Stationary covariance not positive definite")

    flow_parameters = {'B': B, 'sigma': sigma, 'Q': Q}

    stationary_stats = {'Pi': Pi, 'S': S}

    sync_mappings = {'b_mu': b_mu, 'b_eta': b_eta, 'sync': sync}

    return flow_parameters, stationary_stats, sync_mappings, dimensions



