import jax.numpy as jnp
from jax.numpy.linalg import eigvals, det, inv
from jax import random, jacfwd

import numpy as np

def initialize_3d_OU(rng_key = 'default'):
    """
    Initialize the relevant variables of a 3-way OU process with an optional random seed
    and return them in parameter dictionaries
    Arguments:
    ==========
    `rng_key` [str or jax.random PRNGkey]:
    Returns:
    =========
    `flow_parameters`  [Dict]:
    `stationary_stats` [Dict]: 
    `sync_mappings`    [Dict]: 
    `dimensions`       [Dict]:
    """
    eta_dim = [0]
    b_dim = [1]
    mu_dim = [2]
    pi_dim = b_dim + mu_dim # particular states

    dimensions = {'eta': eta_dim, 
                    'b': b_dim, 
                    'mu': mu_dim, 
                    'pi': pi_dim}

    if rng_key == 'default':
        # Define precision
        Pi = jnp.array([ [3., 1., 0.], 
                        [1., 3., 0.5], 
                        [0., 0.5, 2.5] ]) #selected precision matrix
        
        # volatility
        sigma = jnp.array([ [2, 1.5, 0.5],
                            [0., 3., 2.],
                            [0., 0., 2.] ]) #selected non-degenerate noise

        #diffusion tensor
        D = sigma @ sigma.T / 2  # diffusion tensor

        # solenoidal flow
        Q = jnp.array([ [0., 1.5, 0.5],
                        [-1.5, 0., -1.],
                        [-0.5, 1., 0.] ]) #selected solenoidal flow

    else:

        Pi = np.array(random.normal(rng_key, shape=(3,3))) #random precision matrix
        _, next_key = random.split(rng_key)
        Pi[eta_dim, mu_dim] = 0.0
        Pi[mu_dim, eta_dim] = 0.0

        # enforce symmetric
        Pi = (Pi + Pi.T) / 2.0

        # Ensure Pi is positive definite
        while (np.linalg.eigvals(Pi) <= 0).any():
            Pi -= 2 * np.linalg.eigvals(Pi).min() * np.eye(3)
        
        Pi = jnp.array(Pi) # convert back to JAX array at the end

        # arbitrary volatility matrix
        # sigma = jnp.diag(random.normal(next_key, shape=(3,))) # arbitrary diagonal volatility matrix
        sigma = random.normal(next_key, shape=(3,3))
        _, next_key = random.split(next_key)

         #diffusion tensor
        D = sigma @ sigma.T / 2  # diffusion tensor

        # see whether noise is degenerate or not
        print(f'det sigma = {det(sigma)}')

        Q = random.normal(next_key, shape=(3,3))  # arbitrary solenoidal flow
        _, next_key = random.split(next_key)

        Q = (Q - Q.T)/2 # ensure anti-symmetry

    # Compute the stationary covariance
    S = inv(Pi)

    # Check that the synchronisation map is well defined
    assert S[mu_dim, b_dim] != 0, "Synchronisation map not well defined: bold_mu(b) not invertible!"

    '''
    Setting up synchronisation map
    '''
    
    # define the linear synchronisation map
    b_mu = S[mu_dim, b_dim] * S[b_dim, b_dim] ** (-1)  # expected internal state
    b_eta = S[eta_dim, b_dim] * S[b_dim, b_dim] ** (-1)  # expected external state
    sync = Pi[eta_dim, eta_dim] ** (-1) * Pi[eta_dim, b_dim] * Pi[mu_dim, b_dim] ** (-1) * Pi[mu_dim, mu_dim]  # sync map
  
    # Drift matrix
    B = (D + Q) @ Pi  # drift matrix

    if (eigvals(B) <= -1e-5).any():
        print(eigvals(B))
        raise TypeError("Drift should have non-negative spectrum")
        
    # 1) We check it solves the Sylvester equation: BS + SB.T = 2D
    assert jnp.allclose(B @ S + S @ B.T, 2 * D), "Sylvester equation not solved!"

    # 2) we check that there are no numerical errors due to ill conditioning
    assert jnp.allclose(inv(S), Pi), "Precision and inverse covariance are different"

    # We check that the stationary covariance is indeed positive definite
    if (eigvals(S) <= 0).any():
        print(eigvals(S))
        raise TypeError("Stationary covariance not positive definite")

    flow_parameters = {'B': B, 'sigma': sigma, 'Q': Q}

    stationary_stats = {'Pi': Pi, 'S': S}

    sync_mappings = {'b_mu': b_mu, 'b_eta': b_eta, 'sync': sync}

    return flow_parameters, stationary_stats, sync_mappings, dimensions

def initialize_3d_nonlinear(rng_key='default'):
    """
    Initialize the relevant variables of a 3-way diffusion process with an optional random seed
    and return them in parameter dictionaries
    Arguments:
    ==========
    `rng_key` [str or jax.random PRNGkey]:
    Returns:
    =========
    `flow_parameters`  [Dict]:
    `stationary_stats` [Dict]: 
    `sync_mappings`    [Dict]: 
    `dimensions`       [Dict]:
    """

    eta_dim = [0]
    b_dim = [1]
    mu_dim = [2]
    pi_dim = b_dim + mu_dim # particular states

    dimensions = {'eta': eta_dim, 
                    'b': b_dim, 
                    'mu': mu_dim, 
                    'pi': pi_dim}

    if rng_key == 'default':
        # Define precision
        Pi = jnp.array([ [3., 1., 0.], 
                        [1., 3., 0.5], 
                        [0., 0.5, 2.5] ]) #selected precision matrix
    else:
        Pi = np.array(random.normal(rng_key, shape=(3,3))) #random precision matrix
        _, next_key = random.split(rng_key)
        Pi[eta_dim, mu_dim] = 0.0
        Pi[mu_dim, eta_dim] = 0.0

        # enforce symmetric
        Pi = (Pi + Pi.T) / 2.0

        # Ensure Pi is positive definite
        while (np.linalg.eigvals(Pi) <= 0).any():
            Pi -= 2 * np.linalg.eigvals(Pi).min() * np.eye(3)
        
        Pi = jnp.array(Pi) # convert back to JAX array at the end
    
    # Compute the stationary covariance
    S = inv(Pi)

    # Check that the synchronisation map is well defined
    assert S[mu_dim, b_dim] != 0, "Synchronisation map not well defined: bold_mu(b) not invertible!"

    '''
    Setting up synchronisation map
    '''
    
    # define the linear synchronisation map
    b_mu = S[mu_dim, b_dim] * S[b_dim, b_dim] ** (-1)  # expected internal state
    b_eta = S[eta_dim, b_dim] * S[b_dim, b_dim] ** (-1)  # expected external state
    sync = Pi[eta_dim, eta_dim] ** (-1) * Pi[eta_dim, b_dim] * Pi[mu_dim, b_dim] ** (-1) * Pi[mu_dim, mu_dim]  # sync map

    '''
    Parameterise state dependent diffusion and solenoidal flow functions
    '''

    # volatility
    def sigma(y):
        return jnp.diag(y)

    # diffusion
    def D(y):
        sigma_y = sigma(y)
        return (sigma_y @ sigma_y.T) / 2.0

    j_D = jacfwd(D)

    def divD(y):
        return jnp.trace(j_D(y), axis1=0, axis2=2)

    # # divergence of diffusion
    # def divD(y):
    #     return y

    # solenoidal flow
    def Q(y):
        temp = jnp.tile(y, (y.shape[0], 1))
        return temp - temp.T

    j_Q = jacfwd(Q)

    # divergence of solenoidal flow
    def divQ(y):
        return jnp.trace(j_Q(y), axis1=1, axis2=2)

    # drift
    def drift(y):
        # return -(D(y)) @ Pi @ y - divD(y)
        return -(D(y) + Q(y)) @ Pi @ y + divD(y) + divQ(y)

    flow_parameters = {'drift': drift, 'sigma': sigma, 'D': D, 'Q': Q, 'divD': divD, 'divQ': divQ}

    stationary_stats = {'Pi': Pi, 'S': S}

    sync_mappings = {'b_mu': b_mu, 'b_eta': b_eta, 'sync': sync}

    return flow_parameters, stationary_stats, sync_mappings, dimensions

