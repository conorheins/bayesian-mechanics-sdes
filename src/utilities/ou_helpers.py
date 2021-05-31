import jax.numpy as jnp
from jax import random

import numpy as np

RNG_key = random.PRNGKey(1)
numpy_rng = np.random.default_rng(1)

def initialize_random_friction_jax(desired_eigs):
    """
    Creates a random drift or friction matrix `B` from a desired set of eigenvalues `desired_eigs`
    """
    
    ndim = len(desired_eigs) # dimensionality of the system

    D = jnp.diag(desired_eigs)
    
    S = random.uniform(RNG_key, shape = (ndim, ndim) );  # random matrix that is likely to be invertible
    
    rank_S = jnp.linalg.matrix_rank(S)

    while rank_S != ndim:
        S = random.uniform(next_key, shape = (ndim, ndim))  # continue initializing random matrix until it's invertible
        RNG_key, next_key = random.split(RNG_key) 
        rank_S = jnp.linalg.matrix_rank(S) # if rank of matrix is equal to dimensionality of matrix, then it's invertible

    B = jnp.dot(S,D).dot(jnp.linalg.inv(S)) # the B matrix will have the same eigenvalues of the diagonal matrix D   

    return B

def initialize_random_friction_numpy(desired_eigs):
    """
    Creates a random drift or friction matrix `B` from a desired set of eigenvalues `desired_eigs`
    """
    
    ndim = len(desired_eigs) # dimensionality of the system

    D = np.diag(desired_eigs)
    
    S = numpy_rng.random(ndim**2).reshape(ndim,ndim);  # random matrix that is likely to be invertible

    rank_S = np.linalg.matrix_rank(S)

    while rank_S != ndim:
        S = numpy_rng.random(ndim**2).reshape(ndim,ndim)  # continue initializing random matrix until it's invertible
        rank_S = np.linalg.matrix_rank(S) # if rank of matrix is equal to dimensionality of matrix, then it's invertible

    B = np.dot(S,D).dot(np.linalg.inv(S)) # the B matrix will have the same eigenvalues of the diagonal matrix D   

    return B