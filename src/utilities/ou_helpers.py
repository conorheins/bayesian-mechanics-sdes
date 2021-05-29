import jax.numpy as jnp
from jax import random

RNG_key = random.PRNGKey(1)

def initialize_random_friction(desired_eigs):
    """
    Creates a random drift or friction matrix `B` from a desired set of eigenvalues `desired_eigs`
    """
    
    ndim = len(desired_eigs) # dimensionality of the system

    D = jnp.diag(desired_eigs)
    
    S = random.uniform(RNG_key, shape = (ndim, ndim) );  # random matrix that is likely to be invertible
    
    rank_S = jnp.linalg.matrix_rank(S)

    while rank_S != ndim:
        S = random.uniform(RNG_key, shape = (ndim, ndim))  # continue initializing random matrix until it's invertible
        rank_S = jnp.linalg.matrix_rank(S) # if rank of matrix is equal to dimensionality of matrix, then it's invertible

    B = jnp.dot(S,D).dot(jnp.linalg.inv(S)) # the B matrix will have the same eigenvalues of the diagonal matrix D   

    return B