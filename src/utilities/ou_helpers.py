import jax.numpy as jnp
from jax import random

import numpy as np

RNG_key = random.PRNGKey(1)
numpy_rng = np.random.default_rng(1)

def initialize_random_friction_jax(desired_eigs, key = RNG_key):
    """
    Creates a random drift or friction matrix `B` from a desired set of eigenvalues `desired_eigs`
    """
    
    ndim = len(desired_eigs) # dimensionality of the system

    D = jnp.diag(desired_eigs)
    
    S = random.uniform(key, shape = (ndim, ndim) );  # random matrix that is likely to be invertible
    _, next_key = random.split(key)

    rank_S = jnp.linalg.matrix_rank(S)

    while rank_S != ndim:
        S = random.uniform(next_key, shape = (ndim, ndim))  # continue initializing random matrix until it's invertible
        _, next_key = random.split(next_key) 
        rank_S = jnp.linalg.matrix_rank(S) # if rank of matrix is equal to dimensionality of matrix, then it's invertible

    B = jnp.dot(S,D).dot(jnp.linalg.inv(S)) # the B matrix will have the same eigenvalues of the diagonal matrix D   

    return B, next_key

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

def sylvester_equation_solve(B, D):
    """
    Simple way to solve Sylvester equation using linear regression. Ripped from Karl Friston's MATLAB function `spm_ness`
    Sylvester relation: B.dot(D) + D.dot(B.T) = S
    ARGUMENTS:
    ========
    `B` [np.ndarray]: (n_states, n_states) size input matrix 
    `D` [np.ndarray]: (n_states, n_states) size input matrix
    RETURNS:
    ======
    `S` [np.ndarray]: (n_states, n_states) size output matrix (the solution to the Sylvester equation)
    """

    n_states = B.shape[0]

    I = jnp.eye(n_states)
    X = jnp.kron(I,B) + jnp.kron(jnp.conj(B),I) 
    Y = (2*D).flatten()
    S = jnp.linalg.solve(X, Y).reshape(n_states,n_states) # solve for S is now converted into a simple linear regression  problem 

    return S