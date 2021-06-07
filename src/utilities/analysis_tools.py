import jax.numpy as jnp

import numpy as np
from numpy.linalg import matrix_rank
from scipy.linalg import null_space as ker

def rank(A):  # compute matrix rank
    if A.size == 0:
        return 0
    else:
        return matrix_rank(A)

def compute_FE_landscape(b_domain, mu_domain, b_eta, sync, precision):
    """
    Computes the free energy landscapeof a joint Gaussian log density up to an additive constant
    Arguments
    ========
    `b_domain`  [1D jax.DeviceArray]:
    `mu_domain` [1D jax.DeviceArray]:
    `prior`     [1D jax.DeviceArray]:
    `posterior` [1D jax.DeviceArray]:
    `precision` [2D jax.DeviceArray]: inverse of the stationary covariance of particular states
    """

    full_domain = jnp.stack(jnp.meshgrid(b_domain, mu_domain), axis =0)

    reshaped_domain = full_domain.reshape(2, jnp.multiply(*full_domain.shape[1:]))

    potential_term = ((precision @ reshaped_domain) * reshaped_domain).sum(axis=0) / 2.0 # this assumes blanket precisions are stacked on top of internal state precisions

    bold_eta = b_eta * reshaped_domain[0,:] # mapping from blanket states to expected external states
    pred_eta = sync * reshaped_domain[1,:]  # mapping from internal states to expected external states via synchronization map

    KL_term =  precision[1,1] * (pred_eta - bold_eta) ** 2 / 2.0 # full free energy

    F = potential_term + KL_term

    return F.reshape(full_domain.shape[1:])

def compute_kernel_condition(Pi_mub, Pi_etab):
    """
    Check whether rank of the kernel of 
    """

    full_kernel = np.append(ker(Pi_mub), ker(Pi_etab), axis = 1)
    eta_b_kernel = ker(Pi_etab)

    return rank(full_kernel) > rank(eta_b_kernel)


