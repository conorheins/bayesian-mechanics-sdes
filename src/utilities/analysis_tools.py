import jax.numpy as jnp
from jax import vmap

import numpy as np
from numpy.linalg import matrix_rank
from scipy.linalg import null_space as ker

def rank(A):  # compute matrix rank
    if A.size == 0:
        return 0
    else:
        return matrix_rank(A)

def compute_FE_landscape(b_domain, mu_domain, b_eta, sync, precision, q_precision):
    """
    Computes the free energy landscapeof a joint Gaussian log density up to an additive constant
    Arguments
    ========
    `b_domain`    [1D jax.DeviceArray]:
    `mu_domain`   [1D jax.DeviceArray]:
    `prior`       [1D jax.DeviceArray]:
    `posterior`   [1D jax.DeviceArray]:
    `precision`   [2D jax.DeviceArray]: inverse of the stationary covariance of particular states
    `q_precision` [2D jax.DeviceArray]: precision of external states, equivalent to the variational precision / posterior precision
    """

    full_domain = jnp.stack(jnp.meshgrid(b_domain, mu_domain), axis =0)

    reshaped_domain = full_domain.reshape(2, jnp.multiply(*full_domain.shape[1:]))

    potential_term = ((precision @ reshaped_domain) * reshaped_domain).sum(axis=0) / 2.0 # this assumes blanket precisions are stacked on top of internal state precisions

    bold_eta = b_eta * reshaped_domain[0,:] # mapping from blanket states to expected external states
    pred_eta = sync * reshaped_domain[1,:]  # mapping from internal states to expected external states via synchronization map

    KL_term =  q_precision * (pred_eta - bold_eta) ** 2 / 2.0 # KL divergence term

    F = potential_term + KL_term

    return F.reshape(full_domain.shape[1:])

def compute_kernel_condition(Pi_mub, Pi_etab):
    """
    Check whether rank of the kernel of 
    """

    full_kernel = np.append(ker(Pi_mub), ker(Pi_etab), axis = 1)
    eta_b_kernel = ker(Pi_etab)

    return rank(full_kernel) > rank(eta_b_kernel)

def compute_Fboldmu_blanket_landscape(s_domain, a_domain, b_mu, part_precision):
    """
    Computes the free energy landscapeof a joint Gaussian log density up to an additive constant
    Arguments
    ========
    `s_domain`  [1D vector - jax.DeviceArray]:
    `a_domain` [1D vector - jax.DeviceArray]:
    `b_mu`     [2 x 2 matrix - jax.DeviceArray]:
    `part_precision` [4D jax.DeviceArray]: inverse of the stationary covariance of particular states
    """

    full_blankets = jnp.stack(jnp.meshgrid(s_domain, a_domain), axis =0)

    full_blankets_reshaped = full_blankets.reshape(2, jnp.multiply(*full_blankets.shape[1:]))

    bold_mu = b_mu @ full_blankets_reshaped

    # following 2 lines assume that blanket precisions are stacked on top of internal state precisions
    reshaped_domain = jnp.concatenate( (full_blankets_reshaped, bold_mu), axis = 0)
    F = ((part_precision @ reshaped_domain) * reshaped_domain).sum(axis=0) / 2.0 

    return F.reshape(full_blankets.shape[1:])

def compute_Fboldmu_blanket_over_time(blanket_hist, b_mu, part_precision):
    """
    Computes particular free energy F(a, s, bold_mu) over time
    """

    bold_mu_over_time = lambda blankets: b_mu @ blankets
    compute_F_over_time = lambda particular_bmu: ((part_precision @ particular_bmu) * particular_bmu).sum(axis=0) / 2.0 

    bold_mu_over_time_vec = vmap(bold_mu_over_time, in_axes = 1, out_axes = 1)
    F_over_time_vec = vmap(compute_F_over_time, in_axes = 1, out_axes = 1)

    boldmu_hist = bold_mu_over_time_vec(blanket_hist) # compute bold_mu over time, vectorized across realizations

    F_over_time = F_over_time_vec(jnp.concatenate((blanket_hist, boldmu_hist), axis = 0))

    return F_over_time

def eigsorted(cov):
    vals, vecs = jnp.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]




