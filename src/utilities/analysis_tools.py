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

def compute_F_over_time(part_states_hist, b_eta, sync, precision, q_precision):
    """
    Computes variational free energy variational free energy of particular states F(pi) where pi = {b, mu} 
    over time, for multiple parallel realizations of the O-U process
    """

    bold_eta_over_time = b_eta * part_states_hist[1,:,:] # first row of part_states_hist is blanket states
    pred_eta_over_time = sync * part_states_hist[2,:,:]  # second row of part_states_hist is internal states

    compute_potential_over_time = lambda part_states: ((precision @ part_states) * part_states).sum(axis=0) / 2.0 # log potential term over time

    potential_over_time_vec = vmap(compute_potential_over_time, in_axes = 1, out_axes = 1) # log potential term over time, vectorized across realizations

    potential_over_time = potential_over_time_vec(part_states_hist) # compute log potential timeseries across realizations

    KL_over_time =  q_precision * (pred_eta_over_time - bold_eta_over_time) ** 2 / 2.0 # KL divergence term over time and realizations

    F_over_time = potential_over_time + KL_over_time.T

    return F_over_time

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
    """
    Get eigenvalues / eigenvectors and return them sorted in order of greatest-to-smallest eigenvalue
    """
    vals, vecs = jnp.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def compute_time_dependent_density(trajectories, b_dim, mu_dim, start_T = None, end_T = None, back_window = None, forward_window = None):
    """
    Compute evolving histogram/state density using sliding window method to allow past histograms to influence final result
    """

    if start_T is None:
        start_T = 0

    if end_T is None:
        end_T = trajectories.shape[0]

    if back_window is None:
        back_window = end_T
    
    if forward_window is None:
        forward_window = 10
    
    n_real = trajectories.shape[2]

    b_edges = jnp.linspace(jnp.min(trajectories[:, b_dim, :]) - 1, jnp.max(trajectories[:, b_dim, :]) + 0.5, 100)   # blanket state-space bin edges
    mu_edges = jnp.linspace(jnp.min(trajectories[:, mu_dim, :]) - 1, jnp.max(trajectories[:, mu_dim, :]) + 0.5, 105) # internal state-space bin edges

    bin_edges = [b_edges, mu_edges]

    num_part_states = len(b_dim + mu_dim)

    # cast everything to numpy so we can change the density over time

    density_over_time = np.zeros((b_edges.shape[0]-1, mu_edges.shape[0]-1))

    p_t_prev = np.zeros_like(density_over_time) # initialize previous time density

    particular_states_hist = np.array(trajectories[:,b_dim + mu_dim,:])

    for t in range(start_T,end_T):

        t_indices = np.arange(max(0, t - back_window), min(end_T, t + forward_window))

        particular_states_t = np.transpose(particular_states_hist[t_indices], (1, 0, 2)).reshape(num_part_states, len(t_indices) * n_real).T

        p_t_next = np.histogramdd(particular_states_t, bins = bin_edges, density = True)[0]

        density_over_time[p_t_next > 0.0] = p_t_next[p_t_next > 0.0]

        smoothed_pt = (p_t_prev + p_t_next) / 2.0
        smoothed_pt = smoothed_pt / smoothed_pt.max()

        nonzero_entries = np.where(smoothed_pt)

        if len(nonzero_entries[0]) > 0:
            
            density_over_time[nonzero_entries] = smoothed_pt[nonzero_entries]

        p_t_prev = p_t_next
    
    b_bin_centers = b_edges[:-1] + np.diff(b_edges) / 2.0
    mu_bin_centers = mu_edges[:-1] + np.diff(mu_edges) / 2.0

    return density_over_time, b_bin_centers, mu_bin_centers



