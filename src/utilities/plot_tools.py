import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn

import jax.numpy as jnp
from jax import random
from .analysis_tools import compute_time_dependent_density

default_key = random.PRNGKey(1)

def plot_hot_colourline(x, y, alpha = 1.0, lw=0.5):
    d= jnp.arange(len(x))
    c= cm.hot((d - d.min()) / jnp.ptp(d) )
    ax = plt.gca()
    for i in jnp.arange(len(x) - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], c=c[i], linewidth=lw, alpha = alpha)
    return


def plot_b_mu_evolving_density(trajectories, # history of trajectories of shape (num_timesteps, num_dimensions, num_realizations)
                                b_dim, # indices of the blanket state dimensions
                                mu_dim,  # indices of the internal state dimensions
                                start_T = None, # time at which to start measuring probability densities
                                end_T = None,  # time at which to cease measuring probability density
                                back_window = None, 
                                forward_window = None, 
                                plot_average = False, 
                                plot_paths = False
                            ):  

    b_edges = jnp.linspace(jnp.min(trajectories[:, b_dim, :]) - 1, jnp.max(trajectories[:, b_dim, :]) + 0.5, 100)   # blanket state-space bin edges
    mu_edges = jnp.linspace(jnp.min(trajectories[:, mu_dim, :]) - 1, jnp.max(trajectories[:, mu_dim, :]) + 0.5, 105) # internal state-space bin edges

    bin_edges = [b_edges, mu_edges]

    density_over_time, b_bin_centers, mu_bin_centers = compute_time_dependent_density(trajectories, b_dim, mu_dim, start_T, end_T, back_window, forward_window)

    plt.contourf(b_bin_centers, mu_bin_centers, density_over_time.T, levels=100, cmap='turbo')  # plotting the free energy landscape

    plt.ylabel('internal state $ \mu$')
    plt.xlabel('blanket state $b$')

    if plot_average:
        avg_b_trajectory = trajectories[:, b_dim, :].mean(axis=2).squeeze()
        avg_mu_trajectory = trajectories[:, mu_dim, :].mean(axis=2).squeeze()
        plot_hot_colourline(avg_b_trajectory[:end_T], avg_mu_trajectory[:end_T], lw = 1.5)

    if plot_paths:
        num_real_to_show = 25
        realization_idx = random.choice(default_key, trajectories.shape[2], shape = (num_real_to_show,))
        for real_i in realization_idx:
            b_real = trajectories[start_T:end_T, b_dim, real_i].squeeze()
            mu_real = trajectories[start_T:end_T, mu_dim, real_i].squeeze()
            plot_hot_colourline(b_real,mu_real, alpha = 0.65, lw = 0.25)

    return density_over_time, b_bin_centers, mu_bin_centers
