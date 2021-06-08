import os
from diffusions import LinearProcess
from utilities import initialize_random_friction_numpy, compute_FE_landscape, plot_hot_colourline
import jax.numpy as jnp
from jax.numpy.linalg import inv
from jax import random
import numpy as np 
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

# import the 3way configuration variables 
from configs.config_3d import initialize_3d_OU

key = random.PRNGKey(0)

pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}  # setup matplotlib to use latex for output
plt.style.use('seaborn-white')

figures_folder = 'figures'
if not os.path.isdir(figures_folder):
    os.mkdir(figures_folder)

'''
Setting up the steady-state
'''

flow_parameters, stationary_stats, sync_mappings, dimensions = initialize_3d_OU(rng_key = 'default')
# flow_parameters, stationary_stats, sync_mappings, dimensions = initialize_3d_OU(rng_key = random.PRNGKey(1)) # if you want to randomly initialize the quantities of interest

n_var = 3       # dimensionality

eta_dim = dimensions['eta']
b_dim = dimensions['b']
mu_dim = dimensions['mu']
pi_dim = dimensions['pi']

Pi, S = stationary_stats['Pi'], stationary_stats['S']

# Setting up the OU process
process = LinearProcess(dim=n_var, friction=flow_parameters['B'], volatility=flow_parameters['sigma'])  # create process

'''
OU process steady-state simulation
'''
#this stationary simulation is super important to see how well the synchronisation map works for the stationary process
#despite errors in numerical discretisation and matrix ill-conditioning
#all subsequent simulations can only be trusted to the extent that the synchronisation map
# works in the steady-state simulation

dt, T, n_real = 0.01, 500, 1000 # duration of single time-step, total number of integration timesteps, number of parallel paths to simulate

# initialize process starting state
x0 = jnp.transpose(random.multivariate_normal(key, jnp.zeros(n_var), S, shape = (n_real,) ), (1, 0))
_, key = random.split(key)

# sample paths
x = process.integrate(T, n_real, dt, x0, rng_key = key) # run simulation

'''
Figure 1: sync map OU process
'''

b_mu, b_eta, sync = sync_mappings['b_eta'], sync_mappings['b_mu'], sync_mappings['sync']

# Compute an empirical histogram of the blanket states
x_for_histogram = np.array(jnp.transpose(x, (1, 0, 2)).reshape(x.shape[1], x.shape[0]*x.shape[2]).T)

b_hist = x_for_histogram[:,b_dim].squeeze()

_, bin_edges = np.histogram(b_hist, bins = 650)
bin_idx_b = np.digitize(b_hist, bin_edges, right=True)
bin_counts = np.histogram(b_hist, bin_edges)[0]
bin_centers = bin_edges[:-1] + 0.5*np.diff(bin_edges)

unique_bins, unique_bin_occurrences = np.unique(bin_idx_b, return_inverse = True)

# Calculate empirical average of internal and external states, given particular occurrences of a 'bin' of blanket states

mu_cond_b = np.empty_like(bin_centers)
eta_cond_b = np.empty_like(bin_centers)

mu_binned = x_for_histogram[:,mu_dim].squeeze()
eta_binned = x_for_histogram[:,eta_dim].squeeze()

for unique_bin_idx in range(len(unique_bins)):

    mu_cond_b[unique_bins[unique_bin_idx]-1] = mu_binned[unique_bin_occurrences == unique_bin_idx].mean()
    eta_cond_b[unique_bins[unique_bin_idx]-1] = eta_binned[unique_bin_occurrences == unique_bin_idx].mean()

# map the empirical most likely internal state to a corresponding expected external state, via the synchronization map
sync_boldmu = sync * mu_cond_b

plt.figure(1)
plt.suptitle('Synchronisation map')
plt.scatter(bin_centers[bin_counts > 1000], sync_boldmu[bin_counts > 1000], s=1, alpha=0.5, label='Prediction: $\sigma(\mathbf{\mu}(b_t))$')  # scatter plot theoretical expected internal state
plt.scatter(bin_centers[bin_counts > 1000], eta_cond_b[bin_counts > 1000], s=1, alpha=0.5, label='Actual: $\mathbf{\eta}(b_t)$')
plt.xlabel('Blanket state space $\mathcal{B}$')
plt.ylabel('External state space $\mathcal{E}$')
cor = pearsonr(sync_boldmu[bin_counts > 1000], eta_cond_b[bin_counts > 1000])[0]
plt.title(f'Pearson correlation = {jnp.round(cor, 6)}...')
plt.legend(loc='upper right')
plt.savefig(os.path.join(figures_folder, "sync_map_3wayOUprocess.png"), dpi=100)
plt.close()

'''
OU process perturbed simulation
'''

_, key = random.split(key)

b = 10. # specify perturbed blanket state

b_init = b * jnp.ones(n_real)

#initialising external and internal states at posterior distributions
mu_init = random.multivariate_normal(key, jnp.array([b_mu*b]), inv(Pi[mu_dim,mu_dim][...,None]), shape = (n_real,) ).squeeze()
_, key = random.split(key)
eta_init = random.multivariate_normal(key, jnp.array([b_eta*b]), inv(Pi[eta_dim,eta_dim][...,None]), shape = (n_real,) ).squeeze()
_, key = random.split(key)

x0 = jnp.stack( (eta_init, b_init, mu_init), axis = 0)

# sample paths
x = process.integrate(T, n_real, dt, x0, rng_key = key) # run simulation

'''
Figure 2: Heat map of F(mu,b) and sample path
'''

S_part_inv = inv(S[np.ix_(pi_dim, pi_dim)]) # stationary covariance of particular states

internal = jnp.linspace(jnp.min(x[:, mu_dim, :]) - 1, jnp.max(x[:, mu_dim, :]) + 0.5, 105)  # internal state-space points
blanket = jnp.linspace(jnp.min(x[:, b_dim, :]) - 1, jnp.max(x[:, b_dim, :]) + 0.5, 100)  # internal state-space points

F_landscape = compute_FE_landscape(blanket, internal, b_eta, sync, S_part_inv)

realisation_idx = 3  # which sample path to show (between 0 and N)

plt.figure(2)
plt.title('Free energy $F(b, \mu)$')
plt.contourf(blanket, internal, F_landscape, levels=100, cmap='turbo')  # plotting the free energy landscape
plt.colorbar()
plt.ylabel('internal state $ \mu$')
plt.xlabel('blanket state $b$')
plt.plot(blanket, b_mu * blanket, c='white')  # plot expected internal state as a function of blanket states

blanket_trajectory = x[:, b_dim, realisation_idx].squeeze()
mu_trajectory = x[:, mu_dim, realisation_idx].squeeze()

plot_hot_colourline(blanket_trajectory, mu_trajectory, lw=0.5)
plt.text(s='$\mathbf{\mu}(b)$', x= x[:, b_dim, :].min()  - 0.7, y= b_mu * (x[:, b_dim, :].min() - 0.7) + 0.2, color='white')

plt.text(s='$(b_t, \mu_t)$', x=blanket_trajectory[0] - 2, y=mu_trajectory[0], color='black')
plt.savefig(os.path.join(figures_folder,"Sample_perturbed_3wayOU.png"), dpi=100)
plt.close()



