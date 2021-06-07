import os
from diffusions import NonlinearProcess
from utilities import plot_hot_colourline
import jax.numpy as jnp
from jax import random, jacfwd
from matplotlib import pyplot as plt
import numpy as np 
import matplotlib.cm as cm
import seaborn as sns

from scipy.stats import pearsonr

# import the 3way configuration variables 
from configs.config_3d import initialize_3d_nonlinear

key = random.PRNGKey(1) # fix random seed for reproducibility

pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}  # setup matplotlib to use latex for output
plt.style.use('seaborn-white')

figures_folder = 'figures'
if not os.path.isdir(figures_folder):
    os.mkdir(figures_folder)

'''
Import the parameters of the state-dependent drift/diffusion process
'''

flow_parameters, stationary_stats, sync_mappings, dimensions = initialize_3d_nonlinear(rng_key = 'default')
# flow_parameters, stationary_stats, sync_mappings, dimensions = initialize_3d_nonlinear(rng_key = random.PRNGKey(2)) # if you want to randomly initialize the precision matrix

n_var = 3       # dimensionality

eta_dim = dimensions['eta']
b_dim = dimensions['b']
mu_dim = dimensions['mu']
pi_dim = dimensions['pi']

Pi, S = stationary_stats['Pi'], stationary_stats['S']

# instantiate the diffusion process
diff_process = NonlinearProcess(n_var, flow_parameters['drift'], flow_parameters['sigma'])

# simulation parameters
dt, T, n_real = 10 ** (-3), 5 * 10 ** 2, 10 ** 5 # integration window duration, number of total timesteps, number of parallel sample paths to simulate

# Setting up the initial condition
x0 = jnp.transpose(random.multivariate_normal(key, jnp.zeros(n_var), jnp.linalg.inv(Pi), shape = (n_real,) ), (1, 0))
_, key = random.split(key) # split the random key

'''
Run simulation
'''

x_t = diff_process.integrate(T, n_real, dt, x0, rng_key = key) # integrate the process

# 2D histogram of stationary distribution
plt.figure()
plt.hist2d(x0[0, :], x0[1, :], bins=(10, 10), cmap=cm.jet)
plt.suptitle('Initial distribution (2D heatmap)')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.close()

# sample path
plt.figure()
for i in range(3):
    plot_hot_colourline(x_t[:, 0, i], x_t[:, 1, i])
plt.close()

# 2D histogram of last time-steps
t = int(T / 2) #intermediate time
plt.figure()
plt.hist2d(x_t[t:, 0, :].reshape(t * n_real), x_t[t:, 1, :].reshape(t * n_real), bins=(100, 100), cmap=cm.jet)
plt.suptitle('2D longtime distribution')
plt.xlabel('x axis')
plt.ylabel('y axis')
fig = plt.gcf()
ratio = 1.3
fig_length = 5
fig.set_size_inches(ratio * fig_length, fig_length, forward=True)
plt.savefig(os.path.join(figures_folder,"longtime_distribution_3way.png"), dpi=100)
plt.close()

# 2D histogram of joint distribution to show x is not a Gaussian process but has Gaussian marginals

# Custom the inside plot: options are: “scatter” | “reg” | “resid” | “kde” | “hex”
sns.set(style="white", color_codes=True)
sns.jointplot(x=x_t[t, 0, :], y=x_t[-1, 0, :], kind='hex', space=0, cmap='Blues', color='skyblue')
plt.xlabel('$x_s$',labelpad=0)
plt.ylabel('$x_t$',labelpad=0)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.tick_params(axis='both', which='major', pad=-3)
fig = plt.gcf()
ratio = 1.3
fig_length = 5
fig.set_size_inches(ratio * fig_length, fig_length, forward=True)
plt.savefig(os.path.join(figures_folder,"non-Gaussian_diffprocess_3way.png"), dpi=100)
plt.close()


'''
Figure: sync map diffusion process
'''

b_mu, b_eta, sync = sync_mappings['b_eta'], sync_mappings['b_mu'], sync_mappings['sync']

# Compute an empirical histogram of the blanket states
x_for_histogram = np.array(jnp.transpose(x_t, (1, 0, 2)).reshape(x_t.shape[1], x_t.shape[0]*x_t.shape[2]).T)

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
plt.savefig(os.path.join(figures_folder, "sync_map_3way_diffprocess.png"), dpi=100)
plt.close()
