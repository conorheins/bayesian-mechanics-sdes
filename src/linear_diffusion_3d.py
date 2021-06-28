import os
from diffusions import LinearProcess
from utilities import plot_hot_colourline, parse_command_line
from utilities import compute_FE_landscape, compute_F_over_time, plot_b_mu_evolving_density

import jax.numpy as jnp
from jax.numpy.linalg import inv
from jax import vmap
from scipy.linalg import sqrtm
from jax import random
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy import stats

# import the 3way configuration variables 
from configs.config_3d import initialize_3d_OU

key, save_mode = parse_command_line(default_key = 1)

pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}  # setup matplotlib to use latex for output
plt.style.use('seaborn-white')

if save_mode:
    figures_folder = 'figures'
    if not os.path.isdir(figures_folder):
        os.mkdir(figures_folder)
    
    figures_folder = os.path.join(figures_folder, '3d_bayesmech')
    if not os.path.isdir(figures_folder):
        os.mkdir(figures_folder)

    seed_folder = os.path.join(figures_folder, f'seed_{key[1]}')
    if not os.path.isdir(seed_folder):
        os.mkdir(seed_folder)

'''
Setting up the steady-state
'''

flow_parameters, stationary_stats, sync_mappings, dimensions = initialize_3d_OU(rng_key = 'default') # default parameterisation, gives a pre-defined steady-state / precision matrix
# flow_parameters, stationary_stats, sync_mappings, dimensions = initialize_3d_OU(rng_key = key) # if you want to randomly initialize the quantities of interest

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

b_eta, b_mu, sync = sync_mappings['b_eta'], sync_mappings['b_mu'], sync_mappings['sync']

'''
Figure 1: sync map OU process
'''

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
plt.scatter(bin_centers[bin_counts > 1000], eta_cond_b[bin_counts > 1000], s=1, alpha=0.5, label='External: $\mathbf{\eta}(b_t)$')
plt.xlabel('Blanket state-space $\mathcal{B}$')
plt.ylabel('External state-space $\mathcal{E}$')
cor = stats.pearsonr(sync_boldmu[bin_counts > 1000], eta_cond_b[bin_counts > 1000])[0]
plt.title(f'Pearson correlation = {np.round(cor, 6)}...')
plt.legend(loc='upper right')
if save_mode:
    figure_name = "sync_map_3wayOUprocess.png"
    plt.savefig(os.path.join(seed_folder,figure_name), dpi=100)
    plt.close()

'''
OU process perturbed simulation
'''

T = 1000 # do a long enough trajectory for the predictive processing figure later

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

T_end_fe = 500

S_part_inv = inv(S[np.ix_(pi_dim, pi_dim)]) # stationary covariance of particular states

internal = jnp.linspace(jnp.min(x[:, mu_dim, :]) - 1, jnp.max(x[:, mu_dim, :]) + 0.5, 105)  # internal state-space points
blanket = jnp.linspace(jnp.min(x[:, b_dim, :]) - 1, jnp.max(x[:, b_dim, :]) + 0.5, 100)  # internal state-space points

F_landscape = compute_FE_landscape(blanket, internal, b_eta, sync, S_part_inv, Pi[eta_dim, eta_dim])

# realisation_idx = 100  # which sample path to show (between 0 and n_real)

realisation_idx = random.randint(key, shape=(), minval = 0, maxval = n_real)  # which sample path to show (between 0 and n_real)
print(f'Sample path index being shown: {realisation_idx}\n')

plt.figure(2)
plt.title('Free energy $F(b, \mu)$',fontsize=16)
plt.contourf(blanket, internal, F_landscape, levels=100, cmap='turbo')  # plotting the free energy landscape
cb  = plt.colorbar(ticks = np.round(np.arange(0,400,step=50)))
cb.ax.tick_params(labelsize=14)
plt.ylabel('internal state $ \mu$',fontsize=14)
plt.xlabel('blanket state $b$',fontsize=14)
plt.plot(blanket, b_mu * blanket, c='white')  # plot expected internal state as a function of blanket states

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

blanket_trajectory = x[:T_end_fe, b_dim, realisation_idx].squeeze()
mu_trajectory = x[:T_end_fe, mu_dim, realisation_idx].squeeze()

plot_hot_colourline(blanket_trajectory, mu_trajectory, lw=0.5)
plt.text(s='$\mathbf{\mu}(b)$', x= x[:T_end_fe, b_dim, :].min()  - 0.7, y= b_mu * (x[:T_end_fe, b_dim, :].min() - 0.7) + 0.25, color='white', fontsize=16)
plt.text(s='$(b_t, \mu_t)$', x=x[1, b_dim, realisation_idx] - 1.2, y=x[1, mu_dim, realisation_idx] + 0.35, color='black', fontsize=16)

if save_mode:
    figure_name = "Sample_perturbed_3wayOU.png"
    plt.savefig(os.path.join(seed_folder,figure_name), dpi=100)
    plt.close()


'''
Figure 3: average free energy over trajectories
'''


particular_states_hist = jnp.transpose(x[:T_end_fe,b_dim + mu_dim,:], (1, 0, 2)) # get full particular states (blanket & internal)
F_over_time = compute_F_over_time(particular_states_hist, b_eta, sync, S_part_inv, Pi[eta_dim, eta_dim])

mean_F = F_over_time.mean(axis=0)

plt.figure(3)
plt.clf()
plt.title('Average free energy over time',fontsize=16)
plot_hot_colourline(np.arange(T_end_fe), mean_F)
xlabel_pos = int(T_end_fe * 0.4)  # position of text on x axis
plt.text(s='$F(b_t, \mu_t)$', x=xlabel_pos, y=mean_F[xlabel_pos] + 0.05 * (np.max(mean_F) - mean_F[xlabel_pos]), color='black', fontsize=16)
plt.xlabel('Time',fontsize=14)
plt.ylabel('Free energy $F(b_t, \mu_t)$',fontsize=14)
plt.xlim(0, T_end_fe)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

if save_mode:
    figure_name = "FE_vs_time_perturbed_3wayOU.png"
    plt.savefig(os.path.join(seed_folder,figure_name), dpi=100)
    plt.close()

'''
Figure 4: predictive processing
'''
T_end_PP = 500

sample_trajectory = x[:T_end_PP,:,realisation_idx] # choose sample trajectory

eta_samples = sample_trajectory[:,eta_dim].squeeze()
mu_samples = sample_trajectory[:,mu_dim].squeeze()
# b_samples = sample_trajectory[:,b_dim].squeeze()

posterior_means = sync @ mu_samples.reshape(1,-1) # predicted external states, given instantaneous internal states -- sigma(mu)
# posterior_means = b_eta @ b_samples.reshape(1,-1) # predicted external states, given instantaneous blanket states -- sigma(boldmu)

posterior_cov = inv(Pi[eta_dim,eta_dim][...,None])

conf_interval_param = 1.96
std_mu_0 = jnp.sqrt(posterior_cov)
pred_upper_CI_mu0 = posterior_means + conf_interval_param * std_mu_0
pred_lower_CI_mu0 = posterior_means - conf_interval_param * std_mu_0

t_axis = np.arange(T_end_PP)
plt.figure()

plt.clf()
plt.title('Predictive processing: $q_{\mu_t}(\eta)$ vs $\eta_t$',fontsize = 16)
# plt.title('Predictive processing: $q_{\mathbf{\mu_t}}(\eta)$ vs $\eta_t$',fontsize = 16)

plt.fill_between(t_axis,pred_upper_CI_mu0[0,:], pred_lower_CI_mu0[0,:], color='#4ba2d1', alpha=0.25)
eta1_real_line = plt.plot(t_axis, eta_samples, lw = 1.25, color = '#d12f13', alpha=1.0, label='External: $\eta_{t}$')
# mu1_mean_line = plt.plot(t_axis,posterior_means, color='#27739c',label='Prediction: $q_{\mathbf{\mu}_t}(\eta)$',lw=1.5)
mu1_mean_line = plt.plot(t_axis,posterior_means, color='#27739c',label='Prediction: $q_{\mu_t}(\eta)$',lw=1.5)

ci_patch_1 = Patch(color='#4ba2d1',alpha=0.2, label=' ')

first_legend = plt.legend(handles=[ci_patch_1], fontsize=14, loc=(0.275,0.126), ncol = 1)
# Add the legend manually to the current Axes.
plt.gca().add_artist(first_legend)
plt.legend(handles=[mu1_mean_line[0], eta1_real_line[0]], loc='lower center', ncol = 1,  fontsize=14)

min_value = min( pred_lower_CI_mu0.min(), eta_samples.min() )
max_value = max( pred_upper_CI_mu0.max(), eta_samples.max() )

plt.xlim(t_axis[0], t_axis[-1])
plt.ylim(1.25 * min_value, 1.25 * max_value)

plt.gca().tick_params(axis='both', which='both')
plt.gca().set_xlabel('Time', fontsize=14)
plt.gca().set_ylabel('External state-space $\mathcal{E}$', fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

if save_mode:
    figure_name = "average_prediction_3way_OUprocess.png"
    plt.savefig(os.path.join(seed_folder,figure_name), dpi=100)
    plt.close()


'''
Figure 5 - plot the prediction errors evolving over time
'''

predictions = sync * jnp.transpose(x[:,mu_dim,:], (1,0,2)).squeeze()
# predictions = b_eta * jnp.transpose(x[:,b_dim,:], (1,0,2)).squeeze()

eta_samples = jnp.transpose(x[:,eta_dim,:], (1,0,2)).squeeze()
p_pe_t_by_n = Pi[eta_dim, eta_dim] * (eta_samples - predictions) # precision weighted prediction errors (T x n_real)

no_bins = 50
freq_pred_error = np.empty([T_end_PP, no_bins])
_, bin_edges = np.histogram(p_pe_t_by_n, bins = no_bins)

for t in range(T_end_PP):
    freq_pred_error[t,:] = np.histogram(p_pe_t_by_n[t,:], bins= bin_edges)[0]

# #start figure
plt.figure()
plt.clf()
plt.title('Precision weighted prediction errors $\mathbf{\Pi}_{\eta}(\eta_t - \sigma(\mu_t))$',fontsize=16)

# #set up heatmap of prediction error paths

bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2.0
t_axis = range(T_end_PP)
# #plot heat map of prediction error paths
plt.contourf(t_axis, bin_centers, freq_pred_error.T, levels=100, cmap='Blues')

#plot sample prediction error path
handle = plt.plot(t_axis, p_pe_t_by_n[:T_end_PP,realisation_idx], color = 'darkorange',linewidth=0.8,label='Sample path')

#set axis labels and save
plt.xlabel('Time',fontsize=14)
plt.ylabel('$\mathbf{\Pi}_{\eta}(\eta_t - \sigma(\mu_t))$',fontsize=14)

plt.legend(loc='lower right',fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

if save_mode:
    figure_name = "Prediction_errors_time_3wayOU.png"
    plt.savefig(os.path.join(seed_folder,figure_name), dpi=100)
    plt.close()

'''
Figure 6: Plot evolving heatmap of probability density of blanket and internal states over time
'''

# # Run perturbation experiments with way more realizations but shorter time
# T, n_real = 150, 2 * 10**4

# _, key = random.split(key)

# b = 10. # specify perturbed blanket state 

# b_init = b * jnp.ones(n_real)

# #initialising external and internal states at posterior distributions
# mu_init = random.multivariate_normal(key, jnp.array([b_mu*b]), inv(Pi[mu_dim,mu_dim][...,None]), shape = (n_real,) ).squeeze()
# _, key = random.split(key)
# eta_init = random.multivariate_normal(key, jnp.array([b_eta*b]), inv(Pi[eta_dim,eta_dim][...,None]), shape = (n_real,) ).squeeze()
# _, key = random.split(key)

# x0 = jnp.stack( (eta_init, b_init, mu_init), axis = 0)

# # sample paths
# x = process.integrate(T, n_real, dt, x0, rng_key = key) # run simulation

# plt.figure(3)
# density_over_time, b_bin_centers, mu_bin_centers = plot_b_mu_evolving_density(x, b_dim, mu_dim, 
#                                     start_T = 0, end_T = T, forward_window = 5,
#                                     plot_average = True, plot_paths = True)
# if save_mode:
#     figure_name = "path_density_3wayOU.png"
#     plt.savefig(os.path.join(seed_folder,figure_name), dpi=100)
#     plt.close()


