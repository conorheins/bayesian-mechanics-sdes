import os
from diffusions import LinearProcess
from utilities import compute_Fboldmu_blanket_landscape, compute_Fboldmu_blanket_over_time, plot_hot_colourline, eigsorted
import jax.numpy as jnp
from jax.numpy.linalg import inv
from jax import random
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
import matplotlib.lines as mlines
import matplotlib.cm as cm

from configs.config_6d import initialize_6d_OU

# initialization_key = 5    # default configuration in `config_6d.py` file if no key is passed, lots of solenoidal flow / oscillations. Non-monotonic FE descent

# other interesting seeds (that I know work - by which I mean the initialization conditions don't break due to inequality of Sylvester relation)
# initialization_key = 16   
# initialization_key = 14   
# initialization_key = 13   
# initialization_key = 11   
initialization_key = 4  # decent one, some solenoidal flow but not too much
# initialization_key = 3  # decent one
# initialization_key = 2  # not too much solenoidal, nearly straight descent to VFE minimum

# fix random seed for reproducibility
key = random.PRNGKey(initialization_key)   

save_mode = True

pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}  # setup matplotlib to use latex for output
plt.style.use('seaborn-white')

figures_folder = 'figures'
if not os.path.isdir(figures_folder):
    os.mkdir(figures_folder)

figures_folder = os.path.join(figures_folder, '6d_inference')
if not os.path.isdir(figures_folder):
    os.mkdir(figures_folder)

seed_folder = os.path.join(figures_folder, f'seed_{initialization_key}')
if not os.path.isdir(seed_folder):
    os.mkdir(seed_folder)

flow_parameters, stationary_stats, sync_mappings, dimensions = initialize_6d_OU(rng_key = key)

n_var = 6      # dimensionality

eta_dim = dimensions['eta']
b_dim = dimensions['b']
mu_dim = dimensions['mu']
pi_dim = dimensions['pi']

Pi, S = stationary_stats['Pi'], stationary_stats['S']

# Setting up the OU process
process = LinearProcess(dim=n_var, friction=flow_parameters['B'], volatility=flow_parameters['sigma'])  # create process

'''
OU process perturbed blanket states simulation
'''

b_eta = sync_mappings['b_eta']
b_mu  = sync_mappings['b_mu']
sync  = sync_mappings['sync']

dt, T, n_real = 0.01, 500, 100  # duration of single time-step, total number of integration timesteps, number of parallel paths to simulate

# start many trajectories at a really high free energy

# Setting up the initial condition
b = 10.0 * jnp.ones( len(b_dim) )  # initialize ensemble of perturbed blanket states

b_init = jnp.tile(b[...,None], (1, n_real) )

#initialising external and internal states at conditional distributions given the perturbed/surprising blanket state
mu_init = random.multivariate_normal(key, b_mu @ b, inv(Pi[np.ix_(mu_dim,mu_dim)]), shape = (n_real,) ).T
_, key = random.split(key)

eta_init = random.multivariate_normal(key, b_eta @ b, inv(Pi[np.ix_(eta_dim,eta_dim)]), shape = (n_real,) ).T
_, key = random.split(key)

x0 = jnp.concatenate( (eta_init, b_init, mu_init), axis = 0)

# sample paths
x = process.integrate(T, n_real, dt, x0, rng_key = key) # run simulation

'''
Figure 1: Heat map of F(b, bold mu) and sample path
'''

s_dim, a_dim = dimensions['s'], dimensions['a']
S_part_inv = inv(S[np.ix_(pi_dim, pi_dim)])

sensory = jnp.linspace(jnp.min(x[:, s_dim, :]) - 1, jnp.max(x[:, s_dim, :]) + 0.5, 105)  # sensory state-space points
active = jnp.linspace(jnp.min(x[:, a_dim, :]) - 1, jnp.max(x[:, a_dim, :]) + 0.5, 100)  # active state-space points

Z = compute_Fboldmu_blanket_landscape(sensory, active, b_mu, S_part_inv)

# real_idx = random.randint(key, shape=(), minval = 0, maxval = n_real)  # which sample path to show (between 0 and n_real)
# real_idx = 20  # this will be in the index of the sample path if the `initialization_key` is 4
real_idx = 25  #  hand-picked realization

print(f'Sample path index being shown: {real_idx}\n')

# plt.figure(figsize=(14,10))
plt.figure(1)
plt.clf()
plt.title('Free energy $F(b, \mathbf{\mu})$',fontsize=16)
plt.contourf(sensory, active, Z, levels=100, cmap='turbo')  # plotting the free energy landscape
plt.colorbar()
plt.xlabel('blanket state $b_1$',fontsize=14)
plt.ylabel('blanket state $b_2$',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plot_hot_colourline(x[:, s_dim, real_idx].reshape(T), x[:, a_dim, real_idx].reshape(T), lw=0.5)
plt.text(s='$b_t$', x=x[1, s_dim, real_idx] - 1, y=x[1, a_dim, real_idx]+0.5, color='black',fontsize=16)
if save_mode:
    figure_name = f'Sample_perturbed_6wayOU.png'
    plt.savefig(os.path.join(seed_folder,figure_name),dpi=100)
    plt.close()

'''
Figure 2: average time-varying F(b, bold mu), with average taken across trajectories
'''

# vectorized computation of F(s, a, boldmu) over time
blanket_hist = jnp.transpose(x[:,b_dim,:], (1, 0, 2))
F_trajectories = compute_Fboldmu_blanket_over_time(blanket_hist, b_mu, S_part_inv)
mean_F = F_trajectories.mean(axis=0)

# plt.figure(figsize=(14,10))
plt.figure(2)
plt.clf()
plt.title('Average free energy over time',fontsize=16)
plot_hot_colourline(np.arange(T), mean_F)
xlabel = int(T * 0.4)  # position of text on x axis
plt.text(s='$F(b_t, \mathbf{\mu}_t)$', x=xlabel, y=mean_F[xlabel] + 0.05 * (mean_F.max() - mean_F[xlabel]),
         color='black',fontsize=16)
plt.xlabel('Time',fontsize=14)
plt.ylabel('Free energy $F(b_t, \mathbf{\mu}_t)$',fontsize=14)

# plt.autoscale(enable=True, axis='x', tight=True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

if save_mode:
    figure_name = f'FE_vs_time_perturbed_6wayOU.png'
    plt.savefig(os.path.join(seed_folder,figure_name),dpi=100)
    plt.close()

'''
Figure 4a: Evoked response plots (showing how conditional distribution over external states - parameterised with variational mean and variance) changes
over time, locked to beginning of the blanket-state perturbation. 
SINGLE PATH VARIANT 1 -- EXPECTED EXTERNAL PARAMETERISED BY BLANKET STATES PATH
'''

T_end_PP= 200 # up to what timestep to show for the predictive processing simulations 

b_path = x[:T_end_PP,b_dim, real_idx ].squeeze() # Lance's addition, now we use sigma(bold_mu(b)) for the mapping instead of the realized internal state

eta_path = x[:T_end_PP,eta_dim, real_idx ].squeeze()

posterior_means = b_eta.dot(b_path.T) # Lance's addition, now we use sigma(bold_mu(b)) for the mapping instead of the realized internal state

posterior_cov = inv(Pi[np.ix_(eta_dim,eta_dim)])

# compute the marginal predictions with confidence intervals overlaid
conf_interval_param = 1.96

std_mu_0 = jnp.sqrt(posterior_cov[0,0])
pred_upper_CI_mu0 = posterior_means[0]+ conf_interval_param * std_mu_0
pred_lower_CI_mu0 = posterior_means[0]- conf_interval_param * std_mu_0

std_mu_1 = jnp.sqrt(posterior_cov[1,1])
pred_upper_CI_mu1 = posterior_means[1] + conf_interval_param * std_mu_1
pred_lower_CI_mu1 = posterior_means[1] - conf_interval_param * std_mu_1

t_axis = np.arange(T_end_PP)

plt.clf()
plt.title('Predictive processing: $q_{\mathbf{\mu}_t}(\eta)$ vs $\eta_t$',fontsize=16, pad = 10)

plt.fill_between(t_axis,pred_upper_CI_mu0, pred_lower_CI_mu0, color='b', alpha=0.15)
eta1_real_line = plt.plot(t_axis, eta_path[:,0], lw = 1.1, color = 'r', alpha=0.6, label='External: $(\eta_{t})_1$')
mu1_mean_line = plt.plot(t_axis,posterior_means[0], color='b',label='Prediction: $q_{\mathbf{\mu}_t}(\eta_1)$',lw=3.0)

maximum_bottom =  max(posterior_means[0].max(), eta_path[:,0].max()) # find max value of either the prediction or the eta realization(s)
range_top = max(posterior_means[1].max(), eta_path[:,1].max()) - min(posterior_means[1].min(), eta_path[:,1].min()) # find the range of the second variate (between both the prediction and the realisation)
plot_offset = maximum_bottom + range_top

plt.fill_between(t_axis,pred_upper_CI_mu1 + plot_offset, pred_lower_CI_mu1 + plot_offset, color='#3de3ac', alpha=0.25)
eta2_real_line = plt.plot(t_axis, eta_path[:,1] + plot_offset, lw = 1.1, color = '#933aab', alpha=0.8, label='External: $(\eta_{t})_2$')
mu2_mean_line = plt.plot(t_axis,posterior_means[1] + plot_offset, color='#3aab89',label='Prediction: $q_{\mathbf{\mu}_t}(\eta_2)$',lw=3.0)

ci_patch_1 = Patch(color='blue',alpha=0.1, label=' ')
ci_patch_2 = Patch(color='#3de3ac',alpha=0.25, label=' ')

min_value = min( ( pred_lower_CI_mu0.min(), eta_path[:,0].min() ) )
max_value = max( ( (pred_upper_CI_mu1 + plot_offset).max(), (eta_path[:,1]+plot_offset).max() ) )
total = max_value-min_value

plt.xlim(t_axis[0], t_axis[-1])
plt.ylim(min_value - 0.5*total, max_value + 0.1*total)

first_legend = plt.legend(handles=[ci_patch_1], loc=(0.102,0.115), fontsize=14, ncol = 1)
second_legend = plt.legend(handles=[ci_patch_2], loc=(0.102,0.0318), fontsize=14, ncol = 1)

# Add the legend manually to the current Axes.
plt.gca().add_artist(first_legend)
plt.gca().add_artist(second_legend)

plt.legend(handles=[mu1_mean_line[0],mu2_mean_line[0], eta1_real_line[0], eta2_real_line[0]], loc='lower center',fontsize=12, ncol = 2)

plt.gca().tick_params(axis='both', which='both', labelsize=14)
plt.gca().set_xlabel('Time',fontsize=14)
plt.gca().set_ylabel('External state-space $\mathcal{E}$',fontsize=14)

if save_mode:
    figure_name = f'singlepath_prediction_blanket_perturbation_bpath.png'
    plt.savefig(os.path.join(seed_folder,figure_name), dpi=100)
    plt.close()

'''
Figure 4b: Evoked response plots (showing how conditional distribution over external states - parameterised with variational mean and variance) changes
over time, locked to beginning of the blanket-state perturbation. 
SINGLE PATH VARIANT 2 -- EXPECTED EXTERNAL PARAMETERISED BY INTERNAL STATES PATH
'''
T_end_PP= 200 # up to what timestep to show for the predictive processing simulations 

mu_path = x[:T_end_PP,b_dim, real_idx ].squeeze()

eta_path = x[:T_end_PP,eta_dim, real_idx ].squeeze()

posterior_means = sync.dot(mu_path.T) # my edit to Lance's version, looks better for default seed (with lots of solenoidal flow) if we use sigma(mu_t) for the mapping instead of the realized blanket states

posterior_cov = inv(Pi[np.ix_(eta_dim,eta_dim)])

# compute the marginal predictions with confidence intervals overlaid
conf_interval_param = 1.96
std_mu_0 = jnp.sqrt(posterior_cov[0,0])
pred_upper_CI_mu0 = posterior_means[0]+ conf_interval_param * std_mu_0
pred_lower_CI_mu0 = posterior_means[0]- conf_interval_param * std_mu_0

std_mu_1 = jnp.sqrt(posterior_cov[1,1])
pred_upper_CI_mu1 = posterior_means[1] + conf_interval_param * std_mu_1
pred_lower_CI_mu1 = posterior_means[1] - conf_interval_param * std_mu_1

t_axis = np.arange(T_end_PP)
plt.clf()
plt.title('Predictive processing: $q_{\mathbf{\mu}_t}(\eta)$ vs $\eta_t$',fontsize=16, pad = 10)

plt.fill_between(t_axis,pred_upper_CI_mu0, pred_lower_CI_mu0, color='b', alpha=0.15)
eta1_real_line = plt.plot(t_axis, eta_path[:,0], lw = 1.1, color = 'r', alpha=0.6, label='External: $(\eta_{t})_1$')
mu1_mean_line = plt.plot(t_axis,posterior_means[0], color='b',label='Prediction: $q_{\mathbf{\mu}_t}(\eta_1)$',lw=3.0)

maximum_bottom =  max(posterior_means[0].max(), eta_path[:,0].max()) # find max value of either the prediction or the eta realization(s)
range_top = max(posterior_means[1].max(), eta_path[:,1].max()) - min(posterior_means[1].min(), eta_path[:,1].min()) # find the range of the second variate (between both the prediction and the realisation)
plot_offset = maximum_bottom + range_top

plt.fill_between(t_axis,pred_upper_CI_mu1 + plot_offset, pred_lower_CI_mu1 + plot_offset, color='#3de3ac', alpha=0.25)
eta2_real_line = plt.plot(t_axis, eta_path[:,1] + plot_offset, lw = 1.1, color = '#933aab', alpha=0.8, label='External: $(\eta_{t})_2$')
mu2_mean_line = plt.plot(t_axis,posterior_means[1] + plot_offset, color='#3aab89',label='Prediction: $q_{\mathbf{\mu}_t}(\eta_2)$',lw=3.0)

ci_patch_1 = Patch(color='blue',alpha=0.1, label=' ')
ci_patch_2 = Patch(color='#3de3ac',alpha=0.25, label=' ')

min_value = min( ( pred_lower_CI_mu0.min(), eta_path[:,0].min() ) )
max_value = max( ( (pred_upper_CI_mu1 + plot_offset).max(), (eta_path[:,1]+plot_offset).max() ) )
total = max_value-min_value

plt.xlim(t_axis[0], t_axis[-1])
plt.ylim(min_value - 0.5*total, max_value + 0.1*total)

first_legend = plt.legend(handles=[ci_patch_1], loc=(0.019,0.163), fontsize=14, ncol = 1)
second_legend = plt.legend(handles=[ci_patch_2], loc=(0.019,0.0515), fontsize=14, ncol = 1)

# Add the legend manually to the current Axes.
plt.gca().add_artist(first_legend)
plt.gca().add_artist(second_legend)

plt.legend(handles=[mu1_mean_line[0],mu2_mean_line[0], eta1_real_line[0], eta2_real_line[0]], loc='lower center',fontsize=14, ncol = 2)

plt.gca().tick_params(axis='both', which='both', labelsize=14)
plt.gca().set_xlabel('Time',fontsize=14)
plt.gca().set_ylabel('External state-space $\mathcal{E}$',fontsize=14)


if save_mode:
    figure_name = f'singlepath_prediction_blanket_perturbation_mupath.png'
    plt.savefig(os.path.join(seed_folder,figure_name), dpi=100)
    plt.close()

'''
Figure 4: Evoked response plots (showing how conditional distribution over external states - parameterised with variational mean and variance) changes
over time, locked to beginning of the blanket-state perturbation. 
ENSEMBLE AVERAGE VARIANT. Average internal states across realizations and map average path through sigma(<mu_t>). Overlays multiple external states paths.
'''

T_end_PP= 200 # up to what timestep to show for the predictive processing simulations 

mean_trajectory = x.mean(axis=2)
eta_paths = x[:T_end_PP,eta_dim,:]
posterior_means = sync.dot(mean_trajectory[:T_end_PP,mu_dim].T)

posterior_cov = inv(Pi[np.ix_(eta_dim,eta_dim)])

# compute the marginal predictions with confidence intervals overlaid
conf_interval_param = 1.96
std_mu_0 = jnp.sqrt(posterior_cov[0,0])
pred_upper_CI_mu0 = posterior_means[0]+ conf_interval_param * std_mu_0
pred_lower_CI_mu0 = posterior_means[0]- conf_interval_param * std_mu_0

std_mu_1 = jnp.sqrt(posterior_cov[0,0])
pred_upper_CI_mu1 = posterior_means[1] + conf_interval_param * std_mu_1
pred_lower_CI_mu1 = posterior_means[1] - conf_interval_param * std_mu_1

t_axis = np.arange(T_end_PP)
plt.clf()
plt.title('Predictive processing: $q_{\mathbf{\mu}_t}(\eta)$ vs $\eta_t$',fontsize=16, pad = 10)

show_every = 5

plt.fill_between(t_axis,pred_upper_CI_mu0, pred_lower_CI_mu0, color='b', alpha=0.15)
eta1_real_line = plt.plot(t_axis, eta_paths[:,0,::show_every], lw = 1.1, color = 'r', alpha=0.35, label='External: $(\eta_{t})_1$')
mu1_mean_line = plt.plot(t_axis,posterior_means[0], color='b',label='Prediction: $q_{\mathbf{\mu}_t}(\eta_1)$',lw=3.0)

maximum_bottom =  max(posterior_means[0].max(), eta_paths[:,0,::show_every].max())
range_top = max(posterior_means[1].max(), eta_paths[:,1,::show_every].max()) - min(posterior_means[1].min(), eta_paths[:,1,::show_every].min())
plot_offset = maximum_bottom + range_top

plt.fill_between(t_axis,pred_upper_CI_mu1 + plot_offset, pred_lower_CI_mu1 + plot_offset, color='#3de3ac', alpha=0.25)
eta2_real_line = plt.plot(t_axis, eta_paths[:,1,::show_every] + plot_offset, lw = 1.1, color = '#933aab', alpha=0.35, label='External: $(\eta_{t})_2$')
mu2_mean_line = plt.plot(t_axis,posterior_means[1] + plot_offset, color='#3aab89',label='Prediction: $q_{\mathbf{\mu}_t}(\eta_2)$',lw=3.0)

ci_patch_1 = Patch(color='blue',alpha=0.1, label=' ')
ci_patch_2 = Patch(color='#3de3ac',alpha=0.25, label=' ')

min_value = min( ( pred_lower_CI_mu0.min(), eta_path[:,0].min() ) )
max_value = max( ( (pred_upper_CI_mu1 + plot_offset).max(), (eta_path[:,1]+plot_offset).max() ) )
total = max_value-min_value

plt.xlim(t_axis[0], t_axis[-1])
plt.ylim(min_value - 0.5*total, max_value + 0.1*total)

first_legend = plt.legend(handles=[ci_patch_1], loc=(0.011,0.17), fontsize=15, ncol = 1)
second_legend = plt.legend(handles=[ci_patch_2], loc=(0.011,0.048), fontsize=15, ncol = 1)

# Add the legend manually to the current Axes.
plt.gca().add_artist(first_legend)
plt.gca().add_artist(second_legend)

plt.legend(handles=[mu1_mean_line[0],mu2_mean_line[0], eta1_real_line[0], eta2_real_line[0]], loc='lower center',fontsize=14, ncol = 2)

plt.gca().tick_params(axis='both', which='both', labelsize=14)
plt.gca().set_xlabel('Time',fontsize=14)
plt.gca().set_ylabel('External state space $\mathcal{E}$',fontsize=14)


if save_mode:
    figure_name = f'average_prediction_blanket_perturbation.png'
    plt.savefig(os.path.join(seed_folder,figure_name), dpi=100)
    plt.close()

'''
Figure 5 - plot the centered bivariate predictions for external states, overlaid with the analytic covariance / confidence intervals
VARIANT 1: PARAMETERISED BY SIGMA(BOLDMU(B_T)) OF RANDOM BLANKET STATE PATH
'''

x_reshaped = jnp.transpose(x, (1,0,2)).reshape(n_var, T * n_real) # unwrap realizations (third dimension) to make one long matrix

b_flat = x_reshaped[b_dim,:] # all realizations, for all timesteps, of blanket states
eta_flat = x_reshaped[eta_dim,:] # all realizations, for all timesteps, of external states
prediction = b_eta @ b_flat # predicted external states, parameterised by boldmu of instantaneous blanket states

# fig, ax = plt.subplots(figsize=(14,10))
plt.figure()
plt.title('Precision-weighted prediction errors $\mathbf{\Pi}_{\eta}(\eta_t - \sigma(\mathbf{\mu}_t))$',fontsize=16, pad = 10)
prediction_errors = Pi[np.ix_(eta_dim, eta_dim)] @ (eta_flat - prediction) # precision-weighted prediction errors - difference between realization of external state and 'predicted' external state

#axis limits

min_x = prediction_errors[0].min()
max_x = prediction_errors[0].max()
plt.xlim(1.1 * min_x, 1.1 * max_x)
min_y = prediction_errors[1].min()
max_y = prediction_errors[1].max()
plt.ylim(1.35 * min_y, 1.1 * max_y)

#scatter plot

prediction_errors = prediction_errors.reshape([2,T,n_real])

ind = np.arange(0,T) #indices to plot
dots = plt.scatter(x=prediction_errors[0,ind,0],y=prediction_errors[1,ind, 0],s=6,label='Precision-weighted prediction errors', c=ind+10**(-3),cmap=cm.hot)
for n in range(1,n_real,50):
    plt.scatter(x=prediction_errors[0,ind,n], y=prediction_errors[1,ind,n], s=6,c=ind+10**(-3), cmap=cm.hot)

#confidence ellipse

T_ss = int(2*T/3)# time period where steady-state (post-perturbation) is assumed
pred_errors_ss = prediction_errors[:,T_ss:,:].reshape([2, (T-T_ss)*n_real]) #prediction errors once steady-state is assumed

x_tick = np.linspace(1.1 * min_x, 1.1*max_x, 105)  # x axis points
y_tick = np.linspace(1.1 * min_y, 1.1 * max_y, 100)  # y axis points

X,Y = np.meshgrid(x_tick,y_tick)
pos = np.dstack((X, Y))
rv = multivariate_normal(cov= 1.2*np.cov(pred_errors_ss)) #random normal

plt.contourf(X, Y, rv.pdf(pos)**(1/15), levels=0, colors =['white','blue'],alpha =0.25)

ci_patch = Patch(color='blue',alpha=0.4, label='Covariance at steady-state')

#plotting of figure elements


plt.legend(handles=[dots, ci_patch], loc='lower right',fontsize=14)
legend = plt.gca().get_legend()
legend.legendHandles[0].set_color(cm.hot(0.01))

plt.gca().tick_params(axis='both', which='both', labelsize=14)

# plt.gca().set_xlabel('$\mathbf{\epsilon}_1$',fontsize=30)
# plt.gca().set_ylabel('$\mathbf{\epsilon}_2$',fontsize=30)

plt.gca().set_xlabel(r'$\xi_1$',fontsize=14)
plt.gca().set_ylabel(r'$\xi_2$',fontsize=14)

if save_mode:
    figure_name = f'ppe_cov_plot_boldmu_b_t.png'
    plt.savefig(os.path.join(seed_folder,figure_name),dpi=100)
    plt.close()

'''
Figure 6 - plot the centered bivariate predictions for external states, overlaid with the analytic covariance / confidence intervals
VARIANT 1: PARAMETERISED BY SIGMA(MU_T) OF RANDOM INTERNAL STATE PATH
'''

mu_flat = x_reshaped[mu_dim,:] # all realizations, for all timesteps, of internal states
eta_flat = x_reshaped[eta_dim,:] # all realizations, for all timesteps, of external states
prediction = sync @ mu_flat # predicted external states, parameterised by instantaneous internal states
fig, ax = plt.subplots(figsize=(14,10))
plt.title('Precision-weighted prediction errors $\mathbf{\Pi}_{\eta}(\eta_t - \sigma(\mathbf{\mu}_t))$',fontsize=26, pad = 10)
prediction_errors = Pi[np.ix_(eta_dim, eta_dim)] @ (eta_flat - prediction) # precision-weighted prediction errors - difference between realization of external state and 'predicted' external state

#axis limits

min_x = prediction_errors[0].min()
max_x = prediction_errors[0].max()
plt.xlim(1.1 * min_x, 1.1 * max_x)
min_y = prediction_errors[1].min()
max_y = prediction_errors[1].max()
plt.ylim(1.1 * min_y, 1.1 * max_y)

#scatter plot

prediction_errors = prediction_errors.reshape([2,T,n_real])

ind = np.arange(0,T) #indices to plot
dots = ax.scatter(x=prediction_errors[0,ind,0],y=prediction_errors[1,ind, 0],s=6,label='Precision-weighted prediction errors', c=ind+10**(-3),cmap=cm.hot)
for n in range(1,n_real,50):
    ax.scatter(x=prediction_errors[0,ind,n], y=prediction_errors[1,ind,n], s=6,c=ind+10**(-3), cmap=cm.hot)

#confidence ellipse

T_ss = int(2*T/3)# time period where steady-state (post-perturbation) is assumed
pred_errors_ss = prediction_errors[:,T_ss:,:].reshape([2, (T-T_ss)*n_real]) #prediction errors once steady-state is assumed

x_tick = np.linspace(1.1 * min_x, 1.1*max_x, 105)  # x axis points
y_tick = np.linspace(1.1 * min_y, 1.1 * max_y, 100)  # y axis points

X,Y = np.meshgrid(x_tick,y_tick)
pos = np.dstack((X, Y))
rv = multivariate_normal(cov= 1.2*np.cov(pred_errors_ss)) #random normal

ax.contourf(X, Y, rv.pdf(pos)**(1/15), levels=0, colors =['white','blue'],alpha =0.25)

ci_patch = Patch(color='blue',alpha=0.4, label='Covariance at steady-state')

#plotting of figure elements

plt.legend(handles=[dots, ci_patch], loc='lower right',fontsize=20)
legend = ax.get_legend()
legend.legendHandles[0].set_color(cm.hot(0.01))

plt.gca().tick_params(axis='both', which='both', labelsize=25)

plt.gca().set_xlabel(r'$\xi_1$',fontsize=30)
plt.gca().set_ylabel(r'$\xi_2$',fontsize=30)

if save_mode:
    figure_name = f'ppe_cov_plot_sigma_mu_t.png'
    plt.savefig(os.path.join(seed_folder,figure_name),dpi=100)
    plt.close()
