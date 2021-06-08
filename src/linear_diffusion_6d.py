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

key = random.PRNGKey(1) # fix random seed for reproducibility

pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}  # setup matplotlib to use latex for output
plt.style.use('seaborn-white')

figures_folder = 'figures'
if not os.path.isdir(figures_folder):
    os.mkdir(figures_folder)
    
flow_parameters, stationary_stats, sync_mappings, dimensions = initialize_6d_OU()

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

b_mu, b_eta, sync = sync_mappings['b_eta'], sync_mappings['b_mu'], sync_mappings['sync']

dt, T, n_real = 0.01, 500, 100  # duration of single time-step, total number of integration timesteps, number of parallel paths to simulate

# start many trajectories at a really high free energy

# Setting up the initial condition
b = 10.0 * jnp.ones( len(b_dim) )  # initialize ensemble of perturbed blanket states

b_init = jnp.tile(b[...,None], (1, n_real) )
#initialising external and internal states at posterior distributions
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

sensory = jnp.linspace(jnp.min(x[:, s_dim, :]) - 1, jnp.max(x[:, s_dim, :]) + 0.5, 105)  # internal state-space points
active = jnp.linspace(jnp.min(x[:, a_dim, :]) - 1, jnp.max(x[:, a_dim, :]) + 0.5, 100)  # internal state-space points

Z = compute_Fboldmu_blanket_landscape(sensory, active, b_mu, S_part_inv)

real_idx = 0  # index of sample path / realization to show (between 0 and N)

plt.figure(1)
plt.clf()
plt.title('Free energy $F(b_t, \mathbf{\mu}_t)$')
plt.contourf(sensory, active, Z, levels=100, cmap='turbo')  # plotting the free energy landscape
plt.colorbar()
plt.xlabel('blanket state $b_1$')
plt.ylabel('blanket state $b_2$')
plot_hot_colourline(x[:, s_dim, real_idx].reshape(T), x[:, a_dim, real_idx].reshape(T), lw=0.5)
plt.text(s='$b_t$', x=x[1, s_dim, real_idx] - 1, y=x[1, a_dim, real_idx]+0.5, color='black',fontsize=16)
plt.savefig(os.path.join(figures_folder,"Sample_perturbed_6wayOU.png"),dpi=100)
plt.close()

'''
Figure 2: average F(b, bold mu) over trajectories
'''

blanket_hist = jnp.transpose(x[:,b_dim,:], (1, 0, 2))
F_trajectories = compute_Fboldmu_blanket_over_time(blanket_hist, b_mu, S_part_inv)
mean_F = F_trajectories.mean(axis=0)


plt.figure(2)
plt.clf()
plt.title('Average free energy over time')
plot_hot_colourline(np.arange(T), mean_F)
xlabel = int(T * 0.4)  # position of text on x axis
plt.text(s='$F(b_t, \mathbf{\mu}_t)$', x=xlabel, y=mean_F[xlabel] + 0.05 * (mean_F.max() - mean_F[xlabel]),
         color='black')
plt.xlabel('Time')
plt.ylabel('Free energy $F(b_t, \mathbf{\mu}_t)$')
plt.savefig(os.path.join(figures_folder,"FE_vs_time_perturbed_6wayOU.png"),dpi=100)
plt.close()

'''
Figure 3: Evoked response plots (showing how conditional distribution over external states - parameterised with variational mean and variance) changes
over time, locked to beginning of the blanket-state perturbation.
'''

T_fig= 200

mean_trajectory = x.mean(axis=2)
eta_paths = x[:T_fig,eta_dim,:]

posterior_means = sync.dot(mean_trajectory[:T_fig,mu_dim].T)
posterior_cov = inv(Pi[np.ix_(eta_dim,eta_dim)])
posterior_stds = sqrtm(posterior_cov)

# compute the marginal predictions with confidence intervals overlaid
conf_interval_param =1.96
pred_upper_CI_mu0 = posterior_means[0]+ conf_interval_param * posterior_stds[0,0]
pred_lower_CI_mu0 = posterior_means[0]- conf_interval_param * posterior_stds[0,0]

pred_upper_CI_mu1 = posterior_means[1] + conf_interval_param * posterior_stds[1,1]
pred_lower_CI_mu1 = posterior_means[1] - conf_interval_param * posterior_stds[1,1]

t_axis = np.arange(T_fig)
plt.figure(figsize=(19,12))
plt.clf()
plt.title('Prediction errors: $q_{\mathbf{\mu}_t}(\eta)$ vs $\eta_t$',fontsize=30, pad = 10)

plt.fill_between(t_axis,pred_upper_CI_mu0, pred_lower_CI_mu0, color='b', alpha=0.15)
eta1_real_line = plt.plot(t_axis, eta_paths[:,0,::5], lw = 0.5, color = 'r', alpha=0.35, label='Sample paths: $(\eta_{t})_1$')
mu1_mean_line = plt.plot(t_axis,posterior_means[0], color='b',label='Prediction: $q_{\mathbf{\mu}_t}(\eta_1)$',lw=3.0)

plot_offset = 10.0
plt.fill_between(t_axis,pred_upper_CI_mu1 + plot_offset, pred_lower_CI_mu1 + plot_offset, color='#3de3ac', alpha=0.25)
eta2_real_line = plt.plot(t_axis, eta_paths[:,1,::5] + plot_offset, lw = 0.5, color = '#933aab', alpha=0.35, label='Sample paths: $(\eta_{t})_2$')
mu2_mean_line = plt.plot(t_axis,posterior_means[1] + plot_offset, color='#3aab89',label='Prediction: $q_{\mathbf{\mu}_t}(\eta_2)$',lw=3.0)

ci_patch_1 = Patch(color='blue',alpha=0.1, label=' ')
ci_patch_2 = Patch(color='#3de3ac',alpha=0.25, label=' ')

first_legend = plt.legend(handles=[ci_patch_1], loc=(0.205,0.091), fontsize=28, ncol = 1)
second_legend = plt.legend(handles=[ci_patch_2], loc=(0.205,0.022), fontsize=28, ncol = 1)

# Add the legend manually to the current Axes.
plt.gca().add_artist(first_legend)
plt.gca().add_artist(second_legend)

plt.legend(handles=[mu1_mean_line[0],mu2_mean_line[0], eta1_real_line[0], eta2_real_line[0]], loc='lower center',fontsize=24, ncol = 2)

min_value = min( ( pred_lower_CI_mu0.min(), eta_paths[:,0,:].min() ) )
max_value = max( ( (pred_upper_CI_mu1 + plot_offset).max(), (eta_paths[:,1,:]+plot_offset).max() ) )

plt.xlim(t_axis[0], t_axis[-1])
plt.ylim(1.25 * min_value, 1.1 * max_value )

plt.gca().tick_params(axis='both', which='both', labelsize=25)
plt.gca().set_xlabel('Time',fontsize=30)
plt.gca().set_ylabel('External state space $\mathcal{E}$',fontsize=30)
plt.savefig(os.path.join(figures_folder,"average_prediction_blanket_perturbation.png"), dpi=100)


'''
Figure 4 - plot the centered bivariate predictions for external states, overlaid with the analytic covariance / confidence intervals
'''

x_reshaped = jnp.transpose(x, (1,0,2)).reshape(n_var, T * n_real) # unwrap realizations (third dimension) to make one long matrix

mu_flat = x_reshaped[mu_dim,:] # all realizations, for all timesteps, of internal states
eta_flat = x_reshaped[eta_dim,:] # all realizations, for all timesteps, of external states

q_mu = sync @ mu_flat # predicted external states, parameterised by instantaneous internal states

fig, ax = plt.subplots(figsize=(14,10))

plt.title('Prediction errors $\eta - \mathbf{\mu}$',fontsize=26, pad = 10)

prediction_errors = eta_flat - q_mu # prediction errors - difference between realization of external state and 'predicted' external state


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
dots = ax.scatter(x=prediction_errors[0,ind,0],y=prediction_errors[1,ind, 0],s=6,label='Prediction errors', c=ind+10**(-3),cmap=cm.hot)
for n in range(1,n_real,50):
    ax.scatter(x=prediction_errors[0,ind,n], y=prediction_errors[1,ind,n], s=6,c=ind+10**(-3), cmap=cm.hot)


#confidence ellipse

T_ss = int(2*T/3)# time period where steady-state (post-perturbation) is assumed

pred_errors_ss = prediction_errors[:,T_ss:,:].reshape([2, (T-T_ss)*n_real]) #prediction errors once steady-state is assumed

# prediction_centroid = pred_errors_ss.mean(axis=1) # mean over time should be [0.0, 0.0]

x_tick = np.linspace(1.1 * min_x, 1.1*max_x, 105)  # x axis points
y_tick = np.linspace(1.1 * min_y, 1.1 * max_y, 100)  # y axis points

X,Y = np.meshgrid(x_tick,y_tick)
pos = np.dstack((X, Y))
rv = multivariate_normal(cov= 1.2*np.cov(pred_errors_ss)) #random normal

# vals, vecs = eigsorted(posterior_cov)
# theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

# nstd = 3.0

# height, width = 2 * nstd * jnp.sqrt(vals)
# ellip = Ellipse(xy=prediction_centroid, width=width, height=height, angle=theta, alpha=0.1, color='blue')
# ax.add_artist(ellip)
# ci_patch = Patch(color='blue',alpha=0.1, label='Confidence ellipse')

# plt.legend(handles=[dots, ci_patch], loc='lower right',fontsize=20)

ax.contourf(X, Y, rv.pdf(pos)**(1/15), levels=0, colors =['white','blue'],alpha =0.25)

ci_patch = Patch(color='blue',alpha=0.4, label='Covariance at steady-state')

#plotting of figure elements

plt.legend(handles=[dots, ci_patch], loc='upper right',fontsize=20)
legend = ax.get_legend()
legend.legendHandles[0].set_color(cm.hot(0.01))

plt.gca().tick_params(axis='both', which='both', labelsize=25)
plt.gca().set_xlabel('$(\eta - \mathbf{\mu})_1$',fontsize=30)
plt.gca().set_ylabel('$(\eta - \mathbf{\mu})_2$',fontsize=30)
plt.savefig(os.path.join(figures_folder,"prediction_error_cov_plot.png"),dpi=100)

