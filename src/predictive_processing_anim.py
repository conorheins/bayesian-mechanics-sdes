import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import jax.numpy as jnp
from jax import random

from diffusions import LinearProcess
from configs.config_3d import initialize_3d_OU

pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}  # setup matplotlib to use latex for output
plt.style.use('seaborn-white')

initialization_key = 1 # choose a key to fix the random seed for reproducibility

flow_parameters, stationary_stats, sync_mappings, dimensions = initialize_3d_OU(rng_key = 'default') # default parameterisation, gives a pre-defined steady-state / precision matrix

n_var = 3       # dimensionality

# Setting up the OU process
process = LinearProcess(dim=n_var, friction=flow_parameters['B'], volatility=flow_parameters['sigma'])  # create process

eta_dim = dimensions['eta']
b_dim = dimensions['b']
mu_dim = dimensions['mu']


dt = 0.01     # duration of single time-step 
T = 250       # total number of integration timesteps,
n_real = 1    # number of parallel paths to simulate

S, Pi = stationary_stats['S'], stationary_stats['Pi']

key = random.PRNGKey(initialization_key)

b_eta, b_mu, sync = sync_mappings['b_eta'], sync_mappings['b_mu'], sync_mappings['sync']

b = -7.5 # specify perturbed blanket state, use this one when doing sigma(boldmu(b_t))

b_init = b * jnp.ones(n_real)

#initialising external and internal states at posterior distributions
mu_init = random.multivariate_normal(key, jnp.array([b_mu*b]), Pi[mu_dim,mu_dim][...,None]**-1, shape = (n_real,) )
_, key = random.split(key)
eta_init = random.multivariate_normal(key, jnp.array([b_eta*b]), Pi[eta_dim,eta_dim][...,None]**-1, shape = (n_real,) )
_, key = random.split(key)

x0 = jnp.stack( (eta_init[0], b_init, mu_init[0]), axis = 0)

# integrate the stochastic differential equation
x = process.integrate(T, n_real, dt, x0, rng_key = key) # run simulation

x = x.squeeze() # we can squeeze out the last dimension since there's only one sample path

pred_mean = b_eta * x[:,b_dim].squeeze() # most likely external state, given blanket states
pred_std = jnp.sqrt(stationary_stats['Pi'][eta_dim,eta_dim]**-1) # standard deviation of belief

upper_conf_prediction = pred_mean + 1.96 * pred_std # pseudo Bayesian credible intervals
lower_conf_prediction = pred_mean - 1.96 * pred_std # pseudo Bayesian credible intervals

min_axes1_yvalue = 1.1 * jnp.array([lower_conf_prediction, x[:,eta_dim].squeeze()]).min()
max_axes1_yvalue = 1.1 * jnp.array([upper_conf_prediction, x[:,eta_dim].squeeze()]).max()

past_tstep = 10
lwidths = jnp.linspace(0,10,past_tstep)
colors = cm.Blues(jnp.linspace(0,1,past_tstep))
t_axis = jnp.arange(T)


fig = plt.figure(figsize=(12,20))

ax0 = fig.add_subplot(2,1,1,projection="3d")
ax1 = fig.add_subplot(2,1,2)

axes = [ax0, ax1]

for ax in axes:
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    ax.tick_params(bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    

def animate(i):
    axes[0].cla() # clear the previous image

    axes[1].cla() # clear the previous image

    start_idx = max(0, i - past_tstep)

    eta = x[start_idx:i, eta_dim].squeeze()
    mu = x[start_idx:i, mu_dim].squeeze()
    b = x[start_idx:i, b_dim].squeeze()
    points = jnp.array([eta, mu, b]).T.reshape(-1,1,3)
    segments = jnp.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, linewidths=lwidths,color='blue', colors = colors)

    axes[0].add_collection(lc)
    axes[0].set_xlim3d([x[:,eta_dim].min(), x[:,eta_dim].max()])
    axes[0].set_ylim3d([x[:,mu_dim].min(), x[:,mu_dim].max()])
    axes[0].set_zlim3d([x[:,b_dim].min(), x[:,b_dim].max()])
    axes[0].set_xlabel('$\eta$',fontsize=50,labelpad = 25)
    axes[0].tick_params(axis='x', labelsize=18)
    axes[0].set_ylabel('$\mu$',fontsize=50,labelpad = 25)
    axes[0].tick_params(axis='y', labelsize=18)
    axes[0].set_zlabel('b',fontsize=50,labelpad = 25, rotation=0)
    axes[0].tick_params(axis='z', labelsize=18)
     
    axes[1].fill_between(t_axis[:i],upper_conf_prediction[:i], lower_conf_prediction[:i], color='b', alpha=0.15)
    axes[1].plot(t_axis[:i],pred_mean[:i], color='b',lw=3.0,label='Prediction: $\sigma(\mathbf{\mu}(b_t))$')
    axes[1].plot(t_axis[:i], x[:i,eta_dim].squeeze(),color='r',label='External: $\eta_t$')

    axes[1].set_xlim([0.0, T])
    axes[1].set_ylim([min_axes1_yvalue, max_axes1_yvalue])
    axes[1].set_xlabel('$t$',fontsize=50,labelpad = 25)
    axes[1].tick_params(axis='x', labelsize=18)
    axes[1].set_ylabel('$\eta$',fontsize=50,labelpad = 25)
    axes[1].tick_params(axis='y', labelsize=18)
    axes[1].legend(fontsize=30,loc='upper right')

anim = animation.FuncAnimation(fig, animate, frames = T, interval = 1, blit = False)
# plt.show()

figures_folder = 'figures'
if not os.path.isdir(figures_folder):
    os.mkdir(figures_folder)

anim.save(os.path.join(figures_folder,'preditive_processing.gif'),fps=7.5)
