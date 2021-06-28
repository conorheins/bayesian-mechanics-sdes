import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

import jax.numpy as jnp
from jax import random

from diffusions import LinearProcess
from utilities import compute_FE_landscape, compute_F_over_time, parse_command_line
from configs.config_3d import initialize_3d_OU

pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}  # setup matplotlib to use latex for output
plt.style.use('seaborn-white')

key, save_mode = parse_command_line(default_key = 1)

flow_parameters, stationary_stats, sync_mappings, dimensions = initialize_3d_OU(rng_key = 'default') # default parameterisation, gives a pre-defined steady-state / precision matrix

n_var = 3       # dimensionality

# Setting up the OU process
process = LinearProcess(dim=n_var, friction=flow_parameters['B'], volatility=flow_parameters['sigma'])  # create process

eta_dim = dimensions['eta']
b_dim = dimensions['b']
mu_dim = dimensions['mu']
pi_dim = dimensions['pi']


dt = 0.01     # duration of single time-step 
T = 250       # total number of integration timesteps,
n_real = 1    # number of parallel paths to simulate

S, Pi = stationary_stats['S'], stationary_stats['Pi']


b_eta, b_mu, sync = sync_mappings['b_eta'], sync_mappings['b_mu'], sync_mappings['sync']

b = -10.0 # specify perturbed blanket state, use this one when doing sigma(boldmu(b_t))

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

S_part_inv = jnp.linalg.inv(S[jnp.ix_(pi_dim, pi_dim)]) # stationary covariance of particular states

mu_domain = jnp.linspace(jnp.min(x[:, mu_dim]) - 1, jnp.max(x[:, mu_dim]) + 0.5, 105)  # internal state-space points
b_domain = jnp.linspace(jnp.min(x[:, b_dim]) - 1, jnp.max(x[:, b_dim]) + 0.5, 100)  # blanket state-space points

F_landscape = compute_FE_landscape(b_domain, mu_domain, b_eta, sync, S_part_inv, Pi[eta_dim, eta_dim])

colormap2use = plt.cm.get_cmap('YlGnBu_r')

potential_over_time =  ((x[:,pi_dim] @ S_part_inv) * x[:,pi_dim]).sum(axis=1) / 2.0 # log potential term over time
KL_over_time =  Pi[eta_dim,eta_dim] * (sync * x[:,mu_dim] - b_eta * x[:,b_dim]) ** 2 / 2.0 # KL divergence term over time and realizations
F_over_time = potential_over_time + KL_over_time.squeeze()

max_axes1_yvalue = 1.1 * F_over_time.max()

past_tstep = 10
lwidths = jnp.linspace(0,10,past_tstep)
colors = cm.Reds(jnp.linspace(0,1,past_tstep))

t_axis = jnp.arange(T)

fig = plt.figure(figsize=(12,20))

ax0 = fig.add_subplot(2,1,1)
ax1 = fig.add_subplot(2,1,2)

divider = make_axes_locatable(ax0)
cax = divider.append_axes('right', size='5%', pad=0.05)

axes = [ax0, ax1]

ax0.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax0.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
ax0.tick_params(bottom=False)
for spine in ax0.spines.values():
    spine.set_visible(False)

def animate(i):
    axes[0].cla() # clear the previous image
    axes[1].cla() # clear the previous image

    im = axes[0].contourf(b_domain, mu_domain, F_landscape, levels=100, cmap=colormap2use)

    cb = fig.colorbar(im, cax=cax, orientation='vertical', ticks = np.round(np.arange(0,400,step=25)))
    cb.ax.tick_params(labelsize=22)

    start_idx = max(0, i - past_tstep)

    b = x[start_idx:i, b_dim].squeeze()
    mu = x[start_idx:i, mu_dim].squeeze()
    
    points = jnp.array([b, mu]).T.reshape(-1,1,2)
    segments = jnp.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, linewidths=lwidths,colors = colors)

    axes[0].add_collection(lc)
    axes[0].set_xlabel('$b$',fontsize=50,labelpad = 25)
    axes[0].tick_params(axis='x', labelsize=22)
    axes[0].set_ylabel('$\mu$',fontsize=50,labelpad = 25)
    axes[0].tick_params(axis='y', labelsize=22)
    axes[0].set_title('Free energy $F(b, \mu)$',fontsize=50, pad = 35)


    axes[1].plot(t_axis[:i],F_over_time[:i], color='r',lw=2.0, label='Free energy: $F(b_t, \mu_t)$')

    axes[1].set_xlim([0.0, 150])
    axes[1].set_ylim([0, max_axes1_yvalue])
    axes[1].set_xlabel('$t$',fontsize=50,labelpad = 25)
    axes[1].tick_params(axis='x', labelsize=22)
    axes[1].set_ylabel('$F(b, \mu)$',fontsize=50,labelpad = 25)
    axes[1].tick_params(axis='y', labelsize=22)
    axes[1].legend(fontsize=30,loc='upper right')
     

anim = animation.FuncAnimation(fig, animate, frames = 150, interval = 1, blit = False)

if save_mode:
    figures_folder = 'figures'
    if not os.path.isdir(figures_folder):
        os.mkdir(figures_folder)

    anim.save(os.path.join(figures_folder,'fe_minimization.gif'),fps=7.5)
else:
    plt.show()

