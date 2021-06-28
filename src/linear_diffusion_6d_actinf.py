import os
from diffusions import LinearProcess
from utilities import parse_command_line
from utilities import compute_Fboldmu_blanket_landscape, compute_Fboldmu_blanket_over_time, plot_hot_colourline
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

key, save_mode = parse_command_line(default_key=51) # default key (51) exhibits lots of solenoidal activity, but largely avoids the most-likely sensory/active state line on way to mode of NESS density

## Other keys of interest
# initialization_key = 5    # default configuration in `config_6d.py` file if no key is passed, lots of solenoidal flow / oscillations. Non-monotonic FE descent
# initialization_key = 20   # this one's good too
# initialization_key = 25   # in this one, on the way to steady state the trajectory doesn't quite go along the most likely line, but goes parallel to it
# initialization_key = 26   # lots of solenoidal, kinda avoids the most-likely sensory/active state line
# initialization_key = 50   # lots of solenoidal, kinda avoids the most-likely sensory/active state line

pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}  # setup matplotlib to use latex for output
plt.style.use('seaborn-white')

figures_folder = 'figures'
if not os.path.isdir(figures_folder):
    os.mkdir(figures_folder)

figures_folder = os.path.join(figures_folder, '6d_actinf')
if not os.path.isdir(figures_folder):
    os.mkdir(figures_folder)

seed_folder = os.path.join(figures_folder, f'seed_{key[1]}')
if not os.path.isdir(seed_folder):
    os.mkdir(seed_folder)

flow_parameters, stationary_stats, sync_mappings, dimensions = initialize_6d_OU(rng_key = key)

n_var = 6      # dimensionality

eta_dim = dimensions['eta']
s_dim = dimensions['s']
a_dim = dimensions['a']
b_dim = dimensions['b']
mu_dim = dimensions['mu']
pi_dim = dimensions['pi']

u_dim = dimensions['u'] # dimensions of unperturbed states (i.e. complement of sensory states)

Pi, S = stationary_stats['Pi'], stationary_stats['S']

# Setting up the OU process
process = LinearProcess(dim=n_var, friction=flow_parameters['B'], volatility=flow_parameters['sigma'])  # create process

'''
OU process perturbed blanket states simulation
'''
b_eta = sync_mappings['b_eta']
b_mu  = sync_mappings['b_mu']
sync  = sync_mappings['sync']

dt, T, n_real = 0.01, 500, 1000  # duration of single time-step, total number of integration timesteps, number of parallel paths to simulate

# start many trajectories at a really high free energy (perturbed sensory states)

# Setting up the initial condition
s = 10.0 * jnp.ones( len(s_dim) )  # initialize ensemble of perturbed blanket states

s_init = jnp.tile(s[...,None], (1, n_real) )

u_mean = S[np.ix_(u_dim, s_dim)] @ inv(S[s_dim,s_dim][...,None]) @ s # mean of p(eta, a, mu | s)

x0 = np.empty((n_var, n_real), dtype = np.float32) # initialize as numpy array so we can populate it

x0[s_dim,:] = s_init # fill out with perturbed sensory state

x0[u_dim,:] = np.array(random.multivariate_normal(key, mean=u_mean, cov=inv(Pi[np.ix_(u_dim, u_dim)]), shape = (n_real, )).T)
_, key = random.split(key)

x0 = jnp.array(x0) # convert back to JA device array

# sample paths
x = process.integrate(T, n_real, dt, x0, rng_key = key) # run simulation

'''
Figure 1: Heat map of F(s,a, bold mu) and sample path
'''

S_part_inv = inv(S[np.ix_(pi_dim, pi_dim)])

sensory = jnp.linspace(jnp.min(x[:, s_dim, :]) - 1, jnp.max(x[:, s_dim, :]) + 0.5, 105)  # sensory state-space points
active = jnp.linspace(jnp.min(x[:, a_dim, :]) - 1, jnp.max(x[:, a_dim, :]) + 0.5, 100)  # active state-space points

Z = compute_Fboldmu_blanket_landscape(sensory, active, b_mu, S_part_inv)

real_idx = random.randint(key, shape=(), minval = 0, maxval = n_real)  # which sample path to show (between 0 and n_real)
print(f'Sample path index being shown: {real_idx}\n')

plt.figure(1)
plt.clf()
plt.title('Free energy $F(s,a, \mathbf{\mu})$',fontsize=16)
plt.contourf(sensory, active, Z, levels=100, cmap='turbo')  # plotting the free energy landscape
cb = plt.colorbar(ticks = np.round(np.arange(0,400,step=50)))
cb.ax.tick_params(labelsize=14)
plt.xlabel('sensory state $s$',fontsize=14)
plt.ylabel('active state $a$',fontsize=14)

bold_a_map = S[a_dim, s_dim] / S[s_dim, s_dim] # most likely active state, given sensory states

bold_a = (bold_a_map * sensory).reshape(len(sensory))  # expected active state
plt.plot(sensory, bold_a, c='white')  # plot expected internal state as a function of blanket states

plot_hot_colourline(x[:, s_dim, real_idx].squeeze(), x[:, a_dim, real_idx].squeeze(), lw=0.5)
plt.text(s='$\mathbf{a}(s)$', x=x[:, s_dim, :].min()-0.1, y=bold_a_map * (x[:, s_dim, :].min() - 0.7) + 0.5,
         color='white', fontsize = 16)
plt.text(s='$(s_t,a_t)$', x=x[1, s_dim, real_idx] - 0.5, y=x[1, a_dim, real_idx] + 0.2, color='black', fontsize=16)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
if save_mode:
    figure_name = "Sample_perturbed_AI_6wayOU.png"
    plt.savefig(os.path.join(seed_folder,figure_name),dpi=100)
    plt.close()

'''
Figure 2: average time-varying F(b, bold mu), with average taken across trajectories
'''

# vectorized computation of F(s, a, boldmu) over time
blanket_hist = jnp.transpose(x[:,b_dim,:], (1, 0, 2))
F_trajectories = compute_Fboldmu_blanket_over_time(blanket_hist, b_mu, S_part_inv)
mean_F = F_trajectories.mean(axis=0)


plt.figure(2)
plt.clf()
plt.title('Average free energy over time',fontsize=16)
plot_hot_colourline(np.arange(T), mean_F)
xlabel = int(T * 0.4)  # position of text on x axis
plt.text(s='$F(s_t, a_t, \mathbf{\mu}_t)$', x=xlabel, y=mean_F[xlabel] + 0.05 * (mean_F.max() - mean_F[xlabel]),
         color='black',fontsize=16)
plt.xlabel('Time',fontsize=14)
plt.ylabel('Free energy $F(s_t, a_t, \mathbf{\mu}_t)$',fontsize=14)

if save_mode:
    figure_name = f'FE_vs_time_perturbed_AI_6wayOU.png'
    plt.savefig(os.path.join(seed_folder,figure_name),dpi=100)
    plt.close()
