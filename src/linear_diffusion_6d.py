import os
from diffusions import LinearProcess
from utilities import compute_Fboldmu_blanket_landscape, compute_Fboldmu_blanket_over_time, plot_hot_colourline
import jax.numpy as jnp
from jax.numpy.linalg import inv
from jax import random
import numpy as np 
import matplotlib.pyplot as plt

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

