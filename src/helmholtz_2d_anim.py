import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import jax.numpy as jnp
from jax import random
from scipy.stats import multivariate_normal

from diffusions import LinearProcess
from configs.config_2d import initialize_2d_OU

pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}  # setup matplotlib to use latex for output
plt.style.use('seaborn-white')

key, save_mode = parse_command_line(default_key = 22)

flow_parameters, stationary_stats = initialize_2d_OU(rng_key = key)

n_var = 2    # dimensionality

Pi, S = stationary_stats['Pi'], stationary_stats['S']

# Setting up the OU process
process = LinearProcess(dim=n_var, friction=flow_parameters['B'], volatility=flow_parameters['sigma'])  # create process

'''
OU process steady-state simulation
'''

dt     = 1e-4     # duration of single integration step , 
T      = int(10e3) # total duration of simulation (in timesteps)
n_real = 5         # number of parallel paths to simulate

key = random.PRNGKey(initialization_key)   


# initialize process starting state
x0 = jnp.transpose(random.multivariate_normal(key, jnp.zeros(n_var), S, shape = (n_real,) ), (1, 0))
_, key = random.split(key)

# sample paths
x = process.integrate(3 * T, n_real, dt, x0, rng_key = key) # run simulation

traj_idx_full = random.randint(key, shape=(), minval = 0, maxval = n_real)  # which sample path to show (between 0 and n_real)
_, key = random.split(key)

print(f'Full dynamics sample path index being shown: {traj_idx_full}\n')


'''
Conservative simulation
'''

B_conservative = flow_parameters['Q'] @ stationary_stats['Pi']  # drift matrix

# Setting up the OU process
conservative_process = LinearProcess(dim=n_var, friction=B_conservative, volatility= jnp.zeros((n_var, n_var)))  # create process

n_real = 50
x0 = jnp.transpose(random.multivariate_normal(key, jnp.zeros(n_var), S, shape = (n_real,) ), (1, 0))
_, key = random.split(key)

x = conservative_process.integrate(int(2.5*T), n_real, dt, x0, rng_key = key) # run simulation

# traj_idx_cons = random.randint(key, shape=(), minval = 0, maxval = n_real)  # which sample path to show (between 0 and n_real)
# _, key = random.split(key)
traj_idx_cons = 19  # this is my favorite isocontour personally (not too far, not too close)

print(f'Conservative sample path index being shown: {real_idx_conservative}\n')


lim_x = 2.0
lim_y = 2.0

x_tick = np.linspace(-lim_x, lim_x, 105)  # x axis points
y_tick = np.linspace(-lim_y, lim_y, 100)  # y axis points

X,Y = np.meshgrid(x_tick,y_tick)
pos = np.dstack((X, Y))

rv = multivariate_normal(cov= S) #random normal


def animate(i):

    axes[0].cla()
    axes[1].cla()
    axes[2].cla()

    axes[0].contourf(X, Y, rv.pdf(pos), levels=100, cmap='Blues')  # contour plot of the Gaussian pdf for this OU process
    axes[1].contourf(X, Y, rv.pdf(pos), levels=100, cmap='Blues')  # contour plot of the Gaussian pdf for this OU process
    axes[2].contourf(X, Y, rv.pdf(pos), levels=100, cmap='Blues')  # contour plot of the Gaussian pdf for this OU process
