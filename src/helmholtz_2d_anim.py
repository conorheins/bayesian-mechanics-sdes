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

initialization_key = 22  # choose a key to fix the random seed for reproducibility

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
