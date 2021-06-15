'''
Illustration Helmholtz decomposition of a 2-D OU process
'''

import os
from diffusions import LinearProcess
from utilities import plot_hot_colourline

import jax.numpy as jnp
from jax import random
from scipy.stats import multivariate_normal

import numpy as np 
import matplotlib.pyplot as plt

from configs.config_2d import initialize_2d_OU

# initialization_key = 50    # default configuration in `config_2d.py` file if no key is passed
initialization_key = 45    # default configuration in `config_2d.py` file if no key is passed

key = random.PRNGKey(initialization_key)   

save_mode = True

pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}  # setup matplotlib to use latex for output
plt.style.use('seaborn-white')

figures_folder = 'figures'
if not os.path.isdir(figures_folder):
    os.mkdir(figures_folder)

figures_folder = os.path.join(figures_folder, '2d_HH_demo')
if not os.path.isdir(figures_folder):
    os.mkdir(figures_folder)

seed_folder = os.path.join(figures_folder, f'seed_{initialization_key}')
if not os.path.isdir(seed_folder):
    os.mkdir(seed_folder)

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


# initialize process starting state
x0 = jnp.transpose(random.multivariate_normal(key, jnp.zeros(n_var), S, shape = (n_real,) ), (1, 0))
_, key = random.split(key)

# sample paths
x = process.integrate(T, n_real, dt, x0, rng_key = key) # run simulation


'''
Simulation trajectory on heat map of steady-state
'''

real_idx = random.randint(key, shape=(), minval = 0, maxval = n_real)  # which sample path to show (between 0 and n_real)
print(f'Sample path index being shown: {real_idx}\n')

lim_x = 2
lim_y = 1.5

x_tick = np.linspace(-lim_x, lim_x, 105)  # x axis points
y_tick = np.linspace(-lim_y, lim_y, 100)  # y axis points
#x_tick = np.linspace(np.min(x[:, 0, real_idx]) - 0.5, np.max(x[:, 0, real_idx]) + 0.5, 105)  # x axis points
#y_tick = np.linspace(np.min(x[:, 1, real_idx]) - 0.5, np.max(x[:, 1, real_idx]) + 0.5, 100)  # y axis points

X,Y = np.meshgrid(x_tick,y_tick)
pos = np.dstack((X, Y))

rv = multivariate_normal(cov= S) #random normal

plt.figure(2)
plt.clf()
plt.title('')
plt.contourf(X, Y, rv.pdf(pos), levels=100, cmap='Blues')  # plotting the free energy landscape

plt.suptitle('Full dynamic')
plt.title(r'$dx_t = b_{rev}(x_t)dt+b_{irrev}(x_t)dt+ \varsigma(x_t)dW_t$')

plot_hot_colourline(x[:, 0, real_idx].squeeze(), x[:, 1, real_idx].squeeze(), lw=0.5) # warning: this runs slow for long trajectories
# plt.plot(x[:, 0, real_idx].squeeze(), x[:, 1, real_idx].squeeze(), lw=0.5) # this runs faster due to same color for every point - use this for debugging purposes

if save_mode:
    figure_name = "Helmholtz_complete.png"
    plt.savefig(os.path.join(seed_folder, figure_name), dpi=100)
    plt.close()

# plt.figure(3)
# plt.clf()
# plt.plot(range(T), x[:, 0, real_idx].squeeze())
# plt.plot(range(T), x[:, 1, real_idx].squeeze())
# plt.plot(x[:, 0, real_idx].squeeze(), x[:, 1, real_idx].squeeze())
# plt.close()

'''
Conservative simulation
'''

B_conservative = flow_parameters['Q'] @ stationary_stats['Pi']  # drift matrix

# Setting up the OU process
conservative_process = LinearProcess(dim=n_var, friction=B_conservative, volatility= jnp.zeros((n_var, n_var)))  # create process

_, key = random.split(key)
x = conservative_process.integrate(T, n_real, dt, x0, rng_key = key) # run simulation

'''
Plot trajectory conservative simulation
'''

plt.figure(4)
plt.clf()
plt.title('')
plt.contourf(X, Y, rv.pdf(pos), levels=100, cmap='Blues')
plt.suptitle('Time-irreversible')
plt.title(r'$dx_t = b_{irrev}(x_t)dt$')
plot_hot_colourline(x[:, 0, real_idx].squeeze(), x[:, 1, real_idx].squeeze(), lw=0.5)  # warning: this runs slow for long trajectories
# plt.plot(x[:, 0, real_idx].squeeze(), x[:, 1, real_idx].squeeze(), lw=0.5) # this runs faster due to same color for every point - use this for debugging purposes

if save_mode:
    figure_name = "Helmholtz_conservative.png"
    plt.savefig(os.path.join(seed_folder, figure_name), dpi=100)
    plt.close()

# plt.figure(5)
# plt.clf()
# plt.plot(range(T), x[:, 0, real_idx].squeeze())
# plt.plot(range(T), x[:, 1, real_idx].squeeze())
# plt.plot(x[:, 0, real_idx].squeeze(), x[:, 1, real_idx].squeeze())
# plt.close()


'''
Dissipative simulation
'''

B_dissipative = flow_parameters['D'] @ stationary_stats['S']  # drift matrix

# Setting up the OU process
dissipative_process = LinearProcess(dim=n_var, friction=B_dissipative, volatility=flow_parameters['sigma'])  # create process

_, key = random.split(key)
x = dissipative_process.integrate(T, n_real, dt, x0, rng_key = key) # run simulation

'''
Plot trajectory dissipative simulation
'''

plt.figure(6)
plt.clf()
plt.title('')
plt.contourf(X, Y, rv.pdf(pos), levels=100, cmap='Blues')
plt.suptitle('Time-reversible')
plt.title(r'$dx_t = b_{rev}(x_t)dt+ \varsigma(x_t)dW_t$')
plot_hot_colourline(x[:, 0, real_idx].reshape(T), x[:, 1, real_idx].reshape(T), lw=0.5) # warning: this runs slow for long trajectories
# plt.plot(x[:, 0, real_idx].squeeze(), x[:, 1, real_idx].squeeze(), lw=0.5) # this runs faster due to same color for every point

if save_mode:
    figure_name = "Helmholtz_dissipative.png"
    plt.savefig(os.path.join(seed_folder, figure_name), dpi=100)
    plt.close()


# plt.figure(7)
# plt.clf()
# plt.plot(range(T), x[0, :, n].reshape(T))
# plt.plot(range(T), x[1, :, n].reshape(T))
# plt.plot(x[:, 0, real_idx].squeeze(), x[:, 1, real_idx].squeeze())
# plt.close()


'''
Plot steady-state density
'''

rv_eye = multivariate_normal(cov= np.eye(n_var)) #random normal

lim_x = 2.5
lim_y = 2.5

x_tick = np.linspace(-lim_x, lim_x, 105)  # x axis points
y_tick = np.linspace(-lim_y, lim_y, 100)  # y axis points

X,Y = np.meshgrid(x_tick,y_tick)
pos = np.dstack((X, Y))

plt.figure(8)
plt.clf()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, rv_eye.pdf(pos), rstride=1, cstride=1,cmap='Blues', edgecolor='none')
ax.grid(False)
ax.set_zlim(0,1.5* rv_eye.pdf(pos).max() )
ax.elev +=-15
ax.axis('off')
ratio = 1.3
len = 8
ax.figure.set_size_inches(ratio * len, len, forward=True)

if save_mode:
    figure_name = "3Dplot_Gaussian_density.png"
    plt.savefig(os.path.join(seed_folder, figure_name), dpi=100)
    plt.close()



