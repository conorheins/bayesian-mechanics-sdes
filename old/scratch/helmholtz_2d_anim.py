import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

import numpy as np
import jax.numpy as jnp
from jax import random
from scipy.stats import multivariate_normal

from diffusions import LinearProcess
from configs.config_2d import initialize_2d_OU

pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}  # setup matplotlib to use latex for output
plt.style.use('seaborn-white')

# key, save_mode = parse_command_line(default_key = 22)

key = random.PRNGKey(22)
save_mode = True

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
x_full = process.integrate(3 * T, n_real, dt, x0, rng_key = key) # run simulation

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

x_cons = conservative_process.integrate(int(2.5*T), n_real, dt, x0, rng_key = key) # run simulation

# traj_idx_cons = random.randint(key, shape=(), minval = 0, maxval = n_real)  # which sample path to show (between 0 and n_real)
# _, key = random.split(key)
traj_idx_cons = 19  # this is my favorite isocontour personally (not too far, not too close)

print(f'Conservative sample path index being shown: {traj_idx_cons}\n')

'''
Dissipative simulation
'''

B_dissipative = flow_parameters['D'] @ stationary_stats['Pi']  # drift matrix

# Setting up the OU process
dissipative_process = LinearProcess(dim=n_var, friction=B_dissipative, volatility=flow_parameters['sigma'])  # create process

x0 = jnp.zeros( (2, n_real) ) + 0.2
_, key = random.split(key)
x_diss = dissipative_process.integrate(int(1.2 * T), n_real, dt, x0, rng_key = key) # run simulation

# traj_idx_diss = random.randint(key, shape=(), minval = 0, maxval = n_real)  # which sample path to show (between 0 and n_real)
# _, key = random.split(key)
traj_idx_diss = 6
print(f'Dissipative sample path index being shown: {traj_idx_diss}\n')

'''
Steady state density (Gaussian)
'''

# %%
lim_x = 2.0
lim_y = 2.0

x_tick = np.linspace(-lim_x, lim_x, 105)  # x axis points
y_tick = np.linspace(-lim_y, lim_y, 100)  # y axis points

X,Y = np.meshgrid(x_tick,y_tick)
pos = np.dstack((X, Y))

rv = multivariate_normal(cov= S) #random normal

past_tstep = 50
lwidths = jnp.linspace(0,20,past_tstep)
colors = cm.Reds(jnp.linspace(0,1,past_tstep))

fig = plt.figure(figsize=(20,6))

con_ax = fig.add_subplot(1,3,1)
full_ax = fig.add_subplot(1,3,2)
diss_ax = fig.add_subplot(1,3,3)

def animate(i):

    con_ax.cla()
    full_ax.cla()
    diss_ax.cla()

    con_ax.contourf(X, Y, rv.pdf(pos), levels=100, cmap='Blues')  # contour plot of the Gaussian pdf for this OU process
    full_ax.contourf(X, Y, rv.pdf(pos), levels=100, cmap='Blues')  # contour plot of the Gaussian pdf for this OU process
    diss_ax.contourf(X, Y, rv.pdf(pos), levels=100, cmap='Blues')  # contour plot of the Gaussian pdf for this OU process

    start_idx = max(0, i - past_tstep)

    # update conservative dynamic axis
    x_0 = x_cons[start_idx:i, 0, traj_idx_cons].squeeze()
    x_1= x_cons[start_idx:i, 1, traj_idx_cons].squeeze()
    
    points = jnp.array([x_0, x_1]).T.reshape(-1,1,2)
    segments = jnp.concatenate([points[:-1], points[1:]], axis=1)
    lc_con = LineCollection(segments, linewidths=lwidths,colors = colors)

    con_ax.add_collection(lc_con)

    # update full dynamic axis
    x_0 = x_full[start_idx:i, 0, traj_idx_full].squeeze()
    x_1 = x_full[start_idx:i, 1, traj_idx_full].squeeze()
    
    points = jnp.array([x_0, x_1]).T.reshape(-1,1,2)
    segments = jnp.concatenate([points[:-1], points[1:]], axis=1)
    lc_full = LineCollection(segments, linewidths=lwidths,colors = colors)

    full_ax.add_collection(lc_full)

    # update dissipative dynamic axis

    x_0 = x_diss[start_idx:i, 0, traj_idx_diss].squeeze()
    x_1= x_diss[start_idx:i, 1, traj_idx_diss].squeeze()
    
    points = jnp.array([x_0, x_1]).T.reshape(-1,1,2)
    segments = jnp.concatenate([points[:-1], points[1:]], axis=1)
    lc_diss = LineCollection(segments, linewidths=lwidths,colors = colors)

    diss_ax.add_collection(lc_diss)

anim = animation.FuncAnimation(fig, animate, frames = 100, interval = 1, blit = False)

if save_mode:
    figures_folder = 'figures'
    if not os.path.isdir(figures_folder):
        os.mkdir(figures_folder)

    anim.save(os.path.join(figures_folder,'hh_decomp.gif'),fps=12.5)
else:
    plt.show()
