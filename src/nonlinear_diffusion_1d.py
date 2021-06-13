import os
from diffusions import NonlinearProcess
import jax.numpy as jnp
from jax import random, jacfwd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

initialization_key = 0   # default configuration 

key = random.PRNGKey(initialization_key) # fix random seed for reproducibility

save_mode = True

figures_folder = 'figures'
if not os.path.isdir(figures_folder):
    os.mkdir(figures_folder)

figures_folder = os.path.join(figures_folder, '1d_nonlinear_diffusion')
if not os.path.isdir(figures_folder):
    os.mkdir(figures_folder)

seed_folder = os.path.join(figures_folder, f'seed_{initialization_key}')
if not os.path.isdir(seed_folder):
    os.mkdir(seed_folder)


n_var, dt, T, n_real = 1, 10 ** (-2), 5 * 10 ** 2, 10 ** 5 # duration of single time-step, total number of integration timesteps, number of parallel paths to simulate
Pi = jnp.array([[1.0]])

# state dependent volatility
def sigma(y):
    return  jnp.log(jnp.abs(y) + 0.5)[None,...] # have to expand dims to make it technically a 1 x 1 volatility matrix

# state-dependent diffusion tensor
def D(y):
    return (sigma(y) @ sigma(y).T) / 2.0

jac_sigma = jacfwd(sigma) # elementwise gradient of sigma
divD = lambda y: sigma(y) @ jnp.trace(jac_sigma(y))

# solenoidal flow
Q = lambda y: jnp.array([0.0])

# divergence of solenoidal flow
divQ = lambda y: jnp.array([0.0])

# drift expressed in terms of Helmholtz decomposition
drift = lambda y: -(D(y) + Q(y)) @ Pi @ y + divD(y) + divQ(y)

# instantiate the diffusion process
diff_process = NonlinearProcess(n_var, drift, sigma)

# Setting up the initial condition
x0 = jnp.transpose(random.multivariate_normal(key, jnp.zeros(n_var), jnp.linalg.inv(Pi), shape = (n_real,) ), (1, 0))
_, key = random.split(key) # split the random key

x_t = diff_process.integrate(T, n_real, dt, x0, rng_key = key) # integrate the process

# 2D histogram of stationary distribution
plt.figure()
plt.hist(x0[0, :], bins=100)
plt.suptitle('Initial (stationary) distribution')
plt.xlabel('x axis')
plt.ylabel('y axis')
if save_mode:
    figure_name = "initial_distribution.png"
    plt.savefig(os.path.join(seed_folder,figure_name), dpi=100)


'''
Show results of simulation
'''

# sample path
plt.figure()
plt.suptitle('Sample path')
plt.plot(range(T), x_t[:, 0, 0], linewidth=0.4)
plt.plot(range(T), x_t[:, 0, 1], linewidth=0.4)
plt.plot(range(T), x_t[:, 0, 2], linewidth=0.4)
plt.xlabel('time $t$')
plt.ylabel('$x_t$')
if save_mode:
    figure_name = "sample_path_1d_diffusion.png"
    plt.savefig(os.path.join(seed_folder,figure_name), dpi=100)
    plt.close()


# 2D histogram of last time-step
t = int(T / 2)  # intermediate time

plt.figure()
plt.hist(x_t[t:, 0, :].reshape(t * n_real), bins=100)
plt.suptitle('Longtime distribution')
plt.xlabel('x axis')
plt.ylabel('y axis')
if save_mode:
    figure_name = "longtime_distribution_1d_diffusion.png"
    plt.savefig(os.path.join(seed_folder,figure_name), dpi=100)
    plt.close()


# 2D histogram to show not a Gaussian process
plt.figure()
plt.hist2d(x_t[t, 0, :], x_t[-1, 0, :], bins=(50, 50), cmap=cm.jet)
plt.suptitle('Joint probability at two different times')
plt.xlabel('$x_s$')
plt.ylabel('$x_t$')
if save_mode:
    figure_name = "joint_probability_twotimes_1d_diffusion.png"
    plt.savefig(os.path.join(seed_folder,figure_name), dpi=100)
    plt.close()

# 2D histogram of joint distribution to show x is not a Gaussian process but has Gaussian marginals

# Custom the inside plot: options are: “scatter” | “reg” | “resid” | “kde” | “hex”
sns.set(style="white", color_codes=True)
sns.jointplot(x=x_t[t, 0, :], y=x_t[-1, 0, :], kind='hex', space=0, cmap='Blues', color='skyblue')
plt.xlabel('$x_s$',labelpad=0)
plt.ylabel('$x_t$',labelpad=0)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.tick_params(axis='both', which='major', pad=-3)
fig = plt.gcf()
ratio = 1.3
len = 5
fig.set_size_inches(ratio * len, len, forward=True)
if save_mode:
    figure_name = "non-Gaussian_diffprocess_1d_diffusion.png"
    plt.savefig(os.path.join(seed_folder,figure_name), dpi=100)
    plt.close()
