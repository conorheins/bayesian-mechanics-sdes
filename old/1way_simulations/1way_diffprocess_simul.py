'''
Test diffusion process
'''
import SubRoutines.Diffprocess_functions as diffusion
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from numpy.linalg import inv, det
from numpy.linalg import eigvals as spec
import matplotlib.pyplot as plt
import seaborn as sns
from SubRoutines import Auxiliary as aux
import matplotlib.cm as cm


def simulation1D(d, x0, dt=0.01, T=100, N=1):  # run diffusion process for multiple trajectories
    w = np.random.normal(0, np.sqrt(dt), (T - 1) * d * N).reshape(
        [d, T - 1, N])  # random fluctuations
    # sqrt dt because standard deviation, but variance = dt
    x = np.empty([d, T, N])  # store values of the process
    if x0.shape == x[:, 0, 0].shape:
        x[:, 0, :] = np.tile(x0, N).reshape([d, N])  # initial condition
    elif x0.shape == x[:, 0, :].shape:
        x[:, 0, :] = x0
    else:
        raise TypeError("Initial condition has wrong dimensions")
    for t in range(1, T):
        x[:, t, :] = x[:, t - 1, :] + dt * drift(x[:, t - 1, :]) \
                     + sigma(x[:, t - 1, :]) * w[:, t - 1, :]
        if np.count_nonzero(np.isnan(x)):
            raise TypeError("nan: process went too far")
    return x


dim = 1
Pi = np.array([1.0])

'''
Setting up the diffusion process
'''


# volatility
def sigma(y):
    return np.log(np.abs(y) + 0.5)


# diffusion
def D(y):
    return sigma(y) * sigma(y) / 2


# divergence of diffusion
grad_sigma = egrad(sigma)  # elementwise gradient of sigma


def divD(y):
    return sigma(y) * grad_sigma(y)


# solenoidal flow
def Q(y):
    return 0


# divergence of solenoidal flow
def divQ(y):
    return 0


# drift
def drift(y):
    return -(D(y) + Q(y)) * Pi * y + divD(y) + divQ(y)


# Setting up the diffusion process
process = diffusion.diffusion_process(dim=dim, drift=drift, volatility=sigma)  # create process

'''
Run simulation
'''

epsilon = 10 ** (-2)  # time-step
T = 5 * 10 ** 2  # number of time-steps
N = 10 ** 5  # number of trajectories

# Setting up the initial condition
x0 = np.random.normal(loc=np.zeros(dim), scale=np.sqrt(Pi ** (-1)), size=[N]).reshape(
    [dim, N])  # stationary initial condition

# 2D histogram of stationary distribution
plt.figure()
plt.hist(x0[0, :], bins=100)
plt.suptitle('Initial (stationary) distribution')
plt.xlabel('x axis')
plt.ylabel('y axis')

# sample paths
# x = process.simulation(x0, epsilon, T, N)  # run simulation
x = process.simulation(x0, epsilon, T, N)

# Recalibrate end time if trajectory didn't end because process went too far
T = x.shape[1]

'''
Show results of simulation
'''

# entropy over time (to see if remains at steady-state)
aux.showentropy(x)

# sample path
plt.figure()
plt.suptitle('Sample path')
plt.plot(range(T), x[0, :, 0], linewidth=0.4)
plt.plot(range(T), x[0, :, 1], linewidth=0.4)
plt.plot(range(T), x[0, :, 2], linewidth=0.4)
plt.xlabel('time $t$')
plt.ylabel('$x_t$')

# 2D histogram of last time-step
t = int(T / 2)  # intermediate time

plt.figure()
plt.hist(x[0, t:, :].reshape(t * N), bins=100)
plt.suptitle('Longtime distribution')
plt.xlabel('x axis')
plt.ylabel('y axis')

# 2D histogram to show not a Gaussian process
plt.figure()
plt.hist2d(x[0, t, :], x[0, -1, :], bins=(50, 50), cmap=cm.jet)
plt.suptitle('Joint probability at two different times')
plt.xlabel('$x_{t-1}$')
plt.ylabel('$x_t$')


# 2D histogram of joint distribution to show x is not a Gaussian process but has Gaussian marginals

# Custom the inside plot: options are: “scatter” | “reg” | “resid” | “kde” | “hex”
sns.set(style="white", color_codes=True)
sns.jointplot(x=x[0, t, :], y=x[0, -1, :], kind='hex', space=0, cmap='Blues', color='skyblue')
plt.xlabel('$x_s$',labelpad=0)
plt.ylabel('$x_t$',labelpad=0)
#plt.text(s='$p(x_{s},x_{t})$', x=1, y=-1, color='black',size='large')
#plt.text(s='$p(x_{s},x_{t})$', x=4, y=-1, color='black',size='large',zorder=10)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.tick_params(axis='both', which='major', pad=-3)
plt.show()
fig = plt.gcf()
#plt.suptitle('$p(x_s)$')
ratio = 1.3
len = 5
fig.set_size_inches(ratio * len, len, forward=True)
plt.savefig("non-Gaussian_diffprocess.png", dpi=100)

