'''
Bayesian mechanics simulations three way diffusion process
'''

import SubRoutines.Diffprocess_functions as diff
import autograd.numpy as np
from numpy.linalg import inv, det
from numpy.linalg import eigvals as spec
from autograd import elementwise_grad, jacobian
import matplotlib.pyplot as plt
from SubRoutines import Auxiliary as aux
import matplotlib.cm as cm

pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}  # setup matplotlib to use latex for output
plt.style.use('seaborn-white')
import scipy

'''
Setting up the steady-state
'''

dim = 3  # dimension of state-space
de = slice(0, 1)  # dimensions of external states
db = slice(1, 2)  # dimensions of blanket states (e.g. 2)
di = slice(2, 3)  # dimensions of internal states
dp = slice(1, 3)  # dimensions of particular states (internal + blanket)

std = 1  # define standard deviations of distributions we are sampling random numbers

# Define precision
# Pi = np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim]) #random precision matrix
Pi = np.array([3, 1, 0, 1, 3, 0.5, 0, 0.5, 2.5]).reshape([dim, dim])  # selected precision matrix
# enforce Markov blanket condition
Pi[de, di] = 0
Pi[di, de] = 0
# enforce symmetric
Pi = (Pi + Pi.T) / 2
# make sure Pi is positive definite
if np.any(spec(Pi) <= 0):
    Pi = Pi - 2 * np.min(spec(Pi)) * np.eye(dim)

# We compute the stationary covariance
S = np.linalg.inv(Pi)

# check numerical errors are small
if np.sum(np.abs(inv(S) - Pi)) > 10 ** (-6):
    raise TypeError("Precision and inverse covariance are different")

'''
Setting up synchronisation map
'''

# We check that the synchronisation map is well defined
if S[di, db] == 0:
    print("Synchronisation map not well defined: bold_mu(b) not invertible!")
    mu = 0
    sync = 0
else:
    # define the linear synchronisation map
    mu = S[di, db] * S[db, db] ** (-1)  # expected internal state
    sync = Pi[de, de] ** (-1) * Pi[de, db] * Pi[di, db] ** (-1) * Pi[di, di]  # sync map
# sync = S[de, db] * S[di, db] ** (-1)  # sync map
eta = S[de, db] * S[db, db] ** (-1)  # expected external state

'''
Setting up the diffusion process
'''


# volatility
def sigma(y):
    y=y.reshape(y.shape[0])
    return np.diag(y)

# diffusion
def D(y):
    sigma_y = sigma(y)
    return sigma_y @ sigma_y.T / 2

j_D = jacobian(D)

def divD(y):
    return np.trace(j_D(y), axis1=0, axis2=2)

# divergence of diffusion
def divD(y):
    return y

# solenoidal flow
def Q(y):
    dim = y.shape[0]
    temp = np.tile(y, dim).reshape(dim, dim)
    return temp - temp.T

j_Q = jacobian(Q)

# divergence of solenoidal flow
def divQ(y):
    return np.trace(j_Q(y), axis1=0, axis2=2)


# drift
def drift(y):
    return -(D(y)) @ Pi @ y - divD(y)
    #return -(D(y) + Q(y)) @ Pi @ y + divD(y) + divQ(y)


# Setting up the diffusion process
process = diff.diffusion_process(dim=dim, drift=drift, volatility=sigma)  # create process

'''
Run simulation
'''

epsilon = 10 ** (-3)  # time-step
T = 5 * 10 ** 2  # number of time-steps
N = 10 ** 2  # number of trajectories

# Setting up the initial condition
x0 = np.random.multivariate_normal(mean=np.zeros(dim), cov=S, size=[N]).reshape(
    [dim, N])  # stationary initial condition

# 2D histogram of stationary distribution
plt.figure()
plt.hist2d(x0[0, :], x0[1, :], bins=(10, 10), cmap=cm.jet)
plt.suptitle('Initial distribution (2D heatmap)')
plt.xlabel('x axis')
plt.ylabel('y axis')

# sample paths
x = process.simulation(x0, epsilon, T, N)  # run simulation

#Recalibrate time depending on results
T = x.shape[1]
print(f'Time-steps: {T}')

'''
Show results of simulation
'''

# entropy over time (to see if remains at steady-state)
aux.showentropy(x)

# sample path
plt.figure()
for i in range(3):
    aux.plot_hot_colourline(x[0, :, i], x[1, :, i])

# 2D histogram of last time-steps
t = int(T / 2) #intermediate time
plt.figure()
plt.hist2d(x[0, t:, :].reshape(t * N), x[1, t:, :].reshape(t * N), bins=(100, 100), cmap=cm.jet)
plt.suptitle('2D longtime distribution')
plt.xlabel('x axis')
plt.ylabel('y axis')

# 2D histogram to show not a Gaussian process
plt.figure()
plt.hist2d(x[0, t, :], x[1, -1, :], bins=(20, 20), cmap=cm.jet)
plt.suptitle('Non-Gaussian process')
plt.xlabel('$x_{t-1}$')
plt.ylabel('$x_t$')

'''
Figure: sync map diffusion process
'''

# bin blanket state-space
delta = 10 ** (-2)
bins = np.arange(np.min(x[db, :, :]), np.max(x[db, :, :]), delta)

j = np.zeros(bins.shape)  # index: scores whether we are using a bin

bold_eta_empirical = np.empty(bins.shape)
bold_eta_theoretical = np.empty(bins.shape)
sync_bold_mu = np.empty(bins.shape)
bold_mu_empirical = np.empty(bins.shape)

for i in range(len(bins) - 1):
    indices = (x[db, :] >= bins[i]) * (x[db, :] <= bins[i + 1])  # select indices where blanket state is in desired bin
    indices = indices.reshape([T, N])
    if np.sum(indices) > 10:  # if there are a sufficient amount of samples #usually 1000
        j[i] = 1  # score that we are using this bin
        eta_samples = x[de, indices]  # select samples of internal states given blanket states
        bold_eta_empirical[i] = np.mean(eta_samples)  # select empirical expected external state given blanket state
        mu_samples = x[di, indices]  # select samples of internal states given blanket states
        bold_mu_empirical[i] = np.mean(mu_samples)  # empirical expected internal state
        sync_bold_mu[i] = sync * bold_mu_empirical[i]  # synchronisation map of empirical expected internal state
        bold_eta_theoretical[i] = eta * (bins[i] + delta / 2)

plt.figure()
plt.clf()
plt.suptitle('Synchronisation map')
plt.scatter(bins[j == 1], sync_bold_mu[j == 1], s=1, alpha=0.5,
            label='Prediction: $\sigma(\mathbf{\mu}(b_t))$')  # scatter plot theoretical expected internal state
plt.scatter(bins[j == 1], bold_eta_empirical[j == 1], s=1, alpha=0.5,
            label='Actual: $\mathbf{\eta}(b_t)$')  # scatter plot empirical external internal state
# plt.scatter(bins[j == 1], bold_eta_theoretical[j == 1], s=1, alpha=0.5,label='Theo: $\mathbf{\eta}(b)$')
plt.xlabel('Blanket state space $\mathcal{B}$')
plt.ylabel('External state space $\mathcal{E}$')
plt.legend(loc='upper right')
cor = scipy.stats.pearsonr(sync_bold_mu[j == 1], bold_eta_empirical[j == 1])
plt.title(f'Pearson correlation = {np.round(cor[0], 6)}...')
# plt.xlim(-4, 4)
# plt.ylim(-4, 4)
# plt.savefig("sync_map_3waydiffprocess.png")
