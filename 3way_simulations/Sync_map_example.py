'''
Simulations synchronisation map, 3 way OU process
'''

import numpy as np
import matplotlib.pyplot as plt

#plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{bm}')
#plt.rc('text', usetex=True)
#plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}     # setup matplotlib to use latex for output      # change this if using xetex or lautex
plt.style.use('seaborn-white')
from scipy import stats
'''
Setup
'''

dim = 3  # dimension of state-space
de = slice(0, 1)  # dimensions of external states
db = slice(1, 2)  # dimensions of blanket states (e.g. 2)
di = slice(2, 3)  # dimensions of internal states


#Define precision
Pi = np.random.rand(dim, dim)
# enforce Markov blanket condition
Pi[de, di] = 0
Pi[di, de] = 0
# enforce symmetric
Pi = Pi + Pi.T
#make sure positive definite
if np.any(np.linalg.eigvals(Pi) <= 0):
    Pi = Pi - 1.5 * np.min(np.linalg.eigvals(Pi)) * np.eye(dim)


# We compute the stationary covariance
S= np.linalg.inv(Pi)

# We check that the synchronisation map is well defined
if S[di, db] == 0:
    raise TypeError("Synchronisation map not well defined: bold_mu(b) not invertible!")

# define the linear synchronisation map
mu = S[di, db] * S[db, db] ** (-1)  # expected internal state
eta = S[de, db] * S[db, db] ** (-1)  # expected external state
sync = Pi[de, de] **(-1) * Pi[de,db] * Pi[di,db]**(-1) * Pi[di, di] # sync map
#sync = S[de, db] * S[di, db] ** (-1)  # sync map


'''
Generate samples of Gaussian distribution
'''
N = 10 ** 6   # number of trajectories #usually 3*10**2
x = np.random.multivariate_normal(mean=np.zeros(dim), cov=S, size=N).T

'''
Figure 1
'''
# For each blanket state we plot
# the empirical expected external state
# with the sync(empirical expected internal state)
# to check that the synchronisation map works

delta = 10 ** (-2)  # size of bins
bins = np.arange(np.min(x[db, :]), np.max(x[db, :]), delta)  # partition blanket state-space into several bins
j = np.zeros(bins.shape) #index: scores whether we are using a bin

bold_eta_empirical = np.empty(bins.shape)
sync_bold_mu = np.empty(bins.shape)
bold_mu_empirical = np.empty(bins.shape)

for i in range(len(bins)):
    indices = (x[db, :] >= bins[i]) * (
            x[db, :] <= bins[i] + delta)  # select observations where blanket state is in desired bin
    indices = indices.reshape([N])
    if np.sum(indices) > 1000:  # if there are a sufficient amount of samples #usually 1000
        j[i] = 1  # score that we are using this bin
        eta_samples = x[de, indices]  # select samples of internal states given blanket states
        bold_eta_empirical[i] = np.mean(eta_samples)  # select empirical expected external state given blanket state
        mu_samples = x[di, indices]  # select samples of internal states given blanket states
        bold_mu_empirical[i] = np.mean(mu_samples)  # empirical expected internal state
        sync_bold_mu[i] = sync * bold_mu_empirical[i]  # synchronisation map of empirical expected internal state

plt.figure(1)
plt.clf()
#plt.suptitle('Synchronisation map')
plt.scatter(bins[j == 1], sync_bold_mu[j == 1], s=1, alpha=0.5,
            label='Prediction: $\sigma(\mathbf{\mu}(b))$')  # scatter plot theoretical expected internal state
plt.scatter(bins[j == 1], bold_eta_empirical[j == 1], s=1, alpha=0.5,
            label='External: $\mathbf{\eta}(b)$')  # scatter plot empirical external internal state
plt.xlabel('Blanket state-space $\mathcal{B}$')
plt.ylabel('External state-space $\mathcal{E}$')
plt.legend(loc='upper right')
cor = stats.pearsonr(sync_bold_mu[j == 1], bold_eta_empirical[j == 1])
plt.title(f'Pearson correlation = {np.round(cor[0],6)}...')
#plt.xlim(-4, 4)
#plt.ylim(-4, 4)
plt.savefig("Prediction_sync_map_example.png", dpi=100)