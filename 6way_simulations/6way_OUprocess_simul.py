'''
Bayesian mechanics simulations six way OU process
'''

import SubRoutines.OUprocess_functions as OU
import numpy as np
from numpy.linalg import inv, det, pinv
from numpy.linalg import eigvals as spec
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch

pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}  # setup matplotlib to use latex for output
plt.style.use('seaborn-white')
from scipy.linalg import null_space as ker
from scipy.linalg import sqrtm

'''
Functions
'''


def rank(A):  # compute matrix rank
    if A.size == 0:
        return 0
    else:
        return np.linalg.matrix_rank(A)


def num(s):  # counts number of elements
    if isinstance(s, slice):
        if s.step is None:
            return s.stop - s.start
        else:
            return (s.stop - s.start) / s.step
    elif isinstance(s, float):
        return 1
    elif isinstance(s, np.ndarray):
        return int(np.prod(s.shape))
    else:
        print(type(s))
        raise TypeError('Type not supported by num')


'''
Setting up the steady-state
'''

dim = 6  # dimension of state-space
de = slice(0, 2)  # dimensions of external states
ds = slice(2, 3)  # dimensions of sensory states
da = slice(3, 4)  # dimensions of active states
di = slice(4, 6)  # dimensions of internal states
db = slice(2, 4)  # dimensions of blanket states (sensory + active)
dp = slice(2, 6)  # dimensions of particular states (blanket + internal)
du = [0, 1, 3, 4, 5]  # dimensions of unresponsive states (complement of sensory)

std = 1  # standard deviations of gaussian distributions we are sampling random numbers from

# Define precision
Pi = np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim])
# enforce Markov blanket condition
Pi[de, di] = 0
Pi[di, de] = 0
# enforce symmetric
Pi = (Pi + Pi.T) / 2
# make sure positive definite
if np.any(spec(Pi) <= 0):
    Pi = Pi - 2 * np.min(spec(Pi)) * np.eye(dim)

# We compute the stationary covariance
S = np.linalg.inv(Pi)

'''
Setting up the synchronisation map
'''

# We check that the synchronisation map is well defined according to the conditions outlined in the paper
if rank(np.append(ker(Pi[di, db]), ker(Pi[de, db]), axis=1)) > rank(ker(Pi[de, db])):
    raise TypeError("Synchronisation map not well defined")

# define the linear synchronisation map
mu = S[di, db] @ inv(S[db, db])  # expected internal state
eta = S[de, db] @ inv(S[db, db])  # expected external state
sync = inv(Pi[de, de]) @ Pi[de, db] @ pinv(Pi[di, db]) @ Pi[di, di]  # sync map

'''
Setting up the OU process
'''

# volatility and diffusion
sigma = np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim])  # arbitrary volatility matrix
# sigma = np.zeros([dim,dim]) #no noise
# sigma = np.diag(np.random.normal(scale=std, size=dim))

# print whether noise is degenerate or not
print(f'det sigma = {det(sigma)}')

# diffusion tensor
D = (sigma @ sigma.T) / 2

# solenoidal flow
Q = np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim])  # arbitrary solenoidal flow
# Q = np.zeros([dim, dim])  # no solenoidal flow
Q = (Q - Q.T) / 2

# Drift matrix
B = (D + Q) @ Pi  # (negative) drift matrix
if np.any(spec(B) <= -10 ** (-5)):
    print(spec(B))
    raise TypeError("Drift should have non-negative spectrum")

# 1) We check it solves the Sylvester equation: BS + SB.T = 2D
# 2) we check that there are no numerical errors due to ill conditioning
error_sylvester = np.sum(np.abs(B @ S + S @ B.T - 2 * D))
error_inversion = np.sum(np.abs(S @ Pi - np.eye(dim)))
if np.round(error_sylvester, 7) != 0 or np.round(error_inversion, 7) != 0:
    raise TypeError("Sylvester equation not solved")
if np.sum(np.abs(inv(S) - Pi)) > 10 ** (-5):
    raise TypeError("Precision and inverse covariance are different")

# We check that the stationary covariance is indeed positive definite
if np.any(spec(S) <= 0):
    print(spec(S))
    raise TypeError("Stationary covariance not positive definite")

# Setting up the OU process
process = OU.OU_process(dim=dim, friction=B, volatility=sigma)  # create process

'''
OU process perturbed blanket states simulation 
'''
epsilon = 0.01  # time-step
T = 5 * 10 ** 2  # number of time-steps
N = 10 ** 3  # number of trajectories

# start many trajectories at a really high free energy

# Setting up the initial condition
b = np.zeros(num(db)) + 10  # specify blanket state

x0 = np.empty([dim, N])
x0[db, :] = np.tile(b, N).reshape([num(db), N])
# initialising external and internal states at posterior distributions
x0[di, :] = np.random.multivariate_normal(mean=mu @ b, cov=inv(Pi[di, di]), size=N).T
x0[de, :] = np.random.multivariate_normal(mean=eta @ b, cov=inv(Pi[de, de]), size=N).T

# sample paths
x = process.simulation(x0, epsilon, T, N)  # run simulation


'''
Figure 2: Heat map of F(b, bold mu) and sample path
'''

S_part_inv = inv(S[dp, dp])


def F_boldmu_floats(s, a):
    b = np.array([s, a])  # blanket states
    bold_mu = mu @ b  # expected external state
    part_states = np.append(b, bold_mu)  # particular states
    return part_states @ S_part_inv @ part_states / 2  # potential term = surprise of part. states


def F_boldmu(sensory, active):  # computes F(s, a, bold mu) free energy up to an additive constant
    if num(sensory) > 1 and num(active) > 1:
        Z = np.outer(active, sensory)
        for i in range(num(sensory)):
            for j in range(num(active)):
                Z[j, i] = F_boldmu_floats(sensory[i], active[j])
        return Z
    elif num(sensory) == 1 and num(active) == 1:
        return F_boldmu_floats(sensory, active)


sensory = np.linspace(np.min(x[ds, :, :]) - 1, np.max(x[ds, :, :]) + 0.5, 105)  # sensory state-space points
active = np.linspace(np.min(x[da, :, :]) - 1, np.max(x[da, :, :]) + 0.5, 100)  # active state-space points

Z = F_boldmu(sensory, active)  # free energy up to additive constant

n = 0  # which sample path to show (between 0 and N)

plt.figure(2)
plt.clf()
plt.title('Free energy $F(b, \mathbf{\mu})$')
plt.contourf(sensory, active, Z, levels=100, cmap='turbo')  # plotting the free energy landscape
plt.colorbar()
plt.xlabel('blanket state $b_1$')
plt.ylabel('blanket state $b_2$')
# plt.plot(sensory, mu @ blanket, c='white')  # plot expected internal state as a function of blanket states
OU.plot_hot_colourline(x[ds, :, n].reshape(T), x[da, :, n].reshape(T), lw=0.5)
# plt.text(s='$\mathbf{\mu}(b)$', x=x[db, 0, 0] * 0.7, y=mu * x[db, 0, 0] * 0.7 + 0.3, color='white')
plt.text(s='$b_t$', x=x[ds, 1, n] - 1, y=x[da, 1, n], color='black')
plt.savefig("Sample_perturbed_6wayOU.png")

'''
Figure 3: average F(b, bold mu) over trajectories
'''

# compute average free energy over time
F_z = np.empty([T, N])
for t in range(T):
    for n in range(N):
        F_z[t, n] = F_boldmu(x[ds, t, n], x[da, t, n])  # compute free energy
mean_F = np.mean(F_z, axis=1)  # take mean over trajectories

plt.figure(3)
plt.clf()
plt.title('Average free energy of perturbed system')
OU.plot_hot_colourline(np.arange(T), mean_F)
xlabel = int(T * 0.4)  # position of text on x axis
plt.text(s='$F(b_t, \mathbf{\mu}_t)$', x=xlabel, y=mean_F[xlabel] + 0.05 * (np.max(mean_F) - mean_F[xlabel]),
         color='black')
plt.xlabel('Time')
plt.ylabel('Free energy $F(b, \mathbf{\mu})$')
plt.savefig("FE_vs_time_perturbed_6wayOU.png")


'''
OU process perturbed sensory states simulation 
'''
# start many trajectories at a really high free energy

# Setting up the initial condition
s = 10  # specify sensory state


def insensitive(A):  # from matrix (Pi or its inverse), removes all columns and rows that pertain to sensory states
    B = np.zeros(np.array(A.shape) - 1)
    for i in range(A.shape[0] - 1):
        for j in range(A.shape[1] - 1):
            k = i
            l = j
            if k >= 2:
                k = k + 1
            if l >= 2:
                l = l + 1
            B[i, j] = A[k, l]
    return B


x0 = np.empty([dim, N])
x0[ds, :] = s
# initialising external and internal states at posterior distributions given sensory states
mean_posterior = (S[du, ds] * S[ds, ds] ** (-1) * s).reshape(len(du))  # mean of p(eta, a, mu | s)
x0[du, :] = np.random.multivariate_normal(mean=mean_posterior, cov=inv(insensitive(Pi)), size=N).T

# sample paths
x = process.simulation(x0, epsilon, T, N)  # run simulation

'''
Figure 4: Heat map of F(s,a, bold mu) and sample path
'''

sensory = np.linspace(np.min(x[ds, :, :]) - 1, np.max(x[ds, :, :]) + 0.5, 105)  # sensory state-space points
active = np.linspace(np.min(x[da, :, :]) - 1, np.max(x[da, :, :]) + 0.5, 100)  # active state-space points

Z = F_boldmu(sensory, active)  # free energy up to additive constant

n = 0  # which sample path to show (between 0 and N)

plt.figure(4)
plt.clf()
plt.title('Free energy $F(s,a, \mathbf{\mu})$')
plt.contourf(sensory, active, Z, levels=100, cmap='turbo')  # plotting the free energy landscape
plt.colorbar()
plt.xlabel('sensory state $s$')
plt.ylabel('active state $a$')
bold_a = (S[da, ds] / S[ds, ds] * sensory).reshape(len(sensory))  # expected active state
plt.plot(sensory, bold_a, c='white')  # plot expected internal state as a function of blanket states
OU.plot_hot_colourline(x[ds, :, n].reshape(T), x[da, :, n].reshape(T), lw=0.5)
plt.text(s='$\mathbf{a}(s)$', x=np.min(x[ds, :, :]) - 0.7, y=S[da, ds] / S[ds, ds] * (np.min(x[ds, :, :]) - 0.7) + 0.4,
         color='white')
plt.text(s='$(s_t,a_t)$', x=x[ds, 1, n] - 1, y=x[da, 1, n] + 0.2, color='black')
plt.savefig("Sample_perturbed_AI_6wayOU.png")

'''
Figure 5: average F(b, bold mu) over trajectories
'''

# compute average free energy over time
F_z = np.empty([T, N])
for t in range(T):
    for n in range(N):
        F_z[t, n] = F_boldmu(x[ds, t, n], x[da, t, n])  # compute free energy
mean_F = np.mean(F_z, axis=1)  # take mean over trajectories

plt.figure(5)
plt.clf()
plt.title('Average free energy of perturbed system')
OU.plot_hot_colourline(np.arange(T), mean_F)
xlabel = int(T * 0.4)  # position of text on x axis
plt.text(s='$F(s_t, a_t, \mathbf{\mu}_t)$', x=xlabel, y=mean_F[xlabel] + 0.05 * (np.max(mean_F) - mean_F[xlabel]),
         color='black')
plt.xlabel('Time')
plt.ylabel('Free energy $F(s,a, \mathbf{\mu})$')
plt.savefig("FE_vs_time_perturbed_AI_6wayOU.png")

'''
Figure 6: Evoked response plots (showing how conditional distribution over external states - parameterised with variational mean and variance) changes
over time after the perturbation.
'''

##
## Figure 6a - plot the marginal predictions for each external state, overlaid with true external state and marginal variances
##

mean_trajectory = x.mean(axis=2)
eta_samples = x[de,:,:]

posterior_means = sync.dot(mean_trajectory[di,:])
posterior_stds = sqrtm(inv(Pi[de,de]))

pred_upper_CI_mu0 = posterior_means[0]+ 1.96 * posterior_stds[0,0]
pred_lower_CI_mu0 = posterior_means[0]- 1.96 * posterior_stds[0,0]

pred_upper_CI_mu1 = posterior_means[1] + 1.96 * posterior_stds[1,1]
pred_lower_CI_mu1 = posterior_means[1] - 1.96 * posterior_stds[1,1]

t_axis = np.arange(T)
plt.figure(figsize=(19,12))
plt.clf()
plt.title('Average predictions $\sigma(\mu_t)$ and realized trajectories $\eta_t$',fontsize=30, pad = 10)

plt.fill_between(t_axis,pred_upper_CI_mu0, pred_lower_CI_mu0, color='b', alpha=0.15)
eta1_real_line = plt.plot(t_axis, eta_samples[0,:,::50], lw = 0.5, color = 'r', label='Sample paths: $\eta_1(t)$')
mu1_mean_line = plt.plot(t_axis,posterior_means[0], color='b',label='Average prediction: $\mathbf{\eta}_1(\mu_t)$',lw=2.0)

plot_offset = 3.0
plt.fill_between(t_axis,pred_upper_CI_mu1 + plot_offset, pred_lower_CI_mu1 + plot_offset, color='#3de3ac', alpha=0.25)
eta2_real_line = plt.plot(t_axis, eta_samples[1,:,::50] + plot_offset, lw = 0.5, color = '#933aab', label='Sample paths: $\eta_2(t)$')
mu2_mean_line = plt.plot(t_axis,posterior_means[1] + plot_offset, color='#3aab89',label='Average prediction: $\mathbf{\eta}_2(\mu_t)$',lw=2.0)

ci_patch_1 = Patch(color='blue',alpha=0.1, label='Confidence intervals $\mathbf{\eta}_1(\mu_t)$')
ci_patch_2 = Patch(color='#3de3ac',alpha=0.25, label='Confidence intervals $\mathbf{\eta}_2(\mu_t)$')

plt.legend(handles=[mu1_mean_line[0], eta1_real_line[0], mu2_mean_line[0], eta2_real_line[0], ci_patch_1, ci_patch_2], loc='lower left',fontsize=20, ncol = 3)

min_value = min( ( pred_lower_CI_mu0.min(), eta_samples[0].min() ) )
max_value = max( ( (pred_upper_CI_mu1 + plot_offset).max(), (eta_samples[1]+plot_offset).max() ) )

plt.xlim(t_axis[0], t_axis[-1])
plt.ylim(1.25 * min_value, 1.1 * max_value )

plt.gca().tick_params(axis='both', which='both', labelsize=25)
plt.gca().set_xlabel('Time',fontsize=30)
plt.gca().set_ylabel('External state space $\mathcal{E}$',fontsize=30)
plt.savefig("average_prediction_sensory_perturbation.png", dpi=100)

##
## Figure 6b - plot the centered bivariate predictions for external states, overlaid with the analytic covariance / confidence intervals
##

steady_state_taxis = np.arange(100,T) # time period where steady-state (post-perturbation) is assumed
x_ss = x[:,steady_state_taxis,:].reshape(x.shape[0], len(steady_state_taxis) * x.shape[2]) # unwrap realizations (third dimension) to make one long matrix

mu_ss = x_ss[di,:] # all realizations, for all timesteps, of internal states
eta_ss = x_ss[de,:] # all realizations, for all timesteps, of external states

q_mu = sync.dot(mu_ss) # predicted external states, parameterised by instantaneous internal states

fig, ax = plt.subplots(figsize=(14,10))
plt.title('Prediction errors and covariance $\mathbf{\Pi}_{\eta}^{-1}$',fontsize=26, pad = 10)

prediction_errors = eta_ss - q_mu # prediction errors - difference between realization of external state and 'predicted' external state
prediction_centroid = np.mean(prediction_errors,axis=1) # mean over time should be roughly [0.0, 0.0]

dots = ax.plot(prediction_errors[0,0::100],prediction_errors[1,0::100],'ko',markersize=3.5,label='Centered predictions')
posterior_cov = inv(Pi[de,de]) # analytic covariance of posterior predictions (simply the marginal covariance of the external states)

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

vals, vecs = eigsorted(posterior_cov)
theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

nstd = 3.0

height, width = 2 * nstd * np.sqrt(vals)
ellip = Ellipse(xy=prediction_centroid, width=width, height=height, angle=theta, alpha=0.2, color='blue')
ax.add_artist(ellip)
ci_patch = Patch(color='blue',alpha=0.2, label='Confidence ellipse')

plt.legend(handles=[dots[0], ci_patch], loc='lower right',fontsize=20)

min_value = prediction_errors[1].min()
max_value = prediction_errors[1].max()
plt.ylim(1.25 * min_value, 1.1 * max_value )

plt.gca().tick_params(axis='both', which='both', labelsize=25)
plt.gca().set_xlabel('$\eta_1 - q_\mathbf{\mu}(\eta_1)$',fontsize=30)
plt.gca().set_ylabel('$\eta_2 - q_\mathbf{\mu}(\eta_2)$',fontsize=30)
plt.savefig("prediction_error_cov_plot.png",dpi=100)


