'''
Bayesian mechanics simulations three way OU process
'''

import SubRoutines.OUprocess_functions as OU
import numpy as np
from numpy.linalg import inv, det
from numpy.linalg import eigvals as spec
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import multivariate_normal
import matplotlib.cm as cm
import seaborn as sns

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
Pi = np.array([3, 1, 0, 1, 3, 0.5, 0, 0.5, 2.5]).reshape([dim, dim]) #selected precision matrix
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

'''
Setting up synchronisation map
'''

# We check that the synchronisation map is well defined
if S[di, db] == 0:
    raise TypeError("Synchronisation map not well defined: bold_mu(b) not invertible!")

# define the linear synchronisation map
mu = S[di, db] * S[db, db] ** (-1)  # expected internal state
eta = S[de, db] * S[db, db] ** (-1)  # expected external state
sync = Pi[de, de] ** (-1) * Pi[de, db] * Pi[di, db] ** (-1) * Pi[di, di]  # sync map
# sync = S[de, db] * S[di, db] ** (-1)  # sync map


'''
Setting up the OU process
'''

# volatility
# sigma = np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim])  # arbitrary volatility matrix
# sigma = np.zeros([dim,dim]) #no noise
# sigma = np.diag(np.random.normal(scale=std, size=dim)) # arbitrary diagonal volatility matrix
sigma = np.array([2, 1.5, 0.5, 0, 3, 2, 0, 0, 2]).reshape([dim, dim]) #selected non-degenerate noise
#sigma = np.array([2, 1.5, 0.5, 0, 0, 2, 0, 0, 2]).reshape([dim, dim]) #selected degenerate noise

# see whether noise is degenerate or not
print(f'det sigma = {det(sigma)}')

#diffusion tensor
D = sigma @ sigma.T / 2  # diffusion tensor

# solenoidal flow
#Q = np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim])  # arbitrary solenoidal flow
# Q = np.zeros([dim, dim])  # no solenoidal flow
Q = np.array([0, 1.5, 0.5, -1.5, 0, -1, -0.5, 1, 0]).reshape([dim, dim]) #selected solenoidal flow
Q = (Q - Q.T)/2

# Drift matrix
B = (D + Q) @ Pi  # drift matrix
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
OU process steady-state simulation
'''
#this stationary simulation is super important to see how well the synchronisation map works for the stationary process
#despite errors in numerical discretisation and matrix ill-conditioning
#all subsequent simulations can only be trusted to the extent that the synchronisation map
# works in the steady-state simulation

epsilon = 0.01  # time-step
T = 5 * 10 ** 2  # number of time-steps
N = 10 ** 3  # number of trajectories

# start many trajectories at a really high free energy

# Setting up the initial condition
x0 = np.random.multivariate_normal(mean=np.zeros(dim), cov=S, size=[N]).reshape(
    [dim, N])  # stationary initial condition

# sample paths
x = process.simulation(x0, epsilon, T, N)  # run simulation

'''
Figure 1: sync map OU process
'''

#bin blanket state-space
delta = 10**(-2)
bins = np.arange(np.min(x[db, :, :]), np.max(x[db, :, :]), delta/np.abs(float(eta)) )

j = np.zeros(bins.shape)  # index: scores whether we are using a bin

bold_eta_empirical = np.empty(bins.shape)
bold_eta_theoretical = np.empty(bins.shape)
sync_bold_mu = np.empty(bins.shape)
bold_mu_empirical = np.empty(bins.shape)

for i in range(len(bins)-1):
    indices = (x[db, :] >= bins[i]) * (x[db, :] <= bins[i+1]) #select indices where blanket state is in desired bin
    indices = indices.reshape([T, N])
    if np.sum(indices) > 1000:  # if there are a sufficient amount of samples #usually 1000
        j[i] = 1  # score that we are using this bin
        eta_samples = x[de, indices]  # select samples of internal states given blanket states
        bold_eta_empirical[i] = np.mean(eta_samples)  # select empirical expected external state given blanket state
        mu_samples = x[di, indices]  # select samples of internal states given blanket states
        bold_mu_empirical[i] = np.mean(mu_samples)  # empirical expected internal state
        sync_bold_mu[i] = sync * bold_mu_empirical[i]  # synchronisation map of empirical expected internal state
        bold_eta_theoretical[i] = eta * (bins[i] + delta / 2)

plt.figure(1)
plt.clf()
plt.suptitle('Synchronisation map')
plt.scatter(bins[j == 1], sync_bold_mu[j == 1], s=1, alpha=0.5,
            label='Prediction: $\sigma(\mathbf{\mu}(b_t))$')  # scatter plot theoretical expected internal state
plt.scatter(bins[j == 1], bold_eta_empirical[j == 1], s=1, alpha=0.5,
            label='External: $\mathbf{\eta}(b_t)$')  # scatter plot empirical external internal state
# plt.scatter(bins[j == 1], bold_eta_theoretical[j == 1], s=1, alpha=0.5,label='Theo: $\mathbf{\eta}(b)$')
plt.xlabel('Blanket state-space $\mathcal{B}$')
plt.ylabel('External state-space $\mathcal{E}$')
plt.legend(loc='upper right')
cor = scipy.stats.pearsonr(sync_bold_mu[j == 1], bold_eta_empirical[j == 1])
plt.title(f'Pearson correlation = {np.round(cor[0], 6)}...')
# plt.xlim(-4, 4)
# plt.ylim(-4, 4)
plt.savefig("sync_map_3wayOUprocess.png", dpi=100)



'''
OU process perturbed simulation
'''

b= 10 #specify blanket state

x0[db,:] = b
#initialising external and internal states at posterior distributions
x0[di,:] = np.random.normal(loc=float(mu)*b, scale= np.sqrt(float(Pi[di,di])**(-1)), size=[N])
x0[de,:] = np.random.normal(loc=float(eta)*b, scale= np.sqrt(float(Pi[de,de])**(-1)), size=[N])

# sample paths
x = process.simulation(x0, epsilon, T, N)  # run simulation


'''
Figure 2: Heat map of F(mu,b) and sample path
'''

S_part_inv = inv(S[dp, dp])


def F(internal, blanket):  # computes free energy up to an additive constant
    Z = np.outer(internal, blanket)
    for j in range(len(blanket)):
        b = blanket[j]
        bold_eta = eta * b  # expected external state
        for k in range(len(internal)):
            i = internal[k]
            pred_eta = sync * i  # predicted external state
            part_states = np.array([b, i])  # particular states
            Z[k, j] = part_states @ S_part_inv @ part_states / 2  # potential term = surprise of part. states
            Z[k, j] = Z[k, j] + Pi[di, di] * (pred_eta - bold_eta) ** 2 / 2  # KL term
    return Z


internal = np.linspace(np.min(x[di, :, :]) - 1, np.max(x[di, :, :]) + 0.5, 105)  # internal state-space points
blanket = np.linspace(np.min(x[db, :, :]) - 1, np.max(x[db, :, :]) + 0.5, 100)  # blanket state-space points

Z = F(internal, blanket)  # free energy up to additive constant

n = 3  # which sample path to show (between 0 and N)

plt.figure(2)
plt.clf()
plt.title('Free energy $F(b, \mu)$')
plt.contourf(blanket, internal, Z, levels=100, cmap='turbo')  # plotting the free energy landscape
plt.colorbar()
plt.ylabel('internal state $ \mu$')
plt.xlabel('blanket state $b$')
plt.plot(blanket, float(mu) * blanket, c='white')  # plot expected internal state as a function of blanket states
OU.plot_hot_colourline(x[db, :, n].reshape([T]), x[di, :, n].reshape([T]), lw=0.5)
plt.text(s='$\mathbf{\mu}(b)$', x=np.min(x[db, :, :]) - 0.7, y=mu * (np.min(x[db, :, :]) - 0.7) + 0.2, color='white')
plt.text(s='$(b_t, \mu_t)$', x=x[db, 0, n] - 2, y=x[di, 0, n], color='black')
plt.savefig("Sample_perturbed_3wayOU.png", dpi=100)

'''
Figure 3: average free energy over trajectories
'''

# compute average free energy over time
F_z = np.empty([T, N])
for t in range(T):
    for m in range(N):
        F_z[t, m] = F(x[di, t, m], x[db, t, m])  # compute free energy
mean_F = np.mean(F_z, axis=1)  # take mean over trajectories

plt.figure(3)
plt.clf()
plt.title('Average free energy over time')
OU.plot_hot_colourline(np.arange(T), mean_F)
xlabel = int(T * 0.4)  # position of text on x axis
plt.text(s='$F(b_t, \mu_t)$', x=xlabel, y=mean_F[xlabel] + 0.05 * (np.max(mean_F) - mean_F[xlabel]), color='black')
plt.xlabel('Time')
plt.ylabel('Free energy $F(b_t, \mu_t)$')
plt.savefig("FE_vs_time_perturbed_3wayOU.png", dpi=100)


'''
Figure 4: predictive processing
'''

T_fig= 300

sample_trajectory = x[:,:T_fig,n] # choose sample trajectory
eta_samples = sample_trajectory[de,:]
mu_samples = sample_trajectory[di,:]

posterior_means = sync.dot(mu_samples) # predicted external states = sigma(mu)
posterior_cov = inv(Pi[de,de])

conf_interval_param =1.96
pred_upper_CI_mu0 = posterior_means+ conf_interval_param * posterior_cov
pred_lower_CI_mu0 = posterior_means- conf_interval_param * posterior_cov

t_axis = np.arange(T_fig)
plt.figure(figsize=(8,6))
plt.clf()
plt.title('Predictive processing $q_{\mu_t}(\eta)$ vs $\eta_t$')

plt.fill_between(t_axis,pred_upper_CI_mu0[0,:], pred_lower_CI_mu0[0,:], color='b', alpha=0.15)
eta1_real_line = plt.plot(t_axis, eta_samples[0,:], lw = 0.5, color = 'r', alpha=0.6, label='External: $\eta_{t}$')
mu1_mean_line = plt.plot(t_axis,posterior_means[0,:], color='b',label='Prediction: $q_{\mu_t}(\eta)$',lw=2.0)

ci_patch_1 = Patch(color='blue',alpha=0.1, label=' ')

first_legend = plt.legend(handles=[ci_patch_1], fontsize=12.9, loc=(0.001,0.0135), ncol = 1)
# Add the legend manually to the current Axes.
plt.gca().add_artist(first_legend)
plt.legend(handles=[mu1_mean_line[0], eta1_real_line[0]], loc='lower left', ncol = 2)

min_value = min( pred_lower_CI_mu0.min(), eta_samples[0,:].min() )
max_value = max( pred_upper_CI_mu0.max(), eta_samples[0,:].max() )

plt.xlim(t_axis[0], t_axis[-1])
plt.ylim(1.25 * min_value, 1.25 * max_value)

plt.gca().tick_params(axis='both', which='both')
plt.gca().set_xlabel('Time')
plt.gca().set_ylabel('External state-space $\mathcal{E}$')
plt.savefig("average_prediction_3way_OUprocess.png", dpi=100)



'''
Figure 5 - plot the prediction errors evolving over time
'''

x_flat = x.reshape(dim, T *N) # unwrap realizations (third dimension) to make one long matrix

mu_flat = x_flat[di,:] # all realizations, for all timesteps, of blanket states
eta_flat = x_flat[de,:] # all realizations, for all timesteps, of external states

prediction = sync.dot(mu_flat) # predicted external states = sigma(mu)


prediction_errors = eta_flat - prediction # prediction errors - difference between realization of external state and 'predicted' external state

prediction_errors = prediction_errors.reshape(T,N)


#start figure
plt.figure()
plt.clf()
plt.title('Prediction errors $\eta_t - \sigma(\mu_t)$')
#OU.plot_hot_colourline(np.arange(T), mean_F)


#set up heatmap of prediction error paths

min_y = prediction_errors.min()
max_y = prediction_errors.max()
no_bins = 50 #number of bins for frequency dataset
bins_pred_error = np.linspace(min_y,max_y, no_bins) #partition prediction error space
bin_interval = bins_pred_error[1]-bins_pred_error[0]
freq_pred_error = np.empty([T,no_bins])

for t in range(T):
    for i in range(no_bins):
        temp = bins_pred_error[i]
        freq_pred_error[t,i] = np.sum((prediction_errors[t,:] >= temp- bin_interval/2) * (prediction_errors[t,:] < temp + bin_interval/2))

time = range(T)
#plot heat map of prediction error paths

plt.contourf(time, bins_pred_error, freq_pred_error.T, levels=100, cmap='Blues')

#plot sample prediction error path
handle = plt.plot(range(T), prediction_errors[:,n], color = 'darkorange',linewidth=0.8,label='Sample path')

#set axis labels and save
plt.xlabel('Time')
plt.ylabel('$\eta_t - \sigma(\mu_t)$')
plt.legend(loc='lower right')
plt.savefig("Prediction_errors_time_3wayOU.png", dpi=100)