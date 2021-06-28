# %% If running in an IPYthon kernel (e.g. VSCode), the lines of code below are needed to add OUprocess functions to current Python path
from pathlib import Path
import os
import sys
sys.path.append(Path(os.getcwd()).parent)

# %%
import OUprocess_functions as OU
import utils
import numpy as np
from numpy.linalg import inv, pinv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse, Patch
from scipy.linalg import null_space as ker
from scipy.linalg import sqrtm
from utils import rank

plt.style.use('seaborn-white')

np.random.seed(3) # ensure reproducibility

'''
Global parameters of the process
'''

# Dimensionality specification and partition assignment
num_states = 6 # number of variates in the OU process

# named indices for the different variables
mu = [0, 1]
s = 2
a = 3
b = [s, a]
eta = [4, 5]
part = mu + b

# Stochastic integration parameters
dt = 0.01  # duration of time-step 
T = 10 * 10 ** 4 # number of time-steps to integrate
N = 1 # number of trajectories

'''
Parameterise drift matrix, voltaility, and diffusion matrices
'''

desired_eigs = np.random.uniform(low = 0.1, high = 0.6, size = num_states)
B = OU.initialize_random_friction(desired_eigs)
eig_values = np.linalg.eigvals(B)
while (eig_values <= 0).any() or np.iscomplex(eig_values).any():
    print(f'Conditions for positive/real eigenvalues not met! Reinitializing...\n')
    B = OU.initialize_random_friction(desired_eigs)

diagonal_stds =  np.ones(num_states) # variance of each individual variate of the process
volatility = np.diag(diagonal_stds)   # volatility matrix
noise_cov = volatility @ volatility.T # covariance of the random fluctuations
D = noise_cov / 2.0                   # diffusion matrix

'''
And then use Sylvester relations to solve
for the stationary covariance / precision (S and Pi, respectively), and the solenoidal operator Q.
'''
S = OU.sylvester_equation_solve(B, D) # solve for the stationary covariance by solving the Sylvester equation

Pi = np.linalg.inv(S)

Q = OU.sylvester_equation_solve(B, (B.dot(D) - D.dot(B.T)) / 2.0 ) # solve for the solenoidal flow by solving this Sylvester equation

'''
Enforce a Markov Blanket on the process via the Hessian to ensure desired conditional independencies.
Then resolve using an iterative least squares method (iteratively re-optimizing the solenoidal and drift matrices)
'''

zero_clamped_entries = [(mu, eta), (eta, mu)]

B, Q, Pi, S = utils.itercheck_QB(Q, D, Pi, zero_pairs = zero_clamped_entries, verbose = False, n_iter = 2)
assert np.isclose(OU.sylvester_equation_solve(B, D), np.linalg.inv(Pi)).all(), 'Solution to Sylvester equation does not agree with inverse of precision matrix'

# %% Check for existence of synchronization map

if rank(np.append(ker(Pi[np.ix_(mu, b)]), ker(Pi[np.ix_(eta, b)]), axis=1)) > rank(ker(Pi[np.ix_(eta, b)])):
    raise TypeError("Synchronisation map not well defined")

'''
Generate a stochastic realization of the process
'''

# initial condition
x0 = np.random.multivariate_normal(mean=np.zeros(num_states), cov=S, size=N).T  # stationary samples
x0 = np.zeros(num_states) + 1.0

process = OU.OU_process(dim=num_states, friction=B, volatility=volatility)  # create process
x = process.simulation(x0, dt, T, N)  # run simulation

# %% Conditional density over external states, parameterised by internal states

# sync map, expressed using stationary covariance
sync_map = S[np.ix_(eta, b)] @ pinv(S[np.ix_(mu, b)])

# sync map, expressed using precisions
# sync_map = inv(Pi[np.ix_(eta, eta)]) @ Pi[np.ix_(eta, b)] @ pinv(Pi[np.ix_(mu, b)]) @ Pi[np.ix_(mu, mu)]  

t_axis = np.arange(20000,25000)
mu_t = x[np.ix_(mu, t_axis, [0])].squeeze()
eta_t = x[np.ix_(eta, t_axis, [0])].squeeze()

posterior_means = sync_map.dot(mu_t)
posterior_stds = sqrtm(inv(Pi[np.ix_(eta,eta)]))

pred_upper_CI_mu0 = posterior_means[0]+ 1.96 * posterior_stds[0,0]
pred_lower_CI_mu0 = posterior_means[0]- 1.96 * posterior_stds[0,0]

pred_upper_CI_mu1 = posterior_means[mu[1]] + 1.96 * posterior_stds[1,1]
pred_lower_CI_mu1 = posterior_means[mu[1]] -1.96 * posterior_stds[1,1]

plt.figure(figsize=(12,8))
plt.clf()
plt.title('$\sigma(\mu_t)$ vs. $\eta_t$',fontsize=26, pad = 10)

plt.fill_between(t_axis,pred_upper_CI_mu0, pred_lower_CI_mu0, color='b', alpha=0.25)
plt.plot(t_axis,posterior_means[0], color='b',label='Prediction: $q_{\mu_t}(\eta_1(t))$',lw=2.0)
plt.plot(t_axis, eta_t[0], color = 'r', label='Realized value: $\eta_1(t)$', lw = 1.5)

plot_offset = 10.0
plt.fill_between(t_axis,pred_upper_CI_mu1 + plot_offset, pred_lower_CI_mu1 + plot_offset, color='b', alpha=0.25)
plt.plot(t_axis,posterior_means[1] + plot_offset, color='b',label='Prediction: $q_{\mu_t}(\eta_2(t))$',lw=2.0)
plt.plot(t_axis, eta_t[1] + plot_offset, color = 'r', label='Realized value: $\eta_2(t)$', lw = 1.5)

min_value = min( ( pred_lower_CI_mu0.min(), eta_t[0].min() ) )
max_value = max( ( (pred_upper_CI_mu1 + plot_offset).max(), (eta_t[1]+plot_offset).max() ) )

plt.xlim(t_axis[0], t_axis[-1])
plt.ylim(2 * min_value, 1.1 * max_value )

plt.legend(loc='lower left',fontsize=20, ncol = 2)

plt.gca().tick_params(axis='both', which='both', labelsize=25)
plt.gca().set_xlabel('Time',fontsize=30)
plt.gca().set_ylabel('$\mathbf{\eta}_t$',fontsize=30)

# %% Plot prediction error w/ analytic covariance 

fig, ax = plt.subplots(figsize=(12,8))
plt.title('Residuals and covariance $\Pi_{\eta}^{-1}$',fontsize=26, pad = 10)

prediction_errors = eta_t - posterior_means
prediction_centroid = np.mean(prediction_errors,axis=1) # mean over time should be [0.0, 0.0]

dots = ax.plot(prediction_errors[0,0::5],prediction_errors[1,0::5],'ko',label='Centered predictions')
posterior_cov = inv(Pi[np.ix_(eta,eta)])

def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

vals, vecs = eigsorted(posterior_cov)
theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

nstd = 3.0

# Width and height are "full" widths, not radius
height, width = 2 * nstd * np.sqrt(vals)
ellip = Ellipse(xy=prediction_centroid, width=width, height=height, angle=theta, alpha=0.1, color='blue')
ax.add_artist(ellip)
ci_patch = Patch(color='blue',alpha=0.1, label='Confidence ellipse')
# ax.add_artist(ci_patch)

plt.legend(handles=[dots[0], ci_patch], loc='lower right',fontsize=20)

min_value = prediction_errors[1].min()
max_value = prediction_errors[1].max()
plt.ylim(1.25 * min_value, 1.1 * max_value )

plt.gca().tick_params(axis='both', which='both', labelsize=25)
plt.gca().set_xlabel('$\eta_1 - q_\mathbf{\mu}(\eta_1)$',fontsize=30)
plt.gca().set_ylabel('$\eta_2 - q_\mathbf{\mu}(\eta_2)$',fontsize=30)


# %% Validate prediction by measuring how often the realized eta is within the predicted confidence intervals

start_idx = 20000

pred_means = (sync_map * x[mu,start_idx:,0]).squeeze()
pred_upper_CI = pred_means + 1.96 * np.sqrt(Pi[mu_idx,mu_idx]**-1)
pred_lower_CI = pred_means - 1.96 * np.sqrt(Pi[mu_idx,mu_idx]**-1)

realized_eta = x[eta, start_idx:, 0].squeeze()

outside_intervals = np.logical_or(realized_eta > pred_upper_CI, realized_eta < pred_lower_CI)

percent_outside = outside_intervals.mean()

print(f'True realized value within confidence intervals {100*(1 - percent_outside)} % of the total time')


# %% Repeatedly perturb external states with a fixed stimulus, and then measure the resulting 'ERP'

start_idx = int(1e4)

n_trials = 1000

stimulus_parameters = {'pre_stim_time': 100,         # pre-stimulus time (in timesteps)
                       'stim_kernel': 'boxcar',  # temporal kernel of stimulus input (options include: `boxcar`, `double_exponential`, `instant`, 'osc')
                       'stim_amplitude': -0.5,        # amplitude of stimulus. Exact meaning depends on the chosen stimulus kernel
                        'stim_timescale': 10}      # timescale parameter of stimulus (in timesteps). Exact meaning depends on the chosen stimulus kernel 


x_init = x[:,start_idx, 0] # the state of the system at the starting point of the perturbation
total_duration = 2000 # how long to run the perturbation experiments (in timesteps)

perturbed_trials = process.simulation_with_stim(x_init, 
                                              stimulus_parameters,
                                              experiment_type = 'stimulate',
                                              which_var = eta,
                                              n_trials = n_trials,
                                              dt=0.01, 
                                              T=total_duration)

mean_responses_stim = perturbed_trials.mean(axis=2)
std_responses_stim = perturbed_trials.std(axis=2)

# %% Run unperturbed control simulations
control_params = stimulus_parameters.copy()
control_params['stim_amplitude'] = 0.0

unperturbed_trials = process.simulation_with_stim(x_init, 
                                              control_params,
                                              experiment_type = 'stimulate',
                                              which_var = s,
                                              n_trials = n_trials,
                                              dt=0.01, 
                                              T=total_duration)


mean_responses_control = unperturbed_trials.mean(axis=2)
std_responses_control = unperturbed_trials.std(axis=2)

# %% 

idx2plot = mu

t_axis = np.arange(total_duration)
plt.figure(figsize=(12,8))
plt.clf()
plt.title('Stimulus-locked response of internal states to external perturbation',fontsize=26, pad = 25)

plt.fill_between(t_axis,mean_responses_stim[idx2plot,:]+std_responses_stim[idx2plot,:], mean_responses_stim[idx2plot,:]-std_responses_stim[idx2plot,:], color='r', alpha=0.25)
plt.plot(t_axis,mean_responses_stim[idx2plot,:], color='r',label='Stimulus',lw=3.0)

mean_responses_control = unperturbed_trials.mean(axis=2)
std_responses_control = unperturbed_trials.std(axis=2)

plt.fill_between(t_axis,mean_responses_control[idx2plot,:]+std_responses_control[idx2plot,:], mean_responses_control[idx2plot,:]-std_responses_control[idx2plot,:], color='b', alpha=0.25)
plt.plot(t_axis,mean_responses_control[idx2plot,:], color='b', label='No stimulus',lw=3.0)

min_value = min( (mean_responses_control[idx2plot,:]-std_responses_control[idx2plot,:]).min(), (mean_responses_stim[idx2plot,:]-std_responses_stim[idx2plot,:]).min())
max_value = max( (mean_responses_control[idx2plot,:]+std_responses_control[idx2plot,:]).max(), (mean_responses_stim[idx2plot,:]+std_responses_stim[idx2plot,:]).max())

plt.xlim(0, total_duration)
plt.ylim(1.15 * min_value, 1.25 * max_value )
plt.legend(loc='lower left',fontsize=20)

plt.gca().tick_params(axis='both', which='both', labelsize=25)
plt.gca().set_xlabel('Time',fontsize=30)
plt.gca().set_ylabel('$\mu_t$',fontsize=30)

# %%

sync_map = S[np.ix_(eta_idx, b_idx)] @ np.linalg.pinv(S[np.ix_(mu_idx, b_idx)])
bold_eta = sync_map * perturbed_trials[mu,:,:] 
bold_eta_unperturbed = sync_map * unperturbed_trials[mu,:,:] 

true_eta_perturbed = perturbed_trials[eta,:,:]
true_eta_unperturbed = unperturbed_trials[eta,:,:]

# %% Different approach:
# Identify 'points-of-interest' i.e. times when a particular bin was occupied and 
# then plot the conditional expectation over external states using that bin as the prediction

ness_idx = int(10e3) # timestamp after which (N)ESS is assumed to have been reached

mu_t = x[mu,ness_idx:,0]
eta_t = x[eta,ness_idx:,0]
bin_counts, bin_edges = np.histogram(mu_t, bins = 100)

half_bin_width = (np.diff(bin_edges)/2.0)[0]
bin_centers = bin_edges[:-1] + half_bin_width
filtered_bins = bin_centers[bin_counts >= 1000]

low_bin, high_bin = np.min(filtered_bins), np.max(filtered_bins)

low_bin_idx = np.where(np.logical_and(mu_t >= (low_bin - half_bin_width), mu_t <= (low_bin + half_bin_width)))[0]
high_bin_idx = np.where(np.logical_and(mu_t >= (high_bin - half_bin_width), mu_t <= (high_bin + half_bin_width)))[0]

low_bin_idx = np.hstack( (low_bin_idx, mu_t.shape[0]) )
high_bin_idx = np.hstack( (high_bin_idx, mu_t.shape[0]) )

# %% Find bin occupancy times that weren't too close to eachother

pre_stim_time = 10
post_stim_time = 50

too_close = np.hstack( (np.diff(low_bin_idx) < post_stim_time, True) )
candidate_low = low_bin_idx[~too_close]

too_close = np.hstack( (np.diff(high_bin_idx) < post_stim_time, True) )
candidate_high = high_bin_idx[~too_close]

time_locked_low = np.zeros( (len(candidate_low), pre_stim_time + post_stim_time, 2) )
time_locked_high = np.zeros( (len(candidate_high), pre_stim_time + post_stim_time, 2) )

for ii in range(len(candidate_low)):
    timestamp = candidate_low[ii]
    time_locked_low[ii,:,0] = eta_t[(timestamp-pre_stim_time) : (timestamp+post_stim_time)]
    time_locked_low[ii,:,1] = sync_map * mu_t[(timestamp-pre_stim_time) : (timestamp+post_stim_time)]

for ii in range(len(candidate_high)):
    timestamp = candidate_high[ii]
    time_locked_high[ii,:,0] = eta_t[(timestamp-pre_stim_time) : (timestamp+post_stim_time)]
    time_locked_high[ii,:,1] = sync_map * mu_t[(timestamp-pre_stim_time) : (timestamp+post_stim_time)]


'''
Figure 1: sync map OU process
'''

#bin blanket state-space
delta = 10**(-2)
bins = np.arange(np.min(x[b_idx, :, 0]), np.max(x[b_idx, :, 0]), delta/np.abs(float(eta)) )

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
            label='Prediction: $\sigma(\mathbf{\mu}(b))$')  # scatter plot theoretical expected internal state
plt.scatter(bins[j == 1], bold_eta_empirical[j == 1], s=1, alpha=0.5,
            label='Actual: $\mathbf{\eta}(b)$')  # scatter plot empirical external internal state
# plt.scatter(bins[j == 1], bold_eta_theoretical[j == 1], s=1, alpha=0.5,label='Theo: $\mathbf{\eta}(b)$')
plt.xlabel('Blanket state space $\mathcal{B}$')
plt.ylabel('External state space $\mathcal{E}$')
plt.legend(loc='upper right')
cor = scipy.stats.pearsonr(sync_bold_mu[j == 1], bold_eta_empirical[j == 1])
plt.title(f'Pearson correlation = {np.round(cor[0], 6)}...')
# plt.xlim(-4, 4)
# plt.ylim(-4, 4)
plt.savefig("sync_map_OUprocess.png")

# %%

ness_idx = int(10e3)

x_reduced = x[np.r_[mu_idx+ b_idx], ness_idx:,0].T

bin_counts, bin_edges = np.histogramdd(x_reduced, bins = 100)

binwise_means = [np.zeros(len(edges)-1) for edges in bin_edges]
vfe_surface = np.zeros([len(edges)-1 for edges in bin_edges])

bin_idx_all = [np.digitize(x_reduced[:,i], bin_edges[i], right = True) for i in range(3)]

bin_counts_all = [np.histogram(x_reduced[:,i], bin_edges[i])[0] for i in range(3)]

idx_count = 0

for mu_bin_i, s_bin_i, a_bin_i in zip(*bin_idx_all):

    binwise_means[0][mu_bin_i-1] += x_reduced[idx_count,0]
    binwise_means[1][s_bin_i-1] += x_reduced[idx_count,1]
    binwise_means[2][a_bin_i-1] += x_reduced[idx_count,2]
    idx_count += 1

binwise_means = [binwise_means[i] / bin_counts_all[i] for i in range(3)]

bin_centers = [edges[:-1] + 0.5*np.diff(edges) for edges in bin_edges]

sync_map_matrix = S[np.ix_(eta_idx, b_idx)] @ np.linalg.pinv(S[np.ix_(mu_idx, b_idx)])
sync_map_matrix = np.linalg.inv(Pi[np.ix_(eta_idx, eta_idx)]) @ Pi[np.ix_(eta_idx, b_idx)] @ np.linalg.pinv(Pi[np.ix_(mu_idx, b_idx)]) @ Pi[np.ix_(mu_idx, mu_idx)]  # sync map

eta_cond_mu = np.zeros_like(bin_centers[mu])
idx_count = 0
for mu_bin_i in bin_idx_all[mu]:
    eta_cond_mu[mu_bin_i-1] += x[eta, idx_count, 0]
    idx_count += 1

eta_cond_mu /= bin_counts_all[mu]

empirical_means = binwise_means[mu][bin_counts_all[mu] > 1000]
empirical_boldeta = eta_cond_mu[bin_counts_all[mu] > 1000]
plt.plot(bin_centers[mu][bin_counts_all[mu] > 1000], (sync_map_matrix * empirical_means).squeeze())
plt.scatter(bin_centers[mu][bin_counts_all[mu] > 1000], empirical_boldeta,c='b')

# %%
