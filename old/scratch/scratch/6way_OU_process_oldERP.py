import OUprocess_functions as OU
import numpy as np
from numpy.linalg import inv, pinv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse, Patch
from scipy.linalg import null_space as ker
from scipy.linalg import sqrtm
from scipy.stats import binned_statistic_2d
from utils import rank, itercheck_QB

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

B, Q, Pi, S = itercheck_QB(Q, D, Pi, zero_pairs = zero_clamped_entries, verbose = False, n_iter = 2)
assert np.isclose(OU.sylvester_equation_solve(B, D), np.linalg.inv(Pi)).all(), 'Solution to Sylvester equation does not agree with inverse of precision matrix'

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


'''
Now generate plots demonstrating predictions / prediction errors
'''

# Identify 'points-of-interest' i.e. times when a particular bin was occupied and 
# then plot the conditional expectation over external states using that bin as the prediction

# Run a bunch of parallel realizations to get sufficient statistics
process = OU.OU_process(dim=num_states, friction=B, volatility=volatility)  # create process
T = 2 * (10 ** 4)
n_realizations = 300 

x0 = np.random.multivariate_normal(mean=np.zeros(num_states), cov=S, size=n_realizations).T  # stationary samples
x = process.simulation(x0, dt, T, N=n_realizations)  # run simulation

ness_idx = int(10e2 / 2.0) # timestamp after which (N)ESS is assumed to have been reached

t_axis = np.arange(ness_idx,T)

mu_t = x[np.ix_(mu, t_axis,  np.arange(n_realizations))]
eta_t = x[np.ix_(eta, t_axis, np.arange(n_realizations))]

mu0_all, mu1_all = mu_t[0].flatten(), mu_t[1].flatten()
bin_counts, bin_edges_0, bin_edges_1, _ = binned_statistic_2d(mu0_all, mu1_all, [mu0_all, mu1_all], statistic='count', bins = 100)

mu_bin_idx_per_realization = np.zeros( mu_t.shape )
for r_i in range(n_realizations):
    _, _, _, mu_bin_idx_per_realization[:,:,r_i] = binned_statistic_2d(mu_t[0,:,r_i],mu_t[1,:,r_i], [mu_t[0,:,r_i], mu_t[1,:,r_i]], statistic='count', bins = (bin_edges_0, bin_edges_1), expand_binnumbers=True)

candidate_bins = np.where(bin_counts[0] >= 1000)

furthest_away_idx = np.argmax(candidate_bins[0] + candidate_bins[1])
# furthest_away_idx = np.argmin(candidate_bins[0] + candidate_bins[1])

highest_bin_idx = (candidate_bins[0][furthest_away_idx], candidate_bins[1][furthest_away_idx])

pre_stim_time = 50
post_stim_time = 100

time_locked_eta = np.zeros((1, 2, pre_stim_time+post_stim_time))
time_locked_eta_control = np.zeros((1, 2, pre_stim_time+post_stim_time))
time_locked_q   = np.zeros((1, 2, pre_stim_time+post_stim_time))
time_locked_mu   = np.zeros((1, 2, pre_stim_time+post_stim_time))

for r_i in range(n_realizations):
    realization_r = mu_bin_idx_per_realization[:,:,r_i]
    event_bin_idx = np.where(np.logical_and(realization_r[0,:] == highest_bin_idx[0]+1, realization_r[1,:] == highest_bin_idx[1]+1))[0]
    event_bin_idx = np.hstack( (event_bin_idx, realization_r.shape[1]) )

    too_close = np.hstack( (np.diff(event_bin_idx) < post_stim_time, True) )
    event_bin_idx = event_bin_idx[~too_close]

    too_early = event_bin_idx < pre_stim_time
    event_bin_idx[~too_early]

    if len(event_bin_idx) >= 1:
        random_event_idx = np.random.choice(pre_stim_time, T-post_stim_time, len(event_bin_idx))
        too_close = np.hstack( (np.diff(random_event_idx) < post_stim_time, True) )
        random_event_idx = random_event_idx[~too_close]

    try:

        time_locked_eta_r = np.zeros( (len(event_bin_idx), 2, pre_stim_time + post_stim_time) )
        time_locked_q_r  = np.zeros( (len(event_bin_idx), 2, pre_stim_time + post_stim_time) )
        time_locked_mu_r = np.zeros( (len(event_bin_idx), 2, pre_stim_time + post_stim_time) )

        for ii in range(len(event_bin_idx)):
            timestamp = event_bin_idx[ii]
            time_locked_eta_r[ii,:,:] = eta_t[:,(timestamp-pre_stim_time) : (timestamp+post_stim_time), r_i]
            time_locked_q_r[ii,:,:] = sync_map.dot(mu_t[:,(timestamp-pre_stim_time) : (timestamp+post_stim_time),r_i])
            time_locked_mu_r[ii,:,:] = mu_t[:,(timestamp-pre_stim_time) : (timestamp+post_stim_time),r_i]
        
        time_locked_eta_control_r = np.zeros( (len(event_bin_idx), 2, pre_stim_time + post_stim_time) )
        for ii in range(len(random_event_idx)):
            timestamp = random_event_idx[ii]
            time_locked_eta_control_r[ii,:,:] = eta_t[:,(timestamp-pre_stim_time) : (timestamp+post_stim_time), r_i]

        time_locked_eta  = np.vstack( (time_locked_eta, time_locked_eta_r))
        time_locked_q    = np.vstack( (time_locked_q, time_locked_q_r))
        time_locked_mu   = np.vstack( (time_locked_mu, time_locked_mu_r))
        time_locked_eta_control = np.vstack( (time_locked_eta_control, time_locked_eta_control_r) )

    except:
        print(f'Too few events for realization {r_i} for the desired bin...\n')

time_locked_eta = time_locked_eta[1:,:,:]
time_locked_q = time_locked_q[1:,:,:]
time_locked_mu = time_locked_mu[1:,:,:]

time_locked_eta_control = time_locked_eta_control[1:,:,:]

# %% Plot the estimates of the hidden state (w/ confidence intervals)

posterior_means = sync_map.dot(time_locked_mu.mean(axis=0))
# posterior_means = time_locked_q.mean(axis=0)

posterior_stds = sqrtm(inv(Pi[np.ix_(eta,eta)]))

# upper_CL = time_locked_q[:,0,:].mean(axis=0) + posterior_stds[0,0]
# lower_CL = time_locked_q[:,0,:].mean(axis=0) - posterior_stds[0,0]

upper_CL = posterior_means[0,:] + posterior_stds[0,0]
lower_CL = posterior_means[0,:] - posterior_stds[0,0]

plt.fill_between(np.arange(-pre_stim_time,post_stim_time), upper_CL, lower_CL, alpha = 0.5 )
# plt.plot(np.arange(-pre_stim_time,post_stim_time), time_locked_q[:,0,:].mean(axis=0),c= 'b')
# plt.plot(np.arange(-pre_stim_time,post_stim_time), time_locked_eta[:,0,:].mean(axis=0),c= 'r')

plt.plot(np.arange(-pre_stim_time,post_stim_time), posterior_means[0,:],c= 'b')
plt.plot(np.arange(-pre_stim_time,post_stim_time), time_locked_eta[:,0,:].mean(axis=0),c= 'r')
# plt.plot(np.arange(-pre_stim_time,post_stim_time), time_locked_eta_control[:,0,:].mean(axis=0),c= 'r')

offset = -1.5

# upper_CL = time_locked_q[:,1,:].mean(axis=0) + posterior_stds[1,1] + offset
# lower_CL = time_locked_q[:,1,:].mean(axis=0) - posterior_stds[1,1] + offset

upper_CL = posterior_means[1,:] + posterior_stds[1,1] + offset
lower_CL = posterior_means[1,:] - posterior_stds[1,1] + offset

plt.fill_between(np.arange(-pre_stim_time,post_stim_time), upper_CL, lower_CL, alpha = 0.5 )
# plt.plot(np.arange(-pre_stim_time,post_stim_time), time_locked_q[:,1,:].mean(axis=0) + offset, c= 'b')
# plt.plot(np.arange(-pre_stim_time,post_stim_time), time_locked_eta[:,1,:].mean(axis=0) + offset, c = 'r')

plt.plot(np.arange(-pre_stim_time,post_stim_time), posterior_means[1,:] + offset, c= 'b')
plt.plot(np.arange(-pre_stim_time,post_stim_time), time_locked_eta[:,1,:].mean(axis=0) + offset, c = 'r')
# plt.plot(np.arange(-pre_stim_time,post_stim_time), time_locked_eta_control[:,1,:].mean(axis=0)+offset,c= 'r')

