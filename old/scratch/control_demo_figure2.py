# # %% If running in an IPYthon kernel (e.g. VSCode), the lines of code below are needed to add OUprocess functions to current Python path
# from pathlib import Path
# import os
# import sys
# sys.path.append(Path(os.getcwd()).parent)

import OUprocess_functions as OU
import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn-white')

np.random.seed(3) # ensure reproducibility

'''
Global parameters of the process
'''

# Dimensionality specification and partition assignment
num_states = 4 # number of variates in the OU process
mu, s, a, eta = 0, 1, 2, 3 # named indices for the different variables 

# Stochastic integration parameters
dt = 0.01  # duration of time-step 
T = 2 * 10 ** 4 # number of time-steps to integrate
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

mu_idx = [mu]
eta_idx = [eta]
b_idx = [s, a]

zero_clamped_entries = [(mu_idx, eta_idx), (eta_idx, mu_idx)]

B, Q, Pi, S = utils.itercheck_QB(Q, D, Pi, zero_pairs = zero_clamped_entries, verbose = False, n_iter = 2)
assert np.isclose(OU.sylvester_equation_solve(B, D), np.linalg.inv(Pi)).all(), 'Solution to Sylvester equation does not agree with inverse of precision matrix'

'''
Generate a stochastic realization of the process
'''

# initial condition
x0 = np.random.multivariate_normal(mean=np.zeros(num_states), cov=S, size=N).T  # stationary samples
x0 = np.zeros(num_states) + 1.0

process = OU.OU_process(dim=num_states, friction=B, volatility=volatility)  # create process
x = process.simulation(x0, dt, T, N)  # run simulation

'''
Perturbation experiments: perturb the sensory state at a timestep after the system
has relaxed to steady state and dissect the relaxation properties of the system.  
Realize multiple parallel trajectories and calculate ensemble averages.
'''

start_idx = int(1e4)

n_trials = 1000

stimulus_parameters = {'pre_stim_time': 100,         # pre-stimulus time (in timesteps)
                       'stim_kernel': 'boxcar',  # temporal kernel of stimulus input (options include: `boxcar`, `double_exponential`, `instant`, 'osc')
                       'stim_amplitude': -10,        # amplitude of stimulus. Exact meaning depends on the chosen stimulus kernel
                        'stim_timescale': 1000}      # timescale parameter of stimulus (in timesteps). Exact meaning depends on the chosen stimulus kernel 



x_init = x[:,start_idx, 0] # the state of the system at the starting point of the perturbation
total_duration = 2000 # how long to run the perturbation experiments (in timesteps)

perturbed_trials = process.simulation_with_stim(x_init, 
                                              stimulus_parameters,
                                              experiment_type = 'clamp',
                                              which_var = s,
                                              n_trials = n_trials,
                                              dt=0.01, 
                                              T=total_duration)
mean_responses_stim = perturbed_trials.mean(axis=2)
std_responses_stim = perturbed_trials.std(axis=2)

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

t_axis = np.arange(total_duration)

idx2plot = a

plt.figure(figsize=(12,8))
plt.clf()
plt.title('Stimulus-locked response of active states to external perturbation',fontsize=26, pad = 25)

plt.fill_between(t_axis,mean_responses_stim[idx2plot,:]+std_responses_stim[idx2plot,:], mean_responses_stim[idx2plot,:]-std_responses_stim[idx2plot,:], color='r', alpha=0.25)
plt.plot(t_axis,mean_responses_stim[idx2plot,:], color='r',label='Stimulus',lw=3.0)

plt.fill_between(t_axis,mean_responses_control[idx2plot,:]+std_responses_control[idx2plot,:], mean_responses_control[idx2plot,:]-std_responses_control[idx2plot,:], color='b', alpha=0.25)
plt.plot(t_axis,mean_responses_control[idx2plot,:], color='b', label='No stimulus',lw=3.0)

min_value = min( (mean_responses_control[idx2plot,:]-std_responses_control[idx2plot,:]).min(), (mean_responses_stim[idx2plot,:]-std_responses_stim[idx2plot,:]).min())
max_value = max( (mean_responses_control[idx2plot,:]+std_responses_control[idx2plot,:]).max(), (mean_responses_stim[idx2plot,:]+std_responses_stim[idx2plot,:]).max())

plt.xlim(0, total_duration)
plt.ylim(1.15 * min_value, 1.25 * max_value )
plt.legend(loc='lower left',fontsize=20)

plt.gca().tick_params(axis='both', which='both', labelsize=25)
plt.gca().set_xlabel('Time',fontsize=30)
plt.gca().set_ylabel('$a_t$',fontsize=30)

plt.savefig("active_states_evoked_response.png")
