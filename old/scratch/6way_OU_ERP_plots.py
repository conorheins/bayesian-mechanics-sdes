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

pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}  # setup matplotlib to use latex for output
plt.style.use('seaborn-white')

np.random.seed(4) # ensure reproducibility

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

'''
Parameterise drift matrix, voltaility, and diffusion matrices
'''

desired_eigs = np.random.uniform(low = 0.1, high = 0.6, size = num_states)
B = OU.initialize_random_friction(desired_eigs)
eig_values = np.linalg.eigvals(B)
while (eig_values <= 0).any() or np.iscomplex(eig_values).any():
    print(f'Conditions for positive/real eigenvalues not met! Reinitializing...\n')
    B = OU.initialize_random_friction(desired_eigs)

diagonal_stds =  0.05 * np.ones(num_states) # variance of each individual variate of the process
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
Figure 3 - "ERP" style plots
Initialize the system from perturbed blanket (e.g. sensory or active) state while sampling the remaining variates from the
posterior distributions e.g. p(mu|s), p(eta|s), p(a|s)
'''

# Initialize the system from perturbed blanket (e.g. sensory or active) state while sampling the remaining variates from the
# posterior distributions e.g. p(mu|b) and p(eta|b)

process = OU.OU_process(dim=num_states, friction=B, volatility=volatility)  # create process
post_stim_time = 3000 # how long to look after the perturbation experiment
n_realizations = 500 # number of trials 
perturbation_amplitude = 5.0

x0 = np.zeros((num_states, n_realizations))
x0[s,:] = perturbation_amplitude

other_s = list(set(range(num_states)) - set([s]))

conditional_means = (S[np.ix_(other_s, [s])] @ inv(S[np.ix_([s],[s])])) * x0[s,:]

sigma_other = inv(Pi[np.ix_(other_s, other_s)])

for ri in range(n_realizations):
    x0[np.r_[other_s],ri] = np.random.multivariate_normal(mean= conditional_means[:,ri], cov=sigma_other, size=1)

# sample paths
perturbations_hist = process.simulation(x0, dt = dt, T = post_stim_time, N = n_realizations)  # run simulation

# %%  Plot the results

# sync map, expressed using stationary covariance
sync_map = S[np.ix_(eta, b)] @ pinv(S[np.ix_(mu, b)])

t_axis = np.arange(post_stim_time)

mean_trajectory = perturbations_hist.mean(axis=2)

# mean_eta_t = mean_trajectory[eta,:]
eta_t_samples = perturbations_hist[eta,:,:]

posterior_means = sync_map.dot(mean_trajectory[mu,:])
posterior_stds = sqrtm(inv(Pi[np.ix_(eta,eta)]))

pred_upper_CI_mu0 = posterior_means[0]+ 1.96 * posterior_stds[0,0]
pred_lower_CI_mu0 = posterior_means[0]- 1.96 * posterior_stds[0,0]

pred_upper_CI_mu1 = posterior_means[1] + 1.96 * posterior_stds[1,1]
pred_lower_CI_mu1 = posterior_means[1] - 1.96 * posterior_stds[1,1]

plt.figure(figsize=(19,12))
plt.clf()
plt.title('Average predictions $\sigma(\mu_t)$ and realized trajectories $\eta_t$',fontsize=30, pad = 10)

plt.fill_between(t_axis,pred_upper_CI_mu0, pred_lower_CI_mu0, color='b', alpha=0.15)
eta1_real_line = plt.plot(t_axis, eta_t_samples[0,:,::50], lw = 0.5, color = 'r', label='Sample paths: $\eta_1(t)$')
mu1_mean_line = plt.plot(t_axis,posterior_means[0], color='b',label='Average prediction: $\mathbf{\eta}_1(\mu_t)$',lw=2.0)

plot_offset = 0.4
plt.fill_between(t_axis,pred_upper_CI_mu1 + plot_offset, pred_lower_CI_mu1 + plot_offset, color='#3de3ac', alpha=0.25)
eta2_real_line = plt.plot(t_axis, eta_t_samples[1,:,::50] + plot_offset, lw = 0.5, color = '#933aab', label='Sample paths: $\eta_2(t)$')
mu2_mean_line = plt.plot(t_axis,posterior_means[1] + plot_offset, color='#3aab89',label='Average prediction: $\mathbf{\eta}_2(\mu_t)$',lw=2.0)

ci_patch_1 = Patch(color='blue',alpha=0.1, label='Confidence intervals $\mathbf{\eta}_1(\mu_t)$')
ci_patch_2 = Patch(color='#3de3ac',alpha=0.25, label='Confidence intervals $\mathbf{\eta}_2(\mu_t)$')

plt.legend(handles=[mu1_mean_line[0], eta1_real_line[0], mu2_mean_line[0], eta2_real_line[0], ci_patch_1, ci_patch_2], loc='lower left',fontsize=20, ncol = 3)

min_value = min( ( pred_lower_CI_mu0.min(), eta_t_samples[0].min() ) )
max_value = max( ( (pred_upper_CI_mu1 + plot_offset).max(), (eta_t_samples[1]+plot_offset).max() ) )

plt.xlim(t_axis[0], t_axis[-1])
plt.ylim(3 * min_value, 1.1 * max_value )

plt.gca().tick_params(axis='both', which='both', labelsize=25)
plt.gca().set_xlabel('Time',fontsize=30)
plt.gca().set_ylabel('External state space $\mathcal{E}$',fontsize=30)
plt.savefig("ERP_plot_multiple_realizations.png")
