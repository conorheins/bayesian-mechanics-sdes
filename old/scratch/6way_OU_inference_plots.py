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

# sync map, expressed using stationary covariance
sync_map = S[np.ix_(eta, b)] @ pinv(S[np.ix_(mu, b)])

'''
Figure 1 - plot the marginal predictions for each external state, overlaid with true external state and marginal variances
'''

t_axis = np.arange(20000,50000) # range of points to use
mu_t = x[np.ix_(mu, t_axis, [0])].squeeze()
eta_t = x[np.ix_(eta, t_axis, [0])].squeeze()

posterior_means = sync_map.dot(mu_t)

posterior_stds = sqrtm(inv(Pi[np.ix_(eta,eta)]))

pred_upper_CI_mu0 = posterior_means[0]+ 1.96 * posterior_stds[0,0]
pred_lower_CI_mu0 = posterior_means[0]- 1.96 * posterior_stds[0,0]

pred_upper_CI_mu1 = posterior_means[mu[1]] + 1.96 * posterior_stds[1,1]
pred_lower_CI_mu1 = posterior_means[mu[1]] -1.96 * posterior_stds[1,1]

plt.figure(figsize=(15,12))
plt.clf()
plt.title('$\sigma(\mu_t)$ vs. $\eta_t$',fontsize=26, pad = 10)

plt.fill_between(t_axis,pred_upper_CI_mu0, pred_lower_CI_mu0, color='b', alpha=0.25)
mu1_mean_line = plt.plot(t_axis,posterior_means[0], color='b',label='Prediction: $q_{\mu_t}(\eta_1(t))$',lw=2.0)
eta1_real_line = plt.plot(t_axis, eta_t[0], color = 'r', label='Realized value: $\eta_1(t)$', lw = 1.5)

plot_offset = 10.0
plt.fill_between(t_axis,pred_upper_CI_mu1 + plot_offset, pred_lower_CI_mu1 + plot_offset, color='b', alpha=0.25)
mu2_mean_line = plt.plot(t_axis,posterior_means[1] + plot_offset, color='b',label='Prediction: $q_{\mu_t}(\eta_2(t))$',lw=2.0)
eta2_real_line = plt.plot(t_axis, eta_t[1] + plot_offset, color = 'r', label='Realized value: $\eta_2(t)$', lw = 1.5)

ci_patch = Patch(color='blue',alpha=0.1, label='Confidence intervals')

plt.legend(handles=[mu1_mean_line[0], eta1_real_line[0], mu2_mean_line[0], eta2_real_line[0], ci_patch], loc='lower left',fontsize=20, ncol = 3)

min_value = min( ( pred_lower_CI_mu0.min(), eta_t[0].min() ) )
max_value = max( ( (pred_upper_CI_mu1 + plot_offset).max(), (eta_t[1]+plot_offset).max() ) )

plt.xlim(t_axis[0], t_axis[-1])
plt.ylim(2 * min_value, 1.1 * max_value )

plt.gca().tick_params(axis='both', which='both', labelsize=25)
plt.gca().set_xlabel('Time',fontsize=30)
plt.gca().set_ylabel('$\eta_t$',fontsize=30)
plt.savefig("overlaid_predictions_plot.png")


'''
Figure 2 - plot the centered bivariate predictions for external states, overlaid with the analytic covariance / confidence intervals
'''

t_axis = np.arange(20000,T) # range of points to use
mu_t = x[np.ix_(mu, t_axis, [0])].squeeze()
eta_t = x[np.ix_(eta, t_axis, [0])].squeeze()

posterior_means = sync_map.dot(mu_t)

fig, ax = plt.subplots(figsize=(14,10))
plt.title('Prediction errors and covariance $\mathbf{\Pi}_{\eta}^{-1}$',fontsize=26, pad = 10)

prediction_errors = eta_t - posterior_means
prediction_centroid = np.mean(prediction_errors,axis=1) # mean over time should be [0.0, 0.0]

dots = ax.plot(prediction_errors[0,0::20],prediction_errors[1,0::20],'ko',markersize=3.5,label='Centered predictions')
posterior_cov = inv(Pi[np.ix_(eta,eta)])

def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

vals, vecs = eigsorted(posterior_cov)
theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

nstd = 3.0

height, width = 2 * nstd * np.sqrt(vals)
ellip = Ellipse(xy=prediction_centroid, width=width, height=height, angle=theta, alpha=0.1, color='blue')
ax.add_artist(ellip)
ci_patch = Patch(color='blue',alpha=0.1, label='Confidence ellipse')

plt.legend(handles=[dots[0], ci_patch], loc='lower right',fontsize=20)

min_value = prediction_errors[1].min()
max_value = prediction_errors[1].max()
plt.ylim(1.25 * min_value, 1.1 * max_value )

plt.gca().tick_params(axis='both', which='both', labelsize=25)
plt.gca().set_xlabel('$\eta_1 - q_\mathbf{\mu}(\eta_1)$',fontsize=30)
plt.gca().set_ylabel('$\eta_2 - q_\mathbf{\mu}(\eta_2)$',fontsize=30)
plt.savefig("prediction_error_cov_plot.png")
