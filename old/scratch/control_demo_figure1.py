import OUprocess_functions as OU
import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn-white')

np.random.seed(2) # ensure reproducibility

'''
Global parameters of the process
'''

# Dimensionality specification and partition assignment
num_states = 4 # number of variates in the OU process
mu, s, a, eta = 0, 1, 2, 3 # named indices for the different variables 

# Stochastic integration parameters
dt = 0.01  # duration of time-step 
T = 5 * 10 ** 4 # number of time-steps to integrate
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

diagonal_stds = 0.05 * np.ones(num_states) # variance of each individual variate of the process
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
Generate 2-dimensional surface of F(s, a, boldmu), where boldmu is a function of s and a
'''

num_grid_points = 150

domain_s = np.linspace(-1.1,1.1,num_grid_points)
domain_a = np.linspace(-1.1,1.1,num_grid_points)

# map from blanket state (s,a) to most-likely internal state (mu)
boldmu_b_map = lambda b: S[np.ix_(mu_idx, b_idx)].dot(np.linalg.inv(S[np.ix_(b_idx, b_idx)]).dot(b))    # expected interal state

K = np.linalg.inv(S[np.ix_( mu_idx + b_idx, mu_idx + b_idx)]) # relevant inverse of submatrix of stationary covariance (AKA 'control gains' or precisions)

vfe_surface = np.zeros( (len(domain_s), len(domain_a)) ) # the variational free energy of the sensory, active and most likely internal state, given blanket state (boldmu_b)
for ii, s_val in enumerate(domain_s):
    for jj, a_val in enumerate(domain_a):
        
        blanket_state = np.array([s_val, a_val])
        boldmu_b = boldmu_b_map(blanket_state)
        boldmu_and_b = np.concatenate([boldmu_b, blanket_state], axis = 0)
        vfe_surface[ii,jj] = boldmu_and_b.dot(K.dot(boldmu_and_b.reshape(-1,1))) # 2* log potential term
       

a_timecourse, s_timecourse = x[np.r_[a],:1500,0].squeeze(), x[np.r_[s],:1500,0].squeeze()

fig, ax = plt.subplots(figsize=(12,10))
colormap2use = plt.cm.get_cmap('PuBu_r')
ax.contourf(domain_a, domain_s, vfe_surface,levels = 200, cmap = colormap2use)  # plotting the free energy
ax.plot(a_timecourse,s_timecourse,c='#e34134',lw=1.0,alpha=1.0)
ax.set_title('Realization of $a_t, s_t$ on VFE surface',fontsize=32, pad = 25)
ax.set_xlabel('$a$',fontsize=30)
ax.set_ylabel('$s$',fontsize=30)
ax.tick_params(axis='both', which='both', labelsize=15)
cbar = fig.colorbar(cm.ScalarMappable(cmap=colormap2use), ax=ax)
cbar.set_label('$F(\mathbf{\mu}, a, s)$', fontsize=24, rotation = 270)
cbar.ax.get_yaxis().labelpad = 35
cbar.ax.tick_params(labelsize=15)

plt.savefig("active_sensory_states_VFE_descent.png")
