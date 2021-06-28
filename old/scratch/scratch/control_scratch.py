# %% If running in an IPYthon kernel (e.g. VSCode), the lines of code below are needed to add OUprocess functions to current Python path
from pathlib import Path
import os
import sys
sys.path.append(Path(os.getcwd()).parent)

import OUprocess_functions as OU
import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn-white')

# %% Initialize an OU process with a 4-way partition, into sensory, active, internal, and external states

# num_states = 6 # number of variates in the OU process
# mu, s, a, eta = [0, 1], 2, 3, [4, 5] # named indices for the different variables 

num_states = 4 # number of variates in the OU process
mu, s, a, eta = 0, 1, 2, 3 # named indices for the different variables 

# Drift matrix
desired_eigs = np.random.uniform(low = 0.1, high = 0.6, size = num_states)
B = OU.initialize_random_friction(desired_eigs)
eig_values = np.linalg.eigvals(B)
while (eig_values <= 0).any() or np.iscomplex(eig_values).any():
    print(f'Conditions for positive/real eigenvalues not met! Reinitializing...\n')
    B = OU.initialize_random_friction(desired_eigs)

# this is a ready-made B matrix that results in approximate conditional independence of internal and external states
# B = np.array([[ 0.67926657, -0.43529797,  0.34025645,  0.15602715], \
#                [-0.14754452,  0.3603955 ,  0.19472513,  0.53870437], \
#                 [ 0.07367017, -0.23669304,  0.73608984,  0.53901522],
#                 [ 0.04825932, -0.13566684,  0.09499801,  0.74156862]] )

print(f'Drift coefficients:\n=======')
utils.matprint(B)
print(f'=======\n')

# volatility and diffusion
diagonal_stds = 0.05 * np.ones(num_states) # variance of each individual variate of the process
volatility = np.diag(diagonal_stds)   # volatility matrix
noise_cov = volatility @ volatility.T # covariance of the random fluctuations
D = noise_cov / 2.0                   # diffusion matrix

S = OU.sylvester_equation_solve(B, D) # solve for the stationary covariance by solving the Sylvester equation

print(f'Stationary covariance of the process:\n=======')
utils.matprint(S.round(3))
print(f'=======\n')

Pi = np.linalg.inv(S)
print(f'Precision (curvature) of the process:\n=======')
utils.matprint(Pi.round(3))
print(f'=======\n')

Q = OU.sylvester_equation_solve(B, (B.dot(D) - D.dot(B.T)) / 2.0 ) # solve for the solenoidal flow by solving this Sylvester equation
print(f'Skew-symmetric component of the flow of the process:\n=======')
utils.matprint(Q.round(3))
print(f'=======\n')

assert np.isclose((Q + D).dot(Pi), B).all(), 'Helmholtz decomposition not satisfied! Check your maths!'

# %% Enforce a Markov Blanket on the process to ensure desired conditional independencies

if not isinstance(mu, list):
    mu_idx = [mu]
else:
    mu_idx = mu

if not isinstance(eta, list):
    eta_idx = [eta]
else:
    eta_idx = eta

zero_clamped_entries = [(mu_idx, eta_idx), (eta_idx, mu_idx)]

B, Q, Pi, S = utils.itercheck_QB(Q, D, Pi, zero_pairs = zero_clamped_entries, verbose = False, n_iter = 2)
assert np.isclose(OU.sylvester_equation_solve(B, D), np.linalg.inv(Pi)).all(), 'Solution to Sylvester equation does not agree with inverse of precision matrix'

#%% Simulate the process

dt = 0.01  # time-step
T = 5 * 10 ** 4 # number of time-steps
N = 1 # number of trajectories

# initial condition
x0 = np.random.multivariate_normal(mean=np.zeros(num_states), cov=S, size=N).T  # stationary samples
x0 = np.zeros(num_states) + 1.0

process = OU.OU_process(dim=num_states, friction=B, volatility=volatility)  # create process
x = process.simulation(x0, dt, T, N)  # run simulation

plt.plot(x[:,10000:15000,:].squeeze().T)

# %% Show that the expected active state is minimizing free energy 

b_idx = [s, a]

if not isinstance(mu, list):
    mu_idx = [mu]
else:
    mu_idx = mu

K = np.linalg.inv(S[np.ix_( mu_idx + b_idx, mu_idx + b_idx)]) # control gains matrix

x_reduced = x[np.r_[mu_idx+ b_idx], :,0].T

bin_counts, bin_edges = np.histogramdd(x_reduced, bins = 100)

binwise_means = [np.zeros(len(edges)-1) for edges in bin_edges]
vfe_surface = np.zeros([len(edges)-1 for edges in bin_edges])

bin_idx_mu = np.digitize(x_reduced[:,0], bin_edges[0], right=True)
bin_idx_s = np.digitize(x_reduced[:,1], bin_edges[1], right=True)
bin_idx_a = np.digitize(x_reduced[:,2], bin_edges[2], right=True)

bin_counts_all = [np.histogram(x_reduced[:,i], bin_edges[i])[0] for i in range(3)]

idx_count = 0

for mu_i, s_i, a_i in zip(bin_idx_mu, bin_idx_s, bin_idx_a):

    binwise_means[0][mu_i-1] += x_reduced[idx_count,0]
    binwise_means[1][s_i-1] += x_reduced[idx_count,1]
    binwise_means[2][a_i-1] += x_reduced[idx_count,2]
    idx_count += 1

binwise_means = [binwise_means[i] / bin_counts_all[i] for i in range(3)]

bin_centers = [edges[:-1] + 0.5*np.diff(edges) for edges in bin_edges]

domain_mu = np.linspace(-1.1,1.1,100)
domain_s = np.linspace(-1.1,1.1,100)
domain_a = np.linspace(-1.1,1.1,100)

for ii, mu_val in enumerate(domain_mu):
    for jj, s_val in enumerate(domain_s):
        for kk, a_val in enumerate(domain_a):

            concat_state = np.array([mu_val, s_val, a_val])
            vfe_surface[ii,jj,kk] = concat_state.dot(K.dot(concat_state.reshape(-1,1)))

# Plot free energy landscape, cut at the bin corresponding to the most likely sensory state

plt.contourf(domain_mu, domain_a, vfe_surface[:,50,:],levels = 200, cmap = 'YlGnBu_r')  # plotting the free energy

plt.plot(x_reduced[:5000,0], x_reduced[:5000,2],c='#d42315',lw=0.75)


# %% Plot the variational free energy evolving over time

vfe_timeseries = (x_reduced.T * K.dot(x_reduced.T)).sum(axis=0)

plt.plot(vfe_timeseries[:1500])

# %% Scatter plot with different states colored by the free energy associated to them

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

time_indices = np.arange(start=30000, stop = 40000, step=1)

vfe_coloring = cm.magma(0.15*vfe_timeseries[time_indices])

chunk_to_show = x_reduced[time_indices,:]
ax.scatter(chunk_to_show[:,0], chunk_to_show[:,1], chunk_to_show[:,2], c=vfe_coloring, alpha = 0.3)
ax.set_xlabel('$\mu$',fontsize=30)
ax.set_ylabel('$s$',fontsize=30)
ax.set_zlabel('$a$',fontsize=30)


# plot the solution to the most likely active state, given sensory states

mu_levels = np.linspace(chunk_to_show[:,0].min(), chunk_to_show[:,0].max(),100)
s_levels = np.linspace(chunk_to_show[:,1].min(), chunk_to_show[:,1].max(),100)

xx, yy = np.meshgrid(mu_levels, s_levels)

most_likely_active = (S[a, s] * S[s,s] ** (-1)) * s_levels

z = np.tile(most_likely_active.reshape(-1,1), (1, 100))

ax.plot_surface(xx, yy, z, alpha=0.75)
# ax.plot_wireframe(xx, yy, z, rstride=5, cstride=5)
ax.view_init(15, 15)
plt.show()

# %% Plot most likely active state and the actual active state on top of eachother

# calculate most likely active state, given a particular bin of the sensory state

a_cond_s = np.zeros_like(bin_centers[s])

idx_count = 0
for s_i in bin_idx_s:
    a_cond_s[s_i-1] += x_reduced[idx_count,a]
    idx_count += 1

a_cond_s /= bin_counts_all[s]

plt.plot(bin_centers[s][bin_counts_all[s] > 1000], (S[a, s] * S[s,s] ** (-1)) * binwise_means[s][bin_counts_all[s] > 1000])

plt.scatter(bin_centers[s][bin_counts_all[s] > 1000], a_cond_s[bin_counts_all[s] > 1000],c='b')

# compute the difference between the realized active state and the most likely (FE-minimizing) active state
# prediction_errors = S[a,a]**(-1) *((S[a, s] * S[s,s] ** (-1)) * x_reduced[:,s] - x_reduced[:,a])**2


# # %%

# sync_map = lambda x: S[eta_idx, b_idx].dot(np.linalg.pinv(S[mu_idx, b_idx].reshape(1,-1)).dot(x))

# eta_cond_mu = np.zeros_like(bin_centers[mu])
# idx_count = 0
# for mu_i in bin_idx_mu:
#     eta_cond_mu[mu_i-1] += x[eta, idx_count, 0]
#     idx_count += 1

# eta_cond_mu /= bin_counts_all[mu]

# empirical_means = binwise_means[mu][bin_counts_all[mu] > 1000].reshape(1,-1)
# empirical_boldeta = eta_cond_mu[bin_counts_all[mu] > 1000]
# plt.plot(bin_centers[mu][bin_counts_all[mu] > 1000], sync_map(empirical_means))
# plt.scatter(bin_centers[mu][bin_counts_all[mu] > 1000], empirical_boldeta,c='b')

# %%

# # Get the line of most likely active and internal states, given a particular sensory state

# vfe_surfaces_by_s = np.transpose(vfe_surface, (1,0,2))
# most_likely_pairs = np.array([np.unravel_index(np.argmin(surface_s), (100,100)) for surface_s in vfe_surfaces_by_s])

# mu_values, a_values = domain_mu[most_likely_pairs[:,0]], domain_a[most_likely_pairs[:,1]]

# ax.plot(mu_values, domain_s, a_values)

# ax.axes.set_xlim3d(-0.2, 0.2)
# ax.axes.set_ylim3d(-0.15, 0.15)
# ax.axes.set_zlim3d(-0.25, 0.4)

# %% Generate 2-dimensional surface of F(s, a, boldmu), where boldmu is a function of s and a

domain_s = np.linspace(-1.1,1.1,150)
domain_a = np.linspace(-1.1,1.1,150)

boldmu_b_map = lambda b: S[np.ix_(eta_idx, b_idx)].dot(np.linalg.inv(S[np.ix_(b_idx, b_idx)]).dot(b))   # expected internal state
boldeta_b_map = lambda b: S[np.ix_(eta_idx, b_idx)].dot(np.linalg.inv(S[np.ix_(b_idx, b_idx)]).dot(b))  # expected external state
sync_map = lambda mu: S[np.ix_(eta_idx, b_idx)].dot(np.linalg.pinv(S[np.ix_(mu_idx, b_idx)]).dot(mu))   # synchronization map, not needed here

sync_map_precis = np.linalg.inv(Pi[np.ix_(eta_idx, eta_idx)]) * Pi[np.ix_(eta_idx, b_idx)] * np.linalg.pinv(Pi[np.ix_(mu_idx, b_idx)]) * Pi[np.ix_(mu_idx, mu_idx)] # the synchronization map in terms of precisions

vfe_surface_boldmu = np.zeros( (len(domain_s), len(domain_a)) )
for jj, s_val in enumerate(domain_s):
    for kk, a_val in enumerate(domain_a):
        
        blanket_state = np.array([s_val, a_val])
        boldmu_b = boldmu_b_map(blanket_state)
        boldmu_and_b = np.concatenate([boldmu_b, blanket_state], axis = 0)
        vfe_surface_boldmu[jj,kk] = boldmu_and_b.dot(K.dot(boldmu_and_b.reshape(-1,1)))


# nice colormaps
# colormap2use = plt.cm.get_cmap('GnBu')
# colormap2use = plt.cm.get_cmap('YlGnBu')
# colormap2use = plt.cm.get_cmap('YlGnBu_r')
# colormap2use = plt.cm.get_cmap('GnBu_r')
colormap2use = plt.cm.get_cmap('PuBu_r')

fig, ax = plt.subplots(figsize=(12,10))
# plt.suptitle('Realization of $a_t, s_t$ on VFE surface',fontsize=32)

ax.contourf(domain_a, domain_s, vfe_surface_boldmu,levels = 200, cmap = colormap2use)  # plotting the free energy
ax.plot(x_reduced[:1500,2], x_reduced[:1500,1],c='#e34134',lw=1.0,alpha=1.0)
ax.set_title('Realization of $a_t, s_t$ on VFE surface',fontsize=32, pad = 25)
ax.set_xlabel('$a$',fontsize=30)
ax.set_ylabel('$s$',fontsize=30)
ax.tick_params(axis='both', which='both', labelsize=15)
cbar = fig.colorbar(cm.ScalarMappable(cmap=colormap2use), ax=ax)
cbar.set_label('$F(\mathbf{\mu}, a, s)$', fontsize=24, rotation = 270)
cbar.ax.get_yaxis().labelpad = 35
cbar.ax.tick_params(labelsize=15)

plt.savefig("active_sensory_states_VFE_descent.png")

# %%
