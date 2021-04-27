'''
Simulations three way OU process
'''

import OU_process.OUprocess_functions as OU
import numpy as np
import matplotlib.pyplot as plt

#plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{bm}')
#plt.rc('text', usetex=True)
#plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}     # setup matplotlib to use latex for output      # change this if using xetex or lautex
plt.style.use('seaborn-white')
import scipy

'''
Setup
'''

dim = 3  # dimension of state-space
de = slice(0, 1)  # dimensions of external states
db = slice(1, 2)  # dimensions of blanket states (e.g. 2)
di = slice(2, 3)  # dimensions of internal states

# Drift matrix
B = np.array([2, 2, 2, 1, 2, -1, -1, 0, 2]).reshape([dim, dim])  # drift matrix
if np.any(np.linalg.eigvals(B) <= 0):
    print(np.linalg.eigvals(B))
    raise TypeError("Drift doesn't have positive spectrum")

# volatility and diffusion
sigma = np.eye(dim)  # np.eye(d)  # volatility matrix
D = sigma @ sigma.T / 2  # diffusion tensor

# We compute the stationary covariance and its inverse
S = np.empty([dim, dim])  # initialise stationary covariance
for i in range(dim):
    for j in range(dim):
        S[i, j] = scipy.integrate.quad(func=OU.integrand, a=0, b=np.inf, args=(B, sigma, i, j))[0]
Pi = np.linalg.inv(S)

# 1) We check it solves the Sylvester equation: BS + SB.T = 2D
# 2) we check that there are no numerical errors due to ill conditioning
error_sylvester = np.sum(np.abs(B @ S + S @ B.T - 2 * D))
error_inversion = np.sum(np.abs(S @ Pi - np.eye(dim)))
if np.round(error_sylvester, 7) != 0 or np.round(error_inversion, 10) != 0:
    raise TypeError("Sylvester equation not solved")

# We check that the stationary covariance is indeed positive definite
if np.any(np.linalg.eigvals(S) <= 0):
    print(np.linalg.eigvals(S))
    raise TypeError("Stationary covariance not positive definite")

# We check that the synchronisation map is well defined
if S[di, db] == 0:
    raise TypeError("Synchronisation map not well defined: bold_mu(b) not invertible!")

# define the linear synchronisation map
mu = S[di, db] * S[db, db] ** (-1)  # expected internal state
eta = S[de, db] * S[db, db] ** (-1)  # expected external state
sync = S[de, db] * S[di, db] ** (-1)  # sync map

'''
Run OU process simulation
'''
epsilon = 0.01  # time-step
T = 10 ** 4  # number of time-steps
N = 3*10 ** 2  # number of trajectories

# initial condition
# x0 = np.repeat(0, dim) #uniform samples
x0 = np.random.multivariate_normal(mean=np.zeros(dim), cov=S, size=N).T  # stationary samples

process = OU.OU_process(dim=dim, friction=B, volatility=sigma)  # create process
x = process.simulation(x0, epsilon, T, N)  # run simulation

'''
Figure 1
'''
# For each blanket state we plot
# the empirical expected external state
# with the sync(empirical expected internal state)
# to check that the synchronisation map works

delta = 10 ** (-2)  # size of bins
bins = np.arange(np.min(x[db, :, :]), np.max(x[db, :, :]), delta)  # partition blanket state-space into several bins
j = np.zeros(bins.shape)

bold_eta_empirical = np.empty(bins.shape)
sync_bold_mu = np.empty(bins.shape)
bold_mu_empirical = np.empty(bins.shape)

for i in range(len(bins)):
    indices = (x[db, :, :] >= bins[i]) * (
            x[db, :, :] <= bins[i] + delta)  # select observations where blanket state is in desired bin
    indices = indices.reshape([T, N])
    if np.sum(indices) > 1000:  # if there are a sufficient amount of samples
        j[i] = 1  # score that we are using this bin
        eta_samples = x[de, indices]  # select samples of internal states given blanket states
        bold_eta_empirical[i] = np.mean(eta_samples)  # select empirical expected external state given blanket state
        mu_samples = x[di, indices]  # select samples of internal states given blanket states
        bold_mu_empirical[i] = np.mean(mu_samples)  # empirical expected internal state
        sync_bold_mu[i] = sync * bold_mu_empirical[i]  # synchronisation map of empirical expected internal state

plt.figure(1)
plt.clf()
plt.suptitle('Synchronisation map')
plt.scatter(bins[j == 1], sync_bold_mu[j == 1], s=1,
            label='Prediction: $\sigma(\mathbf{\mu}(b))$')  # scatter plot theoretical expected internal state
plt.scatter(bins[j == 1], bold_eta_empirical[j == 1], s=1,
            label='Actual: $\mathbf{\eta}(b)$')  # scatter plot empirical external internal state
plt.xlabel('Blanket state space $\mathcal{B}$')
plt.ylabel('External state space $\mathcal{E}$')
plt.legend(loc='upper right')
plt.savefig("Prediction_sync_map.png")

'''
Figure 2
'''
#correlation plot

plt.figure(2)
plt.clf()
plt.suptitle('Correlation plot')  # between sync(bold_mu) and bold_eta
plt.scatter(sync_bold_mu[j == 1], bold_eta_empirical[j == 1], s=1)
cor = scipy.stats.pearsonr(sync_bold_mu[j == 1], bold_eta_empirical[j == 1])
plt.xlabel('$\sigma(\mathbf{\mu}(b))$')
plt.ylabel('$\mathbf{\eta}(b)$')
plt.title(f'Pearson correlation ={cor[0]}')
plt.savefig("Corplot_sync_map.png")


del x

'''
Figure 3
'''
# Heat map of free energy as defined on a plane with internal and blanket states
# On this plane plot the most likely internal state as a function of blanket states
# Then plot a trajectory of the process that's starting from far away


# start a trajectory at a really high free energy
y0 = np.zeros(dim) + 10
length = 500
y = process.simulation_float128(y0, epsilon, length, 2)  # run simulation


Pi = np.linalg.inv(S[:2, :2])  # precision matrix of generative model of external and blanket states

def F(internal, blanket):  # computes free energy up to an additive constant
    Z = np.outer(internal, blanket)
    for j in range(len(blanket)):
        b = blanket[j]
        bold_eta = eta * b
        temp = np.append(bold_eta, b)
        for i in range(len(internal)):
            Z[i, j] = temp @ Pi @ temp  # 2* potential term
            Z[i, j] = Z[i, j] + Pi[de, de] * (sync * internal[i] - bold_eta) ** 2  # additional term
    return Z / 2

internal = np.linspace(np.min(y[di,:,:])-1, np.max(y[di,:,:])+0.5, 105)  # internal state-space points
blanket = np.linspace(np.min(y[db,:,:])-1, np.max(y[db,:,:])+0.5, 100)  # blanket state-space points

Z = F(internal, blanket)  # free energy up to additive constant


plt.figure(3)
plt.clf()
plt.suptitle('Free energy $F(\mu, b)$')
plt.contourf(blanket, internal, Z, levels=100, cmap= 'turbo')  # plotting the free energy
plt.colorbar()
plt.ylabel('internal state $ \mu$')
plt.xlabel('blanket state $b$')
plt.plot(blanket, float(mu) * blanket, c='white')  # plot expected internal state as a function of blanket states
#plt.plot(y[db, :, 0].reshape([length]), y[di, :, 0].reshape([length]), c='black', linewidth = 0.5)  # plot trajectory starting at very high free energy
OU.plot_hot_colourline(y[db, :, 0].reshape([length]), y[di, :, 0].reshape([length]), lw=0.5)
plt.savefig("Sample_FE_large.png")


'''
Figure 4
'''
# plot the average free energy over time of many trajectories starting at a point of high free energy

# start many trajectories at a really high free energy (same starting point as last figure)
z0 = y0
length_avFE = 200
N_avFE = 10000 #number of trajectories
z = process.simulation_float128(z0, epsilon, length_avFE, N_avFE)  # run simulation

#compute average free energy over time
F_z = np.empty([length_avFE, N_avFE])
for t in range(length_avFE):
    for n in range(N_avFE):
        F_z[t,n] = F(z[di, t,n], z[db, t,n]) #compute free energy
mean_F = np.mean(F_z, axis=1) #take mean over trajectories

plt.figure(4)
plt.clf()
plt.suptitle('Average free energy of perturbed system')
OU.plot_hot_colourline(np.arange(length_avFE), mean_F)
plt.xlabel('Time')
plt.ylabel('Free energy')
plt.savefig("FE_vs_time_FE_large.png")


'''
Figure 5
'''
#plot correlation and variance of trajectories starting at a high free energy
#(correlation between mu, b)

cors = np.empty([length_avFE-1]) #will store correlations here
for t in range(1,length_avFE):
    cors[t-1] = scipy.stats.pearsonr(z[di,t,:].reshape([N_avFE]) , z[db,t,:].reshape([N_avFE]))[0]

plt.figure(5)
plt.clf()
plt.suptitle('Correlation of trajectories: perturbed system')
OU.plot_hot_colourline(np.arange(1,length_avFE), cors)
plt.savefig("Cor_FE_large.png")

'''
Figure 6
'''
#plot variances along each axis of trajectories starting at a high free energy

vars = np.empty([length_avFE, 2]) #will store correlations here
for t in range(length_avFE):
    vars[t,0] = np.var(z[db,t,:]) #variance along blanket states
    vars[t,1] = np.var(z[di,t,:]) #variance along internal states

plt.figure(6)
plt.clf()
plt.suptitle('Variances of trajectories: perturbed system')
OU.plot_hot_colourline(np.arange(length_avFE), vars[:,0]) #variance along blanket states
OU.plot_hot_colourline(np.arange(length_avFE), vars[:,1]) #variance along external states
plt.savefig("Var_FE_large.png")



'''
We replicate Figure 3-6 but now we start the dynamic at the free energy minimum
'''

'''
Figure 7
'''
# Heat map of free energy as defined on a plane with internal and blanket states
# On this plane plot the most likely internal state as a function of blanket states
# Then plot a trajectory of the process that's starting from the free energy minimum

# start a trajectory at free energy minimum
y0 = np.zeros(dim)
length = 500
y = process.simulation_float128(y0, epsilon, length, 2)  # run simulation


Pi = np.linalg.inv(S[:2, :2])  # precision matrix of generative model of external and blanket states

def F(internal, blanket):  # computes free energy up to an additive constant
    Z = np.outer(internal, blanket)
    for j in range(len(blanket)):
        b = blanket[j]
        bold_eta = eta * b
        temp = np.append(bold_eta, b)
        for i in range(len(internal)):
            Z[i, j] = temp @ Pi @ temp  # 2* potential term
            Z[i, j] = Z[i, j] + Pi[de, de] * (sync * internal[i] - bold_eta) ** 2  # additional term
    return Z / 2

internal = np.linspace(np.min(y[di,:,:])-1, np.max(y[di,:,:])+0.5, 105)  # internal state-space points
blanket = np.linspace(np.min(y[db,:,:])-1, np.max(y[db,:,:])+0.5, 100)  # blanket state-space points

Z = F(internal, blanket)  # free energy up to additive constant


plt.figure(7)
plt.clf()
plt.suptitle('Free energy $F(\mu, b)$')
plt.contourf(blanket, internal, Z, levels=100, cmap= 'turbo')  # plotting the free energy
plt.colorbar()
plt.ylabel('internal state $ \mu$')
plt.xlabel('blanket state $b$')
plt.plot(blanket, float(mu) * blanket, c='white')  # plot expected internal state as a function of blanket states
#plt.plot(y[db, :, 0].reshape([length]), y[di, :, 0].reshape([length]), c='black', linewidth = 0.5)  # plot trajectory starting at very high free energy
OU.plot_cool_colourline(y[db, :, 0].reshape([length]), y[di, :, 0].reshape([length]), lw=0.5)
plt.savefig("Sample_FE_minim.png")

'''
Figure 8
'''
# plot the average free energy over time of many trajectories starting at the free energy minimum

# start many trajectories at free energy minimum
z0 = y0
length_avFE = 200
N_avFE = 10000 #number of trajectories
z = process.simulation_float128(z0, epsilon, length_avFE, N_avFE)  # run simulation

#compute average free energy over time
F_z = np.empty([length_avFE, N_avFE])
for t in range(length_avFE):
    for n in range(N_avFE):
        F_z[t,n] = F(z[di, t,n], z[db, t,n]) #compute free energy
mean_F = np.mean(F_z, axis=1) #take mean over trajectories

plt.figure(8)
plt.clf()
plt.suptitle('Average free energy of system starting at FE minimum')
OU.plot_cool_colourline(np.arange(length_avFE), mean_F)
plt.xlabel('Time')
plt.ylabel('Free energy')
plt.savefig("FE_vs_time_FE_minim.png")


'''
Figure 9
'''
#correlation plot
#plot correlation and variance of trajectories starting at free energy minimum
#(correlation between mu, b)

cors = np.empty([length_avFE-1]) #will store correlations here
for t in range(1,length_avFE):
    cors[t-1] = scipy.stats.pearsonr(z[di,t,:].reshape([N_avFE]) , z[db,t,:].reshape([N_avFE]))[0]

plt.figure(9)
plt.clf()
plt.suptitle('Correlation of trajectories: perturbed system starting at FE minimum')
OU.plot_cool_colourline(np.arange(1,length_avFE), cors)
plt.savefig("Cor_FE_minim.png")


'''
Figure 10
'''


vars = np.empty([length_avFE, 2]) #will store correlations here
for t in range(length_avFE):
    vars[t,0] = np.var(z[db,t,:]) #variance along blanket states
    vars[t,1] = np.var(z[di,t,:]) #variance along internal states

plt.figure(10)
plt.clf()
plt.suptitle('Variances of trajectories: perturbed system starting at FE minimum')
OU.plot_cool_colourline(np.arange(length_avFE), vars[:,0]) #variance along blanket states
OU.plot_cool_colourline(np.arange(length_avFE), vars[:,1]) #variance along external states
plt.savefig("Var_FE_minim.png")

