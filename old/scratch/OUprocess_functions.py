'''
Functions related to OU processes
Imported from relative entropy 25 March 2021
'''

'''
Imports
'''
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.signal import convolve
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn

'''
Process
'''

class OU_process(object):
# dX_t = -B X_t + s dW_t
    def __init__(self, dim, friction, volatility):
        super(OU_process, self).__init__()  # constructs the object instance
        self.d = dim
        self.B = friction
        self.sigma = volatility

    def simulation(self, x0, dt=0.01, T=100, N=1):  # run OU process for multiple trajectories
    
        w = np.random.normal(0, np.sqrt(dt), (T - 1) * self.d * N).reshape([self.d, T - 1, N])
        # Euler-Maruyama scaling constant (to account for standard deviation of fluctuations per time interval, whose standard deviation will be sqrt(dt))

        x = np.empty([self.d, T, N])  # store values of the process
        if x0.shape == x[:, 0, 0].shape:
            x[:, 0, :] = np.tile(x0, N).reshape([self.d, N])  # initial condition
        elif x0.shape == x[:, 0, :].shape:
            x[:, 0, :] = x0
        else:
            raise TypeError("Initial condition has wrong dimensions")
        for t in range(1, T):
        
            x[:, t, :] = x[:, t - 1, :] - dt * np.tensordot(self.B, x[:, t - 1, :], axes=1) \
                         + np.tensordot(self.sigma, w[:, t - 1, :], axes=1)

            if np.count_nonzero(np.isnan(x)):
                raise TypeError("nan")
        return x

    def simulation_float128(self, x0, dt=0.01, T=100, N=1):  # run OU process for multiple trajectories float 128
        w = np.random.normal(0, np.sqrt(dt), (T - 1) * self.d * N).reshape([self.d, T - 1, N])  
        # Euler-Maruyama scaling constant (to account for standard deviation of fluctuations per time interval, whose standard deviation will be sqrt(dt))

        x = np.empty([self.d, T, N])  # store values of the process
        if x0.shape == x[:, 0, 0].shape:
            x[:, 0, :] = np.tile(x0, N).reshape([self.d, N])  # initial condition
        elif x0.shape == x[:, 0, :].shape:
            x[:, 0, :] = x0
        else:
            raise TypeError("Initial condition has wrong dimensions")
        redo = 1
        while redo > 0:
            for t in range(1, T):
                
                x[:, t, :] = x[:, t - 1, :] - dt * np.tensordot(self.B, x[:, t - 1, :], axes=1) \
                         + np.tensordot(self.sigma, w[:, t - 1, :], axes=1)

                if redo == 2 and not np.all(np.isfinite(x)):  # if there is a nan and we have redone the loop
                    raise TypeError("nan")  # returns an error
                elif not np.all(np.isfinite(x)):  # if there is a nan in x
                    x = np.array(x, dtype=np.float128)  # update to float128
                    print('redo')
                    redo = 2  # scores that we redid the loop
                else:
                    redo = 0  # there is no nan/inf in x so we can leave the loop
        return x
    
    def simulation_with_stim(self, x0, stimulus_params_dict, experiment_type = 'stimulate', which_var = None, n_trials = 1, dt=0.01, T=100):  # run OU process with external stimulus applied to a particular stimulus
        
        if which_var is None:
            print('No particular variable(s) chosen for stimulation, defaulting to stimulating all variables...\n')
            which_var = list(range(x0.shape[0]))
        else:
            if isinstance(which_var, int):
                which_var = [which_var]
            elif not isinstance(which_var, (list, np.ndarray)):
                raise TypeError("type(which_var) must be an int, list or numpy array")

        stimulus_train = generate_input_kernel(**stimulus_params_dict)
        
        if experiment_type == 'stimulate':
            stimulus_train = np.pad(stimulus_train, (0, T - len(stimulus_train)), mode = 'constant')

        w = np.random.normal(0, np.sqrt(dt), (T - 1) * self.d * n_trials).reshape([self.d, T - 1, n_trials])
        # Euler-Maruyama scaling constant (to account for standard deviation of fluctuations per time interval, whose standard deviation will be sqrt(dt))

        x = np.empty([self.d, T, n_trials])  # store values of the process
        if x0.shape == x[:, 0, 0].shape:
            x[:, 0, :] = np.tile(x0, n_trials).reshape([self.d, n_trials])  # initial condition
        elif x0.shape == x[:, 0, :].shape:
            x[:, 0, :] = x0
        else:
            raise TypeError("Initial condition has wrong dimensions")

        if experiment_type == 'clamp': # Version 1

            for t in range(1, T):
        
                x[:, t, :] = x[:, t - 1, :] - dt * np.tensordot(self.B, x[:, t - 1, :], axes=1) \
                            + np.tensordot(self.sigma, w[:, t - 1, :], axes=1)

                # Version 1: clamp the chosen variates of the process with values of the stimulus train
                if t < len(stimulus_train):
                    x[ np.r_[which_var], t, :] = stimulus_train[t]
            
                if np.count_nonzero(np.isnan(x)):
                    raise TypeError("nan")

        elif experiment_type == 'stimulate': # Version 2

            for t in range(1, T):
            
                x[:, t, :] = x[:, t - 1, :] - dt * np.tensordot(self.B, x[:, t - 1, :], axes=1) \
                            + np.tensordot(self.sigma, w[:, t - 1, :], axes=1)
                
                # Version 2: additively perturb the chosen variates of the process with the stimulus train
                x[ np.r_[which_var], t, :] = x[ np.r_[which_var], t, :] + stimulus_train[t]

                if np.count_nonzero(np.isnan(x)):
                    raise TypeError("nan")
        return x

    def showpaths(self, x, color='blue', label='helloword'):  # compute and plot statistics of the simulation
        [d, T, N] = np.shape(x)
        '''
        Sample trajectory
        '''

        plt.figure(1)
        plt.suptitle("OU process trajectory")
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.plot(x[0, :, 0], x[1, :, 0], c=color, linewidth=0.1, label=label)
        plt.legend()
        plt.savefig("OUprocess.trajectory.2D.png")

    def showentropy(self, x, nbins=10, color='blue',
                    label='helloword'):  # compute and plot statistics of the simulation
        [d, T, N] = np.shape(x)
        '''
        Entropy
        '''
        plt.figure(2)
        plt.suptitle("Entropy OU process")

        entropy_x = entropy(x, nbins)

        plt.xlabel('time')
        plt.ylabel('H[p(x)]')
        plt.plot(range(T), entropy_x, c=color, linewidth=0.5, label=label)
        plt.legend()
        plt.savefig("OUprocess.entropy.2D.png")

    def attributes(self):
        return [self.B, self.sigma]

    def stationary_covariance(self):
        n = self.d
        S = np.empty(n)  # initialise stationary covariance
        for i in range(n):
            for j in range(n):
                S[i, j] = scipy.integrate.quad(func=integrand, a=0, b=np.inf, args=(self.B, self.sigma, i, j))[0]
        return S

    def stat_cov2D(self):  # compute stationary covariance using 2D analytical formula
        if self.d != 2:
            raise TypeError("This code is exclusively for 2D!")
        D = self.sigma @ self.sigma.T / 2  # diffusion matrix
        a = self.B[0, 0]
        b = self.B[0, 1]
        c = self.B[1, 0]
        d = self.B[1, 1]
        u = D[0, 0]
        v = D[1, 1]
        w = D[1, 0]
        detB = np.linalg.det(self.B)
        S = np.array([(detB + d ** 2) * u + b ** 2 * v - 2 * b * d * w,
                      -c * d * u - a * b * v + 2 * a * d * w,
                      -c * d * u - a * b * v + 2 * a * d * w,
                      c ** 2 * u + (detB + a ** 2) * v - 2 * a * c * w]).reshape(2,
                                                                                 2)  # analytical formula for stationary covariance
        return S / ((a + d) * detB)


'''
Functions
'''


def entropy(x, nbins=10):  # estimate entropy of sample
    [d, T, N] = np.shape(x)
    b_range = bin_range(x)
    entropy = np.zeros(T)
    for t in range(T):
        h = np.histogramdd(x[:, t, :].T, bins=nbins, range=b_range)[0]
        entropy[t] = - np.sum(h / N * np.log(h + (h == 0)))
    return entropy + np.log(N)


def inst_epr(x, epsilon, nbins=10):  # compute instantaneous EPR
    [d, T, N] = np.shape(x)
    b_range = bin_range(x) * 2  # double the list

    inf_epr = np.zeros([T - 1])

    for t in range(T - 1):
        x_t = x[:, t:t + 2, :]  # trajectories at time t, t+1
        x_mt = np.flip(x_t, axis=1)  # trajectories at time t+1, t (time reversal)
        x_t = x_t.reshape([d * 2, N])  # samples from (x_t, x_t+1)
        x_mt = x_mt.reshape([d * 2, N])  # samples from (x_t+1, x_t) (time reversal)
        e = np.histogramdd(x_t.T, bins=nbins, range=b_range)[0]  # law of (x_t, x_t+1) (unnormalised)
        h = np.histogramdd(x_mt.T, bins=nbins, range=b_range)[0]  # law of (x_t+1, x_t) (unnormalised)
        # nonzero = (e != 0)*(h != 0)
        # inf_epr[t]= np.sum(np.where(nonzero, e/N * np.log(e / h), 0)) / epsilon
        nonzero = (e != 0) * (h != 0)  # shows where e and h are non-zero
        zero = (nonzero == 0)  # shows where e or h are zero
        inf_epr[t] = np.sum(
            e / (N * epsilon) * np.log((e * nonzero + zero) / (h * nonzero + zero)))  # 1/epsilon * KL divergence
    return inf_epr  # 1/epsilon * KL divergence


def epr_via_inst(process, N, T, epsilon, steps=10, bins=10):
    d = process.attributes()[1].shape[1]
    if d == 2:
        S = process.stat_cov2D()  # stationary covariance
    else:
        S = process.stationary_covariance()
    x = np.random.multivariate_normal(mean=np.zeros([d]), cov=S,
                                      size=[N, 1]).T  # generate initial condition at steady-state (since known)
    t = 0
    epr_inst = -np.ones([T])  # instantaneous entropy production rate
    while t < T:
        # steps = 10  # number of steps in a simulation
        x = process.simulation(x[:, -1, :], epsilon, T=steps, N=N)
        # pos[:, t:(t + steps), 0, i] = x[:, :, 0]  # record positions
        epr_inst[t:(t + steps - 1)] = inst_epr(x, epsilon,
                                               nbins=bins)  # record instantaneous entropy production
        t += steps
    epr_v = np.mean(epr_inst[epr_inst >= 0])
    epr_v_median = np.median(epr_inst[epr_inst >= 0])
    return epr_v, epr_v_median


def ent_prod_rate_2D(process):  # analytical formula for the 2D epr
    ''' There seems to be a problem with this formula as I have seen it return negative values'''
    # TODO: update with trace formula which is valid in any dimension (and using pinv)
    [B, sigma] = process.attributes()[0:2]
    D = sigma @ sigma.T / 2  # diffusion matrix
    a = B[0, 0]
    b = B[0, 1]
    c = B[1, 0]
    d = B[1, 1]
    u = D[0, 0]
    v = D[1, 1]
    w = D[1, 0]
    q = c * u - b * v + (d - a) * w  # irreversibility parameter
    phi = q ** 2 / ((a + d) * np.linalg.det(D))
    return phi


def epr_int_MC(process, N):  # integral formula for EPR elliptic process
    [B, sigma] = process.attributes()
    d = B.shape[1]  # dimension
    D = sigma @ sigma.T / 2
    if np.linalg.det(D) == 0:
        D_pinv = np.linalg.pinv(D, rcond=10 ** (-15))
        print('Using pseudo-inverse')
    else:
        D_pinv = np.linalg.inv(D)
    if d == 2:
        S = process.stat_cov2D()
    else:
        S = process.stationary_covariance()
    if np.linalg.det(S) == 0:
        print(S)
        raise TypeError('Not hypoelliptic')
    e_p = 0
    x = np.random.multivariate_normal(mean=np.zeros([d]), cov=S, size=[N]).T  # generate stationary samples
    J = B - D @ np.linalg.inv(S)
    for i in range(N):
        Jx = J @ x[:, i]
        e_p += Jx.T @ D_pinv @ Jx / N
    return e_p


def epr_int_MC2(process,
                N):  # integral formula for EPR elliptic process (same as before only uses pinv at a different place)
    [B, sigma] = process.attributes()
    d = B.shape[1]  # dimension
    D = sigma @ sigma.T / 2
    D_pinv = np.linalg.pinv(D)
    if d == 2:
        S = process.stat_cov2D()
    else:
        S = process.stationary_covariance()
    if np.linalg.det(S) == 0:
        print(S)
        raise TypeError('Not hypoelliptic')
    e_p = 0
    x = np.random.multivariate_normal(mean=np.zeros([d]), cov=S, size=[N]).T  # generate stationary samples
    J = D_pinv @ B - np.linalg.inv(S)
    for i in range(N):
        Jx = J @ x[:, i]
        e_p += Jx.T @ D @ Jx / N
    return e_p


'''These work when B = sigma @ A (e.g. elliptic)'''


def epr_BsigmaA(process, hypo=False, A=0):
    B, sigma = process.attributes()
    d = B.shape[1]  # dimension
    if d == 2:
        S = process.stat_cov2D()
    else:
        S = process.stationary_covariance()
    if hypo:
        if np.sum(np.abs(B - sigma @ A)) > 0:
            print(np.abs(B - sigma @ A))
            raise Warning('A incorrectly specified')
    else:
        if np.linalg.det(sigma) == 0:
            raise Warning('Said not hypoelliptic while it is')
        else:
            A = np.linalg.inv(sigma) @ B
    temp = A - sigma.T @ np.linalg.inv(S) / 2
    return 2 * np.trace(S @ temp.T @ temp)


def epr_BsigmaA_MC(process, N, hypo=False, A=0):  # integral formula for EPR elliptic process
    [B, sigma] = process.attributes()
    d = B.shape[1]  # dimension
    if d == 2:
        S = process.stat_cov2D()
    else:
        S = process.stationary_covariance()
    if np.linalg.det(S) == 0:
        print(S)
        raise TypeError('Not hypoelliptic')
    e_p = 0
    x = np.random.multivariate_normal(mean=np.zeros([d]), cov=S, size=[N]).T  # generate stationary samples
    if hypo:
        if np.sum(sigma @ B != A) > 0:
            print(np.sum(sigma @ B != A))
            raise Warning('A incorrectly specified')
    else:
        if np.linalg.det(sigma) == 0:
            raise Warning('Said hypoelliptic while it is not')
        else:
            A = np.linalg.inv(sigma) @ B
    J = A - sigma.T @ np.linalg.inv(S) / 2
    for i in range(N):
        Jx = J @ x[:, i]
        e_p += 2 * Jx.T @ Jx / N
    return e_p


'''Epr 11.7.2020'''


def epr_hypo_1107(process):
    [B, sigma] = process.attributes()
    d = B.shape[1]  # dimension
    if d == 2:
        S = process.stat_cov2D()
    else:
        S = process.stationary_covariance()
    if np.linalg.det(S) == 0:
        print(S)
        raise TypeError('Not hypoelliptic')
    D = sigma @ sigma.T / 2
    A = B + 2 * D * np.linalg.inv(S)
    Q = (B @ S - S @ B.T) / 2
    return np.trace(A.T @ D @ Q)


''' Epr 12.7.2020'''


def epr_hypo_1207(process, epsilon):
    [B, sigma] = process.attributes()
    d = B.shape[1]  # dimension
    if d == 2:
        S = process.stat_cov2D()
    else:
        S = process.stationary_covariance()
    if np.linalg.det(S) == 0:
        print(S)
        raise TypeError('Not hypoelliptic')
    A = sigma @ sigma.T
    Q_eps = np.empty([d, d])
    rQ_eps = np.empty([d, d])  # of reverse process
    Sinv = np.linalg.inv(S)
    for i in range(d):
        for j in range(d):
            Q_eps[i, j] = scipy.integrate.quad(func=integrand, a=0, b=epsilon, args=(B, sigma, i, j))[0]
            rQ_eps[i, j] = scipy.integrate.quad(func=integrand, a=0, b=epsilon, args=(-(B - A @ Sinv), sigma, i, j))[0]
    detQ = np.linalg.det(Q_eps)
    detrQ = np.linalg.det(rQ_eps)
    if detQ == 0 or detrQ == 0:
        print(np.linalg.det(Q_eps))
        print(np.linalg.det(rQ_eps))
        raise TypeError('Error: transient covariance is singular')
    else:
        rQ_inv = np.linalg.inv(rQ_eps)
    C = scipy.linalg.expm(epsilon * (B - A @ Sinv)) - scipy.linalg.expm(-epsilon * B)
    c_epr = np.trace(rQ_inv @ Q_eps) - d + np.log(detrQ / detQ) + np.trace(S @ C.T @ rQ_inv @ C)
    return c_epr / (2 * epsilon)


def epr_hypo_3107(process):
    [B, sigma] = process.attributes()
    if not np.all(np.linalg.eigvals(B) > 0):
        print(B)
        print(np.linalg.eigvals(B))
        raise TypeError("B not all positive eigenvalues")
    d = B.shape[1]  # dimension
    if d == 2:
        S = process.stat_cov2D()
    else:
        S = process.stationary_covariance()
    if np.linalg.det(S) == 0:
        print(S)
        raise TypeError('Not hypoelliptic detS=0')
    Sinv = np.linalg.inv(S)
    C = S @ B.T @ Sinv
    Ainv = np.linalg.inv(sigma @ sigma.T)
    return np.trace(S @ (B - C).T @ Ainv @ (B - C)) / 2


'''Epr via markov chain #TODO'''


def stationary_trans_density(x, nbins=10):  # return normalised transition matrix and unnormalised stationary density
    # the input is the discretisation of the process
    [d, T, N] = np.shape(x)
    b_range = bin_range(x) * 2  # double the list

    x = np.append(x[:, :-1, :], x[:, 1:, :], axis=0)

    P = np.zeros([nbins ** d, nbins ** d])
    for t in range(T - 1):
        x_tt = x[:, t, :].reshape([2 * d, N])  # x_t,x_t+1
        hist = np.histogramdd(x_tt.T, bins=nbins, range=b_range)[0]  # stationary law of x unnormalised
        h = hist.reshape([nbins ** d, nbins ** d])
        P += h.T  # transpose so that it corresponds to an unnormalised stochastic matrix

    mu = np.sum(P, axis=0)  # unnormalised stationary distribution
    mu = mu / np.sum(mu)  # stationary density
    P = P / np.sum(P, axis=0)  # transition probabilities (stochastic matrix)
    return P, mu


def stationary_density(x, nbins=10):  # return unnormalised stationary density
    [d, T, N] = np.shape(x)
    b_range = bin_range(x)

    x = x.reshape([d, N * T])  # samples from (x_t, x_t+1)

    hist = np.histogramdd(x.T, bins=nbins, range=b_range)[0]  # stationary law of x unnormalised

    # hist= hist/np.sum(hist) # stationary law of x normalised to probabilities

    mu = hist.reshape([nbins ** d])
    return mu

'''
Matrix algebra functions
'''

def initialize_random_friction(desired_eigs):
    """
    Creates a random drift or friction matrix `B` from a desired set of eigenvalues `desired_eigs`
    """
    
    ndim = len(desired_eigs) # dimensionality of the system

    D = np.diag(desired_eigs)
    
    S = np.random.rand(ndim, ndim);  # random matrix that is likely to be invertible
    
    rank_S = np.linalg.matrix_rank(S)

    while rank_S != ndim:
        S = np.random.rand(ndim, ndim)  # continue initializing random matrix until it's invertible
        rank_S = np.linalg.matrix_rank(S) # if rank of matrix is equal to dimensionality of matrix, then it's invertible

    B = np.dot(S,D).dot(np.linalg.inv(S)) # the B matrix will have the same eigenvalues of the diagonal matrix D   

    return B

def sylvester_equation_solve(B, D):
    """
    Simple way to solve Sylvester equation using linear regression. Ripped from Karl Friston's MATLAB function `spm_ness`
    Sylvester relation: B.dot(D) + D.dot(B.T) = S
    ARGUMENTS:
    ========
    `B` [np.ndarray]: (n_states, n_states) size input matrix 
    `D` [np.ndarray]: (n_states, n_states) size input matrix
    RETURNS:
    ======
    `S` [np.ndarray]: (n_states, n_states) size output matrix (the solution to the Sylvester equation)
    """

    n_states = B.shape[0]

    I = np.eye(n_states)
    X = np.kron(I,B) + np.kron(np.conj(B),I) 
    Y = (2*D).flatten()
    S = np.linalg.solve(X, Y).reshape(n_states,n_states) # solve for S is now converted into a simple linear regression  problem 

    return S

'''
Functions relevant to stimulation experiments
'''

def generate_input_kernel(pre_stim_time = 0, stim_kernel = 'boxcar', stim_amplitude = 1.0, stim_timescale = 100):
    """
    ARGUMENTS:
    ========
    `pre_stim_time`    [int]: pre-stimulus time period in timesteps
    `stim_kernel`      [str]: type of stimulus kernel to use
    `stim_amplitude` [float]: amplitude or plateau value of the stimulus kernel
    `stim_timescale` [int or list]: if int, the duration in timesteps of the boxcar impulse response function. If list, then a list of timescale parameters,
                                    whose type and interpretation depend on the chosen stimulus kernel.
    RETURNS:
    ======
    `total_stim_tseries` [np.ndarray]: length (pre_stim_time + len(convolved_point_process)) output vector that contains the total stimulus train to be presented
    """

    pre_stim_vector = np.zeros(pre_stim_time)

    if stim_kernel == 'boxcar':
        impulse_response = stim_amplitude * np.ones(stim_timescale)
    elif stim_kernel == 'double_exponential':
        impulse_response = generate_double_exponential_kernel(amplitude = stim_amplitude, rise_tau = stim_timescale[0], decay_tau = stim_timescale[1])
    elif stim_kernel == 'instant':
        impulse_response = np.array([stim_amplitude])
    elif stim_kernel == 'osc':
        impulse_response = generate_osc_kernel(amplitude = stim_amplitude, frequency = stim_timescale[0], duration = stim_timescale[1])
    
    if stim_kernel in ['boxcar', 'instant']:
        total_stim_tseries = np.concatenate( (pre_stim_vector, impulse_response) )
    elif stim_kernel in ['double_exponential', 'osc']:
        point_process = np.zeros(impulse_response.shape) # this could be adapted later to allow for stimulus trains 
        point_process[0] = 1.0                           # this could be adapted later to allow for stimulus trains 
        convolved_point_process = convolve(point_process, impulse_response,'same')[1:]
        total_stim_tseries = np.concatenate( (pre_stim_vector, convolved_point_process) )

    return total_stim_tseries

def generate_double_exponential_kernel(amplitude = 1.0, rise_tau = 5, decay_tau = 15):
    """
    Generate double exponential kernel
    ARGUMENTS:
    ========
    `amplitude` [float]: maximum value of the double exponential kernel
    `rise_tau` [int]: rise time of the first of the two exponentials
    `decay_tau` [int]: decay time of the second of the two exponentials
    ======
    `total_kernel` [np.ndarray]: length (pre_stim_time + stim_timescale) output vector that contains the total stimulus train to be presented
    """

    t_steps = int(np.exp(1)*(rise_tau + decay_tau))  # number of timesteps

    rise_part = np.exp(-(np.arange(t_steps)/rise_tau))
    decay_part = np.exp(-(np.arange(t_steps)/decay_tau))

    total_kernel = (rise_part - decay_part) / (rise_tau - decay_tau) 

    total_kernel *= amplitude / total_kernel.max()

    return total_kernel

'''Auxiliary functions'''

def integrand(t, B, sigma, i, j):  # to compute stationary covariance
    mx = scipy.linalg.expm(-B * t) @ sigma
    mx = mx @ mx.T
    return mx[i, j]


def bin_range(x):
    b_range = []
    for d in range(np.shape(x)[0]):
        b_range.append([np.min(x[d, :, :]), np.max(x[d, :, :])])
    return b_range


'''Plotting functions'''


def plot_cool_colourline(x, y, lw=0.5):
    d= np.arange(len(x))
    c= cm.cool_r((d - np.min(d)) / (np.max(d) - np.min(d)))
    ax = plt.gca()
    for i in np.arange(len(x) - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], c=c[i], linewidth=lw)
    plt.show()
    return

def plot_hot_colourline(x, y, lw=0.5):
    d= np.arange(len(x))
    c= cm.hot((d - np.min(d)) / (np.max(d) - np.min(d)))
    ax = plt.gca()
    for i in np.arange(len(x) - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], c=c[i], linewidth=lw)
    plt.show()
    return


def plot_cool_scatter(x, y, c, lw=0.5):
    c = cm.cool((c - np.min(c)) / (np.max(c) - np.min(c)))
    plt.scatter(x, y, c=c, s=lw, zorder=1)
    plt.show()
    return


# magma colourline
def plot_magma_colourline(x, y, c, lw=0.5):
    c = cm.magma((c - np.min(c)) / (np.max(c) - np.min(c)))
    ax = plt.gca()
    for i in np.arange(len(x) - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], c=c[i], linewidth=lw)
    plt.show()
    return


def clear_figures(f):
    for i in np.arange(1, f + 1):
        plt.figure(i)
        plt.clf()
