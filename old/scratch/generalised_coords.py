'''
Implementation of multivariate OU process that approximates
a non-Markovian process through the use of generalised coordinates 
and an analytic form of the noise correlations at multiple embedding orders
'''

'''
Imports
'''
import numpy as np
import pandas as pd
import scipy
from scipy.stats import multivariate_normal
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn

from utils import get_temporal_cov

'''
Model class
'''

class generalised_OU(object):

    def __init__(self, 
                num_states, # dimensionality of states at the 0-th order
                B,          # drift or friction matrix (at the 0-th order)
                C_spatial,  # covariance matrix of random fluctuations (at the 0-th order)
                num_do,     # number of generalised coordinates or embedding orders
                s,          # smoothness parameter (parameterises covariance of fluctuations among generalised orders)
                x0 = None   # initial state of the process (optional)
                ):                     
        
        self.num_states = num_states 
        self.num_do = num_do      
        self.num_states_total = num_states * num_do 
        self.B = B
        self.C_spatial = C_spatial
        self.s = s

        _, self.C_temporal = get_temporal_cov(truncation_order = num_do, smoothness = s, form = 'Gaussian')

        self.full_cov = np.kron(self.C_temporal, self.C_spatial)
            
        # create the generalised flow by creating block diagonal matrix of the flow at each order.
        generalised_drift = block_diag(*[self.B for _ in range(num_do)] )
     
        # define the flow function, given the generalised flow parameters and the generalised mean vectors
        def f(x):
            return -generalised_drift.dot(x)

        self.f = f

        if x0 is None:
            self.x0 = np.random.multivariate_normal(mean = np.zeros(self.num_states_total), cov=self.full_cov)
    
    def simulation(self, dt = 0.01, T = 100, N = 1, x0 = None):
        """
        Generate trajectories of the process
        ARGUMENTS:
        =========
        `dt` [float]: integration window in seconds
        `T` [int]: number of timesteps. Total duration in seconds of simulation will be dt * T
        `N` [int]: number of parallel trajectories to run
        `x0` [np.ndarray or None]: initial condition(s) of the process, either a single initial state or one state per parallel realization.
                                    If not provided, x0 is initialized to a zero-mean noise vector with covariance `self.cov`
        RETURNS:
        =======
        `trajectories` [np.ndarray]: history of trajectories of the multivariate OU process in generalised coordinates. 
        Also stored as data in the instance in `self.trajectories`
        """
        
        self.trajectories = np.zeros( (self.num_states_total, T, N), dtype = np.float32)  # initialize array to store parallel trajectories / solutions to the stochastic process
        
        if x0 is not None:
            if x0.ndim == 1:
                self.trajectories[:,0,:] = np.tile(x0, N).reshape(self.num_states_total, N)
            elif x0.shape[1] == N:
                self.trajectories[:,0,:] = x0
        else:
            if self.x0 is not None:
                self.trajectories[:,0,:] = np.tile(self.x0, N).reshape(self.num_states_total, N)
            else:
                self.trajectories[:,0,:] = np.random.multivariate_normal(mean = np.zeros(self.num_states_total), cov = self.full_cov, size =  N).T

        for n in range(N):
            self.forward(n = n, dt = dt, t_start = 1, t_duration = (T-1))

        return self.trajectories

    def forward(self, n = 0, dt=0.01, t_start = 0, t_duration = 10):  
        """
        Forward function of the OU process (dX_t = -B X_t + s dW_t) for a particular realization `n`
        ARGUMENTS:
        =========
        `n` [int]: index of the parallel realization to update
        `dt` [float]: integration window in seconds
        `T` [int]: number of timesteps. Total duration in seconds of simulation will be dt * T
        `t_start` [int]: index of the timestep to start rolling forward from
        `t_duration` [int]: duration in timesteps (starting at the current timestep) to run the process forward
        """
    
        w = np.random.multivariate_normal(mean = np.zeros(self.num_states_total), cov = self.full_cov, size = t_duration).T # history of random fluctuations for the process
        
        sqrtdt = np.sqrt(dt) # Euler-Maruyama scaling constant (to account for standard deviation of fluctuations per time interval, whose standard deviation will be sqrt(dt))

        time_range = range(t_start, t_start + t_duration - 1)

        for t in time_range:
            past_x = self.trajectories[:,t-1,n]
            self.trajectories[:,t,n] = past_x + dt * self.f(past_x) + sqrtdt * w[:,t]

        return
        