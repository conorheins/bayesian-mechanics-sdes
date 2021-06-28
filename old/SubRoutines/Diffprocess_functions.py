'''
Functions related to diffusion processes
'''

'''
Imports
'''
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import scipy
import matplotlib.pyplot as plt
from SubRoutines import Auxiliary as aux
import matplotlib.cm as cm
import seaborn

'''
Process
'''


class diffusion_process(object):
    # dX_t = b(X_t)dt + s(X_t)dW_t
    def __init__(self, dim, drift, volatility):
        super(diffusion_process, self).__init__()  # constructs the object instance
        self.d = dim
        self.b = drift
        self.s = volatility  # volatility

    def simulation(self, x0, dt=0.01, T=100, N=1):  # run diffusion process for multiple trajectories
        w = np.random.normal(0, np.sqrt(dt), (T - 1) * self.d * N).reshape(
            [self.d, T - 1, N])  # random fluctuations
        # sqrt epsilon because standard deviation, but epsilon as covariance
        x = np.empty([self.d, T, N])  # store values of the process
        if x0.shape == x[:, 0, 0].shape:
            x[:, 0, :] = np.tile(x0, N).reshape([self.d, N])  # initial condition
        elif x0.shape == x[:, 0, :].shape:
            x[:, 0, :] = x0
        else:
            raise TypeError("Initial condition has wrong dimensions")
        if self.d > 1:
            for n in range(N - 1):
                for t in range(1, T):
                    x[:, t, n + 1] = x[:, t - 1, n] + dt * self.b(x[:, t - 1, n]) \
                                     + np.tensordot(self.s(x[:, t - 1, n]), w[:, t - 1, n], axes=1)
                    if np.count_nonzero(np.isnan(x)):
                        print("Warning nan: process went too far")
                        return x[:,:(t-1),:]

        elif self.d == 1:  # if dimension =1 we vectorise
            for t in range(1, T):
                x[:, t, :] = x[:, t - 1, :] + dt * self.b(x[:, t - 1, :]) + self.s(x[:, t - 1, :]) * w[:, t - 1, :]
                if np.count_nonzero(np.isnan(x)):
                    print("Warning nan: process went too far")
                    return x[:, :(t - 1), :]
        return x

    def attributes(self):
        return [self.b, self.s]
