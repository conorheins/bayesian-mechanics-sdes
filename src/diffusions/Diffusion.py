import jax.numpy as jnp
from jax import jit, lax, random, vmap
from functools import partial

default_key = random.PRNGKey(0)

class DiffusionProcess(object):

    def __init__(self, flow_function, D_function, dim):

        self.f = flow_function
        self.g = D_function
        self.d = dim
    
    def integrate(self, T, N, dt, *scan_args, rng_key = None): 
        """
        Generic integration function for the diffusion process
        Arguments:
        ==========
        `T` [int] : number of timesteps to realize
        `N` [int] : number of parallel processes / sample paths to realize
        `scan_args` [tuple]: positional arguments that are passed to stochastic 
            integration (e.g. for an OU process, this is just the state initial state `x0` and the integration window `dt`)
        
        Returns:
        ========
        `x_t` [JAX DeviceArray]: multidimensional array storing the history of the samples paths over time, for 
        different parallel realizations
        """
        
        if rng_key is None:
            RNG_key = default_key
        else:
            RNG_key = rng_key

        # initialize random samples for the diffusion part of the process (the derivative of Brownian motion, dB/dt)
        w = jnp.transpose(random.multivariate_normal(RNG_key, jnp.zeros(self.d), dt * jnp.eye(self.d), shape = (T, N) ), (0, 2, 1))

        if N == 1:
            x0 = x0.reshape(self.d, 1)

        integration_func = jit(partial(self.one_step_int, dt)) # re-define one-step integration function for the input `dt` fixed

        def scan_fn(carry, w_t):

            next_carry = integration_func(*carry, w_t)

            return (next_carry, ), next_carry

        _, x_t = lax.scan(scan_fn, scan_args, w, length = T) 

        return x_t

class LinearProcess(DiffusionProcess):

    def __init__(self, dim, friction, volatility):
        """
        Arguments:
        `dim` [int]: dimensionality of the process (e.g. a vector valued process will have `dim > 1`)
        `friction` [np.float or np.ndarray]: (negative) drift parameters (scalar or matrix)
        `volatility` [np.float or np.ndarray]: 
        """

        self.d = dim
        self.B = friction  
        self.sigma = volatility

        self._set_flow()
        self._set_D_func()

        super().__init__(self.f,self.g, self.d)
    
    def _set_flow(self):
        """
        Sets the deterministic part of the flow (drift), given the drift matrix of the process
        """

        flow_single = lambda x: -jnp.dot(self.B, x)
        self.f = jit(vmap(flow_single, in_axes = 1, out_axes = 1)) # this assumes that the input array is of size (dim, num_parallel_samples)

    def _set_D_func(self):
        """
        Sets the stochastic / non-deterministic part of the flow (diffusion), given the volatility of the process
        """

        D_func_single = lambda w: jnp.dot(self.sigma, w)
        self.g = jit(vmap(D_func_single, in_axes = 1, out_axes = 1)) # this assumes that the input array is of size (dim, num_parallel_samples)
    
    def one_step_int(self, dt, x_past, w_t):

        return x_past + dt * self.f(x_past) + self.g(w_t)

class NonlinearProcess(DiffusionProcess):

    def __init__(self, dim, friction, volatility):
        """
        Arguments:
        `dim` [int]: dimensionality of the process (e.g. a vector valued process will have `dim > 1`)
        `friction` [function]: flow function of the process
        `volatility` [function]: diffusion/volatility function of the process 
        """

        self.d = dim
        self.B = friction  
        self.S = volatility

        self._set_flow()
        self._set_D_func()

        super().__init__(self.f,self.g, self.d)
    
    def _set_flow(self):
        """
        Sets the deterministic part of the flow (drift), given the drift function of the process
        """

        self.f = jit(vmap(lambda x: self.B(x), in_axes = 1, out_axes = 1)) # this assumes that the input array is of size (dim, num_parallel_samples)

    def _set_D_func(self):
        """
        Sets the stochastic / non-deterministic part of the flow (diffusion), given the volatility of the process
        """

        D_func_single = lambda x, w: jnp.dot(self.S(x), w)
        self.g = jit(vmap(D_func_single, in_axes = 1, out_axes = 1)) # this assumes that the input array is of size (dim, num_parallel_samples)
    
    def one_step_int(self, dt, x_past, w_t):
        """
        Integration for nonlinear diffusion, where the noise is state dependent
        """

        return x_past + dt * self.f(x_past) + self.g(x_past, w_t)