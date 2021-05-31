import jax.numpy as jnp
from jax import lax, random, vmap

RNG_key = random.PRNGKey(0)

class DiffusionProcess(object):

    def __init__(self, flow_function, D_function, dim):

        self.f = flow_function
        self.g = D_function
        self.d = dim

    def integrate(self, x0, dt, T, N = 1): 
        
        em_scalar = jnp.sqrt(dt)

        # initialize random samples for the process
        w = jnp.transpose(random.multivariate_normal(RNG_key, jnp.zeros(self.d), em_scalar * jnp.eye(self.d), shape = (T, N) ), (0, 2, 1))

        if N == 1:
            x0 = x0.reshape(self.d, 1)

        _, x_t = lax.scan(self.one_step_int, (x0, dt), w, length = T) 

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
        self.f = vmap(flow_single, in_axes = 1, out_axes = 1) # this assumes that the input array is of size (dim, num_parallel_samples)
    
    def _set_D_func(self):
        """
        Sets the stochastic / non-deterministic part of the flow (diffusion), given the volatility of the process
        """

        D_func_single = lambda w: jnp.dot(self.sigma, w)
        self.g = vmap(D_func_single, in_axes = 1, out_axes = 1) # this assumes that the input array is of size (dim, num_parallel_samples)
    
    def one_step_int(self, carry, w_t):

        x_past, dt = carry
        x_next = x_past + dt * self.f(x_past) +  self.g(w_t)

        return (x_next, dt), x_next


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

        self.f = vmap(lambda x: self.B(x), in_axes = 1, out_axes = 1) # this assumes that the input array is of size (dim, num_parallel_samples)
    
    def _set_D_func(self):
        """
        Sets the stochastic / non-deterministic part of the flow (diffusion), given the volatility of the process
        """

        self.g = vmap(lambda x: self.S(x), in_axes = 1, out_axes = 1) # this assumes that the input array is of size (dim, num_parallel_samples)
    
    def one_step_int(self, carry, w_t):
        """
        Integration for nonlinear diffusion, where the noise is state dependent
        """

        x_past, dt = carry
        x_next = x_past + dt * self.f(x_past) + self.g(x_past, w_t)

        return (x_next, dt), x_next