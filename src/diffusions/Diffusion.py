import jax.numpy as jnp
from jax import lax, random, vmap

RNG_key = random.PRNGKey(0)

class DiffusionProcess(object):

    def __init__(self, flow_function, D_function, dim):

        self.f = flow_function
        self.g = D_function
        self.d = dim

    def one_step_int(self, carry, w_t):

        x_past, em_scalar = carry
        x_next = x_past + em_scalar * (self.f(x_past) + self.g(w_t))

        return (x_next, em_scalar), x_next

    def integrate(self, x0, dt, T, N = 1): 
        
        em_scalar = jnp.sqrt(dt)

        # initialize random samples for the process
        w = jnp.transpose(random.multivariate_normal(RNG_key, jnp.zeros(self.d), jnp.eye(self.d), shape = (T, N) ), (0, 2, 1))

        if N == 1:
            x0 = x0.reshape(self.d, 1)

        _, x_t = lax.scan(self.one_step_int, (x0, em_scalar), w, length = T) 

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


class NonlinearProcess(DiffusionProcess):

    # def __init__(self, name):
    #     self.name = name

    def __init__(self, f, g, name):
        super().__init__(f, g)
       
    # def integrate(self, dt, T):  
    #     pass

    # def setters(self, args):
    #     pass

    # def getters(self, args):
    #     pass