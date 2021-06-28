# %%
from diffusions import NonlinearProcess
import jax.numpy as jnp
from jax import random, vmap, grad, jacfwd, lax, jit
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

key = random.PRNGKey(2) # fix random seed for reproducibility

n_var, dt, T, n_real = 1, 10 ** (-2), 5 * 10 ** 2, 5  # global parameters for the process

# %% my version
Pi = jnp.array([[1.0]])

# state dependent volatility
def sigma(y):
    return  jnp.log(jnp.abs(y) + 0.5)[None,...] # have to expand dims to make it technically a 1 x 1 volatility matrix

# state-dependent diffusion tensor
def D(y):
    return (sigma(y) @ sigma(y).T) / 2.0

jac_sigma = jacfwd(sigma) # elementwise gradient of sigma
divD = lambda y: sigma(y) @ jnp.trace(jac_sigma(y))

# solenoidal flow
Q = lambda y: jnp.array([0.0])

# divergence of solenoidal flow
divQ = lambda y: jnp.array([0.0])

# drift expressed in terms of Helmholtz decomposition
drift = lambda y: -(D(y) + Q(y)) @ Pi @ y + divD(y) + divQ(y)

# %% Lance version

import autograd.numpy as np
from autograd import elementwise_grad as egrad


dim = 1
Pi_np = np.array([1.0])


# volatility
def sigma_np(y):
    return np.log(np.abs(y) + 0.5)


# diffusion
def D_np(y):
    return sigma_np(y) * sigma_np(y) / 2


# divergence of diffusion
grad_sigma_np = egrad(sigma_np)  # elementwise gradient of sigma


def divD_np(y):
    return sigma_np(y) * grad_sigma_np(y)


# solenoidal flow
def Q_np(y):
    return 0


# divergence of solenoidal flow
def divQ_np(y):
    return 0


# drift
def drift_np(y):
    return -(D_np(y) + Q_np(y)) * Pi_np * y + divD_np(y) + divQ_np(y)

# %%

x0_jax = jnp.transpose(random.multivariate_normal(key, jnp.zeros(n_var), jnp.linalg.inv(Pi), shape = (n_real,) ), (1, 0))
x0_np = np.array(x0_jax)

f = jit(vmap(lambda x: drift(x), in_axes = 1, out_axes = 1))

print(jnp.allclose(drift_np(x0_np), f(x0_jax)))

_, next_key = random.split(key)

w_jax = np.transpose(random.multivariate_normal(next_key, jnp.zeros(n_var), jnp.sqrt(dt) * jnp.eye(n_var), shape = (T, n_real) ), (0, 2, 1))

w_np = np.array(w_jax)

g = jit(vmap(lambda x, w: jnp.dot(sigma(x), w), in_axes = 1, out_axes = 1)) # this assumes that the input array is of size (dim, num_parallel_samples)
    
print(jnp.allclose(sigma_np(x0_np) * w_np[0], g(x0_jax, w_jax[0])))

# %%

def one_step_int(dt, x_past, w_t):
    """
    Integration for nonlinear diffusion, where the noise is state dependent
    """

    return x_past + dt * f(x_past) + g(x_past, w_t)


# %%

Pi = jnp.array([ [3., 1., 0.], 
                        [1., 3., 0.5], 
                        [0., 0.5, 2.5] ]) #selected precision matrix
n_var, n_real = 3, 5

x0_jax = jnp.transpose(random.multivariate_normal(key, jnp.zeros(n_var), jnp.linalg.inv(Pi), shape = (n_real,) ), (1, 0))

def Q(y):
    temp = jnp.tile(y, (y.shape[0], 1))
    return temp - temp.T

j_Q = jacfwd(Q)


y = x0_jax[:,0]
# divergence of solenoidal flow
def divQ(y):
    return jnp.trace(j_Q(y), axis1=0, axis2=2)
