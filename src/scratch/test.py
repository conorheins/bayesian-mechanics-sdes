# use these imports when running frmo module
# from .diffusions import LinearProcess
# from .utilities import initialize_random_friction

# use these imports when running notebook style
import os
os.chdir('..')

from diffusions import LinearProcess, NonlinearProcess
from utilities import initialize_random_friction_numpy
import jax.numpy as jnp
from jax import random, vmap, grad, jacfwd, lax
import numpy as np 
# import autograd.numpy as np
# from autograd import elementwise_grad as egrad

from matplotlib import pyplot as plt

key = random.PRNGKey(0)

n_var, dt, T, n_real = 6, 0.01, 100000, 1000 # global parameters for the process

random_evals = 0.1 * random.uniform(key, shape = (n_var, ))
_, key = random.split(key)

B_numpy = initialize_random_friction_numpy(np.array(random_evals))
B = jnp.array(B_numpy)
sigma = 0.01 * jnp.diag(jnp.ones(n_var))
D = jnp.dot(sigma, sigma.T) / 2.0

 # initialize process starting state
x0 = jnp.transpose(random.multivariate_normal(key, jnp.zeros(n_var), D, shape = (n_real,) ), (1, 0))
_, key = random.split(key)

ou_process = LinearProcess(n_var, B, sigma)

x_out = ou_process.integrate(T, n_real, dt, x0)


'''
Setting up the diffusion process
'''


key = random.PRNGKey(0)

n_var, dt, T, n_real = 1, 0.01, 100000, 1000 # global parameters for the process

# Pi = jnp.array([[1.0]])
Pi = jnp.array([1.0])

# volatility

def sigma(y):
    return  jnp.log(jnp.abs(y) + 0.5)[None, ...]

def D(y):
    return sigma(y).dot(sigma(y)) / 2.0
    
# sigma = lambda y: jnp.log(jnp.abs(y) + 0.5)

# diffusion
# D = lambda y: sigma(y) + sigma(y) / 2.0


# divergence of diffusion 


# grad_sigma_1el = grad(sigma) # elementwise gradient of sigma
# grad_sigma = vmap(grad_sigma_1el) 
# divD = lambda y: sigma(y) * jnp.apply_along_axis(grad_sigma, 0, y)

# solenoidal flow
Q = lambda y: 0.0

# divergence of solenoidal flow
divQ = lambda y: 0.0

# drift
drift = lambda y: -(D(y) + Q(y)) * Pi * y + divD(y) + divQ(y)

diff_process = NonlinearProcess(n_var, drift, sigma)

# Setting up the initial condition
x0 = jnp.transpose(random.multivariate_normal(key, jnp.zeros(n_var), jnp.linalg.inv(Pi[...,None]), shape = (n_real,) ), (1, 0))
_, key = random.split(key)

x_out = diff_process.integrate(500, n_real, dt, x0)

# w = jnp.transpose(random.multivariate_normal(key, jnp.zeros(n_var), jnp.sqrt(dt) * jnp.eye(n_var), shape = (T, n_real) ), (0, 2, 1))

# D_func_single = lambda x, w: sigma(x) * w
# D_func_single(x0, w[0])
# vmapped_D_func = vmap(D_func_single, in_axes = 1, out_axes = 1)
# self.g(x_past, w_t)

# diff_process.f(x0)
# diff_process.g(x0, w[0])

# Setting up the diffusion process
# process = diffusion.diffusion_process(dim=dim, drift=drift, volatility=sigma)  # create process

'''
Run simulation
'''

epsilon = 10 ** (-2)  # time-step
T = 5 * 10 ** 2  # number of time-steps
N = 10 ** 5  # number of trajectories

# Setting up the initial condition
x0 = np.random.normal(loc=np.zeros(dim), scale=np.sqrt(Pi ** (-1)), size=[N]).reshape(
    [dim, N])  # stationary initial condition









# %% OLD STUFF

# x_t = test_process.integrate(T, n_real, x0, dt)

# x_t = test_process.integrate(x0, dt = 0.01, T = T, N = n_real)

# %timeit x_t = test_process.integrate(x0, dt = 0.01, T = 2, N = n_real)

# import numpy as np

# x_t_single = np.zeros( (n_var, T) )

# B_numpy = np.array(B)

# w = np.random.normal(0, np.sqrt(dt), T * n_var).reshape(
#             [n_var, T ]) 
# sigma_numpy = np.array(sigma)

# x0_numpy = np.array(x0)

# for t in range(T):
#     if t == 0:
#         x_t_single[:,t] = x0_numpy[:,0] + dt * -B_numpy.dot(x0_numpy[:,0]) + sigma_numpy.dot(w[:,t])
#     else:
#         x_t_single[:,t] = x_t_single[:,t-1] + dt * -B_numpy.dot(x_t_single[:,t-1]) + sigma_numpy.dot(w[:,t])


# initialize random samples for the process
# w = jnp.transpose(random.multivariate_normal(key, jnp.zeros(n_var), jnp.eye(n_var), shape = (T, n_real) ), (0, 2, 1))

# f_vec = vmap(lambda x: -jnp.dot(B, x), in_axes = 1, out_axes = 1)
# g_vec = vmap(lambda w: jnp.dot(sigma, w), in_axes = 1, out_axes = 1)

# f_vec = lambda x: -jnp.dot(B, x)
# g_vec = lambda w: jnp.dot(sigma, w)

# def integrate_w_scan(x0, w, dt, T, N): 

#     def one_step_int(carry, w_t):

#         x_past, dt, em_scalar = carry
#         x_next = x_past + dt * f_vec(x_past) + em_scalar * g_vec(w_t)

#         return (x_next, dt, em_scalar), x_next

#     em_scalar = jnp.sqrt(dt)

#     _, x_t = lax.scan(one_step_int, (x0, dt, em_scalar), w, length = T) 

#     return x_t

# def integrate_w_for(x0, w, dt, T, N):

#     em_scalar = jnp.sqrt(dt)

#     x_t = []
#     for t in range(T):
#         if t == 0:
#             x_t.append( x0 + dt * f_vec(x0) + em_scalar * g_vec(w[t]) )
#         else:
#             x_t.append( x_t[t-1] + dt * f_vec(x_t[t-1]) + em_scalar * g_vec(w[t]) )
    
#     return jnp.stack(x_t, 0)

# scan_result = integrate_w_scan(x0, w, dt, T, n_real)
# for_result  = integrate_w_for(x0, w, dt, T, n_real)

# sigma = 0.01 * jnp.diag(jnp.ones(n_var))
# D = jnp.dot(sigma, sigma.T) / 2.0
# test_process = LinearProcess(n_var, B, sigma)

# n_real = 50
# T = 100000

# x0 = jnp.transpose(random.multivariate_normal(key, jnp.zeros(n_var), D, shape = (n_real,) ), (1, 0))

# print(x0.shape)
# x_t = test_process.integrate(x0, dt = 0.01, T = T, N = n_real)

# real_idx = random.choice(key,n_real)
# fig, ax = plt.subplots(figsize=(16,12))
# ax.plot(x_t[:,:,real_idx])
# plt.savefig(f"ou_realization_{real_idx}.png")

# N = 50

# x_test = jnp.transpose(random.multivariate_normal(key, jnp.zeros(n_var), jnp.eye(n_var), shape = (N,) ), (1, 0))

# print(x_test.shape)

# print(test_process.f(x_test).shape)

# w_test = jnp.transpose(random.multivariate_normal(key, jnp.zeros(n_var), jnp.eye(n_var), shape = (N,) ), (1, 0))
# print(test_process.g(w_test).shape)


# print(x0.shape)


# x_t = test_process.integrate(x0, 0.01, 10, N = 1)


# key = random.PRNGKey(0)

# T = 100
# d = 5
# N = 12
# w = jnp.transpose(random.multivariate_normal(key, jnp.zeros(d), jnp.eye(d), shape = (T - 1, N) ), (0, 2, 1))

# print(w.shape)

# batch1 = random.uniform(key, (10,50,30))
# batch2 = random.uniform(key, (10,30,20))

# output1 = lax.batch_matmul(batch1, batch2)
# print(f'Output shape of batch matmul: {output1.shape}')
# batch1 = random.uniform(key, (50, 30, 10))
# batch2 = random.uniform(key, (30, 20, 10))

# B = random.uniform(key, (5,5))
# flow_func_single = lambda x: -jnp.dot(B, x)

# x_all = random.uniform(key, (5,100))

# flow_func_batch = vmap(flow_func_single, in_axes = 1, out_axes = 1)
# output2 = flow_func_batch(x_all)

# B = random.uniform(key, (5,5))
# x_all = random.uniform(key, (5,1,100))
# dot_batch = vmap(jnp.dot, in_axes = (None,2), out_axes = 1)
# output2 = dot_batch(B, x_all)

# print(f'Output shape of vmapped dot: {output2.shape}')
