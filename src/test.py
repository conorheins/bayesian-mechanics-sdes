from .diffusions import LinearProcess
import jax.numpy as jnp
from jax import random

from matplotlib import pyplot as plt

key = random.PRNGKey(1)

n_var = 2
B = jnp.array([ [-0.5, 0.5], [0.5, -0.5]])

# print(jnp.linalg.eigvals(B))

# sigma = 0.01 * jnp.diag(jnp.ones(n_var))
# D = jnp.dot(sigma, sigma.T) / 2.0
# test_process = LinearProcess(n_var, B, sigma)

# n_real = 50
# T = 100000

# x0 = jnp.transpose(random.multivariate_normal(key, jnp.zeros(n_var), D, shape = (n_real,) ), (1, 0))

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
