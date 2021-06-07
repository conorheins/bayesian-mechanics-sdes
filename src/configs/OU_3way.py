import jax.numpy as jnp
from jax.numpy.linalg import eigvals, det, inv

eta_dim = [0]
b_dim = [1]
mu_dim = [2]
pi_dim = b_dim + mu_dim # particular states

dimensions = {'eta': eta_dim, 
                'b': b_dim, 
                'mu': mu_dim, 
                'pi': pi_dim}

# Define precision
Pi = jnp.array([ [3., 1., 0.], 
                [1., 3., 0.5], 
                [0., 0.5, 2.5] ]) #selected precision matrix

# enforce symmetric
Pi = (Pi + Pi.T) / 2.0

# Ensure Pi is positive definite
while (eigvals(Pi) <= 0).any():
    Pi -= 2 * jnp.min(eigvals(Pi)) * jnp.eye(n_var)

# Compute the stationary covariance
S = inv(Pi)

'''
Setting up synchronisation map
'''

# Check that the synchronisation map is well defined
assert S[mu_dim, b_dim] != 0, "Synchronisation map not well defined: bold_mu(b) not invertible!"

# define the linear synchronisation map
b_mu = S[mu_dim, b_dim] * S[b_dim, b_dim] ** (-1)  # expected internal state
b_eta = S[eta_dim, b_dim] * S[b_dim, b_dim] ** (-1)  # expected external state
sync = Pi[eta_dim, eta_dim] ** (-1) * Pi[eta_dim, b_dim] * Pi[mu_dim, b_dim] ** (-1) * Pi[mu_dim, mu_dim]  # sync map

# volatility
sigma = jnp.array([ [2, 1.5, 0.5],
                    [0., 3., 2.],
                    [0., 0., 2.] ]) #selected non-degenerate noise

# see whether noise is degenerate or not
print(f'det sigma = {det(sigma)}')

#diffusion tensor
D = sigma @ sigma.T / 2  # diffusion tensor

# solenoidal flow
Q = jnp.array([ [0., 1.5, 0.5],
                [-1.5, 0., -1.],
                [-0.5, 1., 0.] ]) #selected solenoidal flow

Q = (Q - Q.T)/2 # ensure anti-symmetry

# Drift matrix
B = (D + Q) @ Pi  # drift matrix

if (eigvals(B) <= -1e-5).any():
    print(eigvals(B))
    raise TypeError("Drift should have non-negative spectrum")
    
# 1) We check it solves the Sylvester equation: BS + SB.T = 2D
assert jnp.allclose(B @ S + S @ B.T, 2 * D), "Sylvester equation not solved"

# 2) we check that there are no numerical errors due to ill conditioning
assert jnp.allclose(inv(S), Pi), "Precision and inverse covariance are different"

# We check that the stationary covariance is indeed positive definite
if (eigvals(S) <= 0).any():
    print(eigvals(S))
    raise TypeError("Stationary covariance not positive definite")

