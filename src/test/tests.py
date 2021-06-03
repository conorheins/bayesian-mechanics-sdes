import os
import unittest

import numpy as np
import jax.numpy as jnp
from jax import random, vmap, lax

from src.utilities import initialize_random_friction_numpy

N_VAR, dt, T, N_REAL = 4, 1e-3, 1000, 100 # global parameters for the process

TOL = 1e-6

class DiffusionTest(unittest.TestCase):

    def test_vmapped_dot_product(self):

        """
        Validate that the vectorized dot product gives the intended parallelism
        """
        RNG_key = random.PRNGKey(0)

        random_evals = 0.1 * random.uniform(RNG_key, shape = (N_VAR, ))
        RNG_key, next_key = random.split(RNG_key)
        B = jnp.array(initialize_random_friction_numpy(np.array(random_evals)))

        f_vec = vmap(lambda x: -jnp.dot(B, x), in_axes = 1, out_axes = 1)

        x_test = jnp.transpose(random.multivariate_normal(next_key, jnp.zeros(N_VAR), jnp.eye(N_VAR), shape = (N_REAL,) ), (1, 0))

        vmapped_dot_ret = f_vec(x_test)

        non_vmapped_dot = -B.dot(x_test)
        
        self.assertTrue(jnp.isclose(vmapped_dot_ret, non_vmapped_dot).all())

    def test_scan_version(self):
        """
        Validate that the scan implementation of OU process integration is identical to that using a standard for loop
        """
        RNG_key = random.PRNGKey(1)

        random_evals = 0.1 * random.uniform(RNG_key, shape = (N_VAR, ))
        RNG_key, next_key = random.split(RNG_key)

        B = jnp.array(initialize_random_friction_numpy(np.array(random_evals)))
        
        sigma = 0.1 * random.uniform(next_key, shape = (N_VAR, )) * jnp.diag(jnp.ones(N_VAR))
        RNG_key, next_key = random.split(RNG_key)

        # initialize process starting state
        x0 = jnp.transpose(random.multivariate_normal(next_key, jnp.zeros(N_VAR), jnp.eye(N_VAR), shape = (N_REAL,) ), (1, 0))
        RNG_key, next_key = random.split(RNG_key)

        # initialize random samples for the process
        w = jnp.transpose(random.multivariate_normal(next_key, jnp.zeros(N_VAR), jnp.eye(N_VAR), shape = (T, N_REAL) ), (0, 2, 1))

        f_vec = vmap(lambda x: -jnp.dot(B, x), in_axes = 1, out_axes = 1)
        g_vec = vmap(lambda w: jnp.dot(sigma, w), in_axes = 1, out_axes = 1)

        def integrate_w_scan(x0, w, dt, T, N): 

            def one_step_int(carry, w_t):

                x_past, dt, em_scalar = carry
                x_next = x_past + dt * f_vec(x_past) + em_scalar * g_vec(w_t)

                return (x_next, dt, em_scalar), x_next

            em_scalar = jnp.sqrt(dt)

            _, x_t = lax.scan(one_step_int, (x0, dt, em_scalar), w, length = T) 

            return x_t

        def integrate_w_for(x0, w, dt, T, N):

            em_scalar = jnp.sqrt(dt)

            x_t = []
            for t in range(T):
                if t == 0:
                    x_t.append( x0 + dt * f_vec(x0) + em_scalar * g_vec(w[t]) )
                else:
                    x_t.append( x_t[t-1] + dt * f_vec(x_t[t-1]) + em_scalar * g_vec(w[t]) )
            
            return jnp.stack(x_t, 0)

        scan_result = integrate_w_scan(x0, w, dt, T, N_REAL)
        for_result  = integrate_w_for(x0, w, dt, T, N_REAL)
        
        # self.assertTrue(jnp.absolute(scan_result - for_result).max() < TOL)
        self.assertTrue(jnp.isclose(scan_result, for_result, atol = TOL).all())

if __name__ == "__main__":

    unittest.main()
    
