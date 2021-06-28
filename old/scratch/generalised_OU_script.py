# If running in an IPYthon kernel (e.g. VSCode), the lines of code below are needed to add OUprocess functions to current Python path
# from pathlib import Path
# import os
# import sys
# sys.path.append(Path(os.getcwd()).parent)

from generalised_coords import generalised_OU
import numpy as np
import matplotlib.pyplot as plt

# %% Simulate a single variable system using three generalised coordinates (position, velocity, acceleration)
num_states = 1  # dimensionality of states / No. of variates at the 0th order
B = 1.0         # drift  
C_spatial = 0.1 # "spatial" variance - i.e. variance of fluctuations at the 0-th order
num_do = 3      # number of embedding orders
s = 1.5         # smoothnness

gen_ou_proc = generalised_OU(num_states, B, C_spatial, num_do, s)

traj = gen_ou_proc.simulation(dt = 0.01, T = 1000, N = 1)

plt.figure(figsize=(12,12))
plt.plot(traj[0].squeeze().T, label = 'Position')
plt.plot(traj[1].squeeze().T, label = 'Velocity')
plt.plot(traj[2].squeeze().T, label = 'Acceleration')
plt.suptitle('1 state, 3 embedding orders')

plt.legend(loc='upper right')
plt.savefig("1_state_3_gencoord.png")

# %% Now simulate a 3-dimensional process that also has three generalised coordinates (total dimension = 9)

num_states = 3

B = np.array([2, 2, 2, 1, 2, -1, -1, 0, 2]).reshape([num_states, num_states])  # drift matrix
C_spatial = 0.1 * np.eye(num_states) # "spatial" (co)covariance - i.e. covariance of fluctuations at the 0-th order

num_do = 3
s = 1.5

gen_ou_proc = generalised_OU(num_states, B, C_spatial, num_do, s)

traj = gen_ou_proc.simulation(dt = 0.01, T = 1000, N = 1)

plt.figure(figsize=(12,12))
plt.plot(traj.squeeze().T)
plt.suptitle('3 states, 3 embedding orders')

plt.savefig("3_states_3_gencoord.png")
