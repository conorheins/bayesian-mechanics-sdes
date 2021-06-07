import os
from diffusions import LinearProcess
import jax.numpy as jnp
from jax.numpy.linalg import inv
from jax import random
import numpy as np 
import matplotlib.pyplot as plt

from configs.config_6d import initialize_6d_OU

key = random.PRNGKey(1) # fix random seed for reproducibility

pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}  # setup matplotlib to use latex for output
plt.style.use('seaborn-white')

figures_folder = 'figures'
if not os.path.isdir(figures_folder):
    os.mkdir(figures_folder)
    
flow_parameters, stationary_stats, sync_mappings, dimensions = initialize_6d_OU()
