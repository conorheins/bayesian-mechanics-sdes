<!-- ![alt-text](https://github.com/conorheins/bayesian-mechanics-sdes/blob/repo_reorganization/just_fe_contour.gif)
 -->
 
<p align="center">
  <img src="https://github.com/conorheins/bayesian-mechanics-sdes/blob/main/just_fe_contour.gif" width="50%" height="50%"/>
</p>

# Bayesian mechanics for stationary processes

Companion repository for the paper: ["Bayesian mechanics for stationary processes"](https://arxiv.org/submit/3811135) (2021) by Lancelot Da Costa, Karl Friston, Conor Heins, and Grigorios A. Pavliotis.

# Installation 

To run the code in this repo, we recommend creating a conda environment using the provided `environment.yml` file, which 
contains all the required dependencies (see below):

```
conda env create -f environment.yml
```

Once you create the repository, you can activate it using:

```
conda activate bayesmech_statproc
```

# Requirements

The requirements to run the code in this package are all listed as dependencies in the `environment.yml` file. The core functionality depends on [Python 3.8^](https://www.python.org/downloads/release/python-380/) and relies heavily on the following packages for stochastic integration, statistics, and visualization routines:

* [jax](https://github.com/google/jax)
* [NumPy](https://github.com/numpy/numpy)
* [SciPy](http://numpy.scipy.org/)
* [Matplotlib](https://github.com/matplotlib/matplotlib)

# Running the code

Once you've created and activated your conda environment with `conda activate bayesmech_statproc`, you can run any of the code to create the paper's figures using the general format:

```
python3 src/<SCRIPT_NAME>.py
```

For example, to run the code that generates the Bayesian mechanics figures derived from the 3-D Ornstein-Uhlenbeck process (Figure 6), run the following line from a terminal window or other command line interface:

```
python3 src/linear_diffusion_3d.py 
```

or

```
python3 src/linear_diffusion_3d.py  --seed <SOME_INTEGER> --save
```

The optional command line argument `--seed` or `-s` can be used to specify a particular, fixed key for initializing `jax`'s [PRNG or pseudo-random number generation](https://jax.readthedocs.io/en/latest/jax.random.html), which ensures reproducible initial system parameterisations and stochastic sample paths. If you don't enter a `--seed` argument, a default initialization key (the same one used to generate the figures from the paper) will be used instead - thus even in the absence of a user-given `--seed` argument, the code will generate the same figures reproducibly.

If you want to save the figures as .pngs to disk (in the `./figures` folder), you can specify either `--save` or `--nosave` as a final keyword argument to your line (default is `--save`). 


