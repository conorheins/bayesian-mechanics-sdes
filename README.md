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

The requirements to run the code in this package are listed in the `environment.yml` file, but the core functionality depends on [Python 3.8^](https://www.python.org/downloads/release/python-380/) and the following packages are central the numerical and visualization routines:

* [jax](https://github.com/google/jax)
* [NumPy](https://github.com/numpy/numpy)
* [SciPy](http://numpy.scipy.org/)
* [Matplotlib](https://github.com/matplotlib/matplotlib)

