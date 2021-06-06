import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn

import jax.numpy as jnp

def plot_hot_colourline(x, y, lw=0.5):
    d= jnp.arange(len(x))
    c= cm.hot((d - d.min()) / jnp.ptp(d) )
    ax = plt.gca()
    for i in jnp.arange(len(x) - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], c=c[i], linewidth=lw)
    return