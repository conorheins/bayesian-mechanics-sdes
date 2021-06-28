'''
Functions supporting the rest of the package
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def num(s):  # counts number of elements
    if isinstance(s, slice):
        if s.step is None:
            return s.stop - s.start
        else:
            return (s.stop - s.start) / s.step
    elif isinstance(s, float):
        return 1
    elif isinstance(s, np.ndarray):
        return int(np.prod(s.shape))
    else:
        print(type(s))
        raise TypeError('Type not supported by num')

def rank(A):  # compute matrix rank
    if A.size == 0:
        return 0
    else:
        return np.linalg.matrix_rank(A)


'''
Evolution of entropy over time
'''

def showentropy(x, nbins=10, color='blue',
                    label='helloword'):  # compute and plot statistics of the simulation
    [d, T, N] = np.shape(x)
    '''
    Entropy
    '''
    plt.figure()
    plt.suptitle("Entropy diffusion process")

    entropy_x = entropy(x, nbins)

    plt.xlabel('time')
    plt.ylabel('H[p(x)]')
    plt.plot(range(T), entropy_x, c=color, linewidth=0.5, label=label)
    plt.legend()

def entropy(x, nbins=10):  # estimate entropy of sample
    [d, T, N] = np.shape(x)
    b_range = bin_range(x)
    entropy = np.zeros(T)
    for t in range(T):
        h = np.histogramdd(x[:, t, :].T, bins=nbins, range=b_range)[0]
        entropy[t] = - np.sum(h / N * np.log(h + (h == 0)))
    return entropy + np.log(N)

def bin_range(x):
    b_range = []
    for d in range(np.shape(x)[0]):
        b_range.append([np.min(x[d, :, :]), np.max(x[d, :, :])])
    return b_range

'''
Plot paths
'''


def showpaths(x, color='blue', label='helloword'):  # compute and plot statistics of the simulation
    [d, T, N] = np.shape(x)
    '''
    Sample trajectory
    '''
    plt.figure()
    plt.suptitle("Process trajectory")
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.plot(x[0, :, 0], x[1, :, 0], c=color, linewidth=0.1, label=label)
    plt.legend()


def plot_hot_colourline(x, y, lw=0.5):
    d= np.arange(len(x))
    c= cm.hot((d - np.min(d)) / (np.max(d) - np.min(d)))
    ax = plt.gca()
    for i in np.arange(len(x) - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], c=c[i], linewidth=lw)
    plt.show()
