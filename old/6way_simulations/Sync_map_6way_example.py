'''
Simulations synchronisation map 6D (example)
'''

import numpy as np
from numpy.linalg import inv, det, pinv
from numpy.linalg import eigvals as spec
import matplotlib.pyplot as plt

pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}  # setup matplotlib to use latex for output
plt.style.use('seaborn-white')
from scipy.linalg import null_space as ker
from mpl_toolkits.mplot3d import Axes3D


'''
Functions
'''


def rank(A):  # compute matrix rank
    if A.size == 0:
        return 0
    else:
        return np.linalg.matrix_rank(A)


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


'''
Setting up the steady-state
'''

dim = 6  # dimension of state-space
de = slice(0, 2)  # dimensions of external states
ds = slice(2, 3)  # dimensions of sensory states
da = slice(3, 4)  # dimensions of active states
di = slice(4, 6)  # dimensions of internal states
db = slice(2, 4)  # dimensions of blanket states (sensory + active)
dp = slice(2, 6)  # dimensions of particular states (blanket + internal)
du = [0, 1, 3, 4, 5]  # dimensions of unresponsive states (complement of sensory)

std = 1  # standard deviations of gaussian distributions we are sampling random numbers from

# Define precision
Pi = np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim])
# enforce Markov blanket condition
Pi[de, di] = 0
Pi[di, de] = 0
# enforce symmetric
Pi = (Pi + Pi.T) / 2
# make sure positive definite
if np.any(spec(Pi) <= 0):
    Pi = Pi - 2 * np.min(spec(Pi)) * np.eye(dim)

# We compute the stationary covariance
S = np.linalg.inv(Pi)


'''
Setting up the synchronisation map
'''

# We check that the synchronisation map is well defined according to the conditions outlined in the paper
if rank(np.append(ker(Pi[di, db]), ker(Pi[de, db]), axis=1)) > rank(ker(Pi[de, db])):
    raise TypeError("Synchronisation map not well defined")

# define the linear synchronisation map
mu = S[di, db] @ inv(S[db, db])  # expected internal state
eta = S[de, db] @ inv(S[db, db])  # expected external state
sync = inv(Pi[de, de]) @ Pi[de, db] @ pinv(Pi[di, db]) @ Pi[di, di]  # sync map


'''
Generate samples of Gaussian distribution
'''
N = 10 ** 7   # number of trajectories #usually 3*10**2
x = np.random.multivariate_normal(mean=np.zeros(dim), cov=S, size=N).T


'''
Tiling blanket state-space
'''

#blanket state space dimensions
db1= 2
db2= 3

#Tiling blanket state-space
var_b1= S[db1,db1]
var_b2 =S[db2,db2]

#select indices
epsilon = 10**(-2)

#create a grid of mesh size epsilon**2
tile_b1= np.arange(-var_b1, var_b1, epsilon)
tile_b2= np.arange(-var_b2, var_b2, epsilon)

#selecting only those samples that fit in the grid
indices = (x[db1,:]>= -var_b1) * (x[db1,:]< var_b1) * (x[db2,:]>= -var_b2) * (x[db2,:]< var_b2)
print(np.sum(indices)/(num(tile_b1)*num(tile_b2)))
x = x[:,indices]



'''
Finding empirically boldeta, sync boldmu
'''

def prediction_error(x, tile_b1, tile_b2):
    i=0
    bold_eta_b = np.empty([2, num(tile_b1), num(tile_b2)])
    sync_boldmu_b = np.empty([2, num(tile_b1), num(tile_b2)])
    pred_error_b = np.empty([2, num(tile_b1), num(tile_b2)])
    for b1 in tile_b1:
        j=0
        for b2 in tile_b2:
            #select all indices corresponding to samples whose blanket states are in desired mesh
            indices = (x[db1,:]>= b1) * (x[db1,:]< b1+epsilon) * (x[db2,:]>= b2) * (x[db2,:]< b2+epsilon)
            #compute empirical bold_eta of b
            boldeta = np.mean(x[de, indices],axis=1)
            #compute empirical bold_mu of b
            boldmu = np.mean(x[di, indices],axis =1)
            sigma_boldmu = sync @ boldmu
            #store samples
            bold_eta_b[:,i,j]= boldeta
            sync_boldmu_b[:, i,j] = sigma_boldmu
            pred_error_b[:,i,j] = boldeta - sigma_boldmu
            #increase counter
            j=j+1
        i=i+1
    return pred_error_b, bold_eta_b, sync_boldmu_b

'''
Creating figure error surface plot
'''

X,Y= np.meshgrid(tile_b2, tile_b1)

F_b1_b2 = prediction_error(x,tile_b1,tile_b2)

for i in range(0, 2):
    boldeta = F_b1_b2[1][i,:,:]
    sync_boldmu = F_b1_b2[2][i,:,:]
    Z = F_b1_b2[1][i, :, :]
    fig = plt.figure(i)
    ax = plt.axes(projection='3d')
    #create legend
    ax.scatter(0, 0, 0, marker='o', c='cornflowerblue',label='Actual: $\mathbf{\eta}$' + f'$(b)_{i + 1}$')
    ax.scatter(0, 0, 0, marker='o',c='darkorange', label='Prediction: $\sigma(\mathbf{\mu}$' + f'$(b))_{i + 1}$')
    ax.legend()
    #plot
    ax.contour3D(X, Y, boldeta, 30, cmap='Blues')
    ax.contour3D(X, Y, sync_boldmu, 30,cmap ='Oranges')
    ax.set_xlabel('$b_1$')
    ax.set_ylabel('$b_2$')
    ax.set_zlabel(f'$\eta_{i + 1}$')
    plt.savefig(f"syncmap_6way_dim{i + 1}_eg.png")
