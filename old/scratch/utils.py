import numpy as np
from scipy.special import factorial, gamma
import OUprocess_functions as OU

def get_temporal_cov(truncation_order = 1, smoothness = 1.0, form = 'Gaussian'):
    '''
    Direct re-implementation of the SPM function 'spm_DEM_R.m'
    Returns the precision- and covariance-matrices of the temporal derivatives of a Gaussian process
    FORMAT R,V = get_temporal_cov(truncation_order,smoothness,form)
    n    - truncation order [default: 1]
    s    - temporal smoothness - s.d. of kernel {bins} [default: 1.0]
    form - 'Gaussian', '1/f' [default: 'Gaussian']
    R    - shape (n,n)     E*V*E: precision of n derivatives
    V    - shape (n,n)     V:    covariance of n derivatives
    '''
    if form == 'Gaussian':
        k = np.arange(0,truncation_order)
        r = np.zeros(1+2*k[-1])
        x = np.sqrt(2.0) * smoothness
        r[2*k] = np.cumprod(1 - (2*k))/(x**(2*k))
    elif form == '1/f':
        k = np.arange(0,truncation_order)
        x = 8.0*smoothness**2
        r[2*k] = (-1)**k * gamma(2*k + 1)/(x**(2*k))

    V = np.zeros((truncation_order, truncation_order))
    for i in range(truncation_order):
        V[i,:] = r[np.arange(0,truncation_order) + i]
        r = -r

    R = np.linalg.inv(V) # covariance matrix

    return R, V

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

def fit_Q(B_init, Q_init, Pi_new, D, n_iter = 10):
    """
    function for iteratively solving for the solenoidal flow matrix Q.
    Not tested for non-diagonal diffusion matrix
    """

    n_states = B_init.shape[0]

    B_optim = B_init

    Q_new = Q_init

    for iter_i in range(n_iter):
        
         # Q_new = np.linalg.solve(Pi_new.T, B_optim.T) # optimizing like this does not ensure skew-symmetry of Q        
        
        row_order = list(np.random.permutation(n_states)) # randomize iteration over the rows
        
        for row_idx in row_order:

            '''
            iteratively optimize Q with fixed B, in order to satisfy constraints endowed by Hessian Pi_new
            '''

            # solve for new row of Q
            proj_row = np.linalg.solve(Pi_new.T, B_optim[row_idx,:].T) - D[:,row_idx] # not sure whether this maths works out in case of non-diagonal D

            # enforce antisymmetry constraints on Q
            Q_new[row_idx,:] = proj_row
            Q_new[:,row_idx] = -proj_row
            
            Q_new -= np.diag(np.diag(Q_new)) # enforce 0 diagonal in Q (practically ensured anyway)

        '''
        enforce constraints after optimizing all the rows (seems to work better in practice than optimizing B after every row)
        '''

        B_optim = (D+Q_new).dot(Pi_new)

    return B_optim, Q_new

def itercheck_QB(Q, D, Pi, zero_pairs = None, n_iter = 5, verbose = True):
    """
    Find a good setting of Q, B, Pi in order to satisfy the 0 index constraints laid out in zero_pairs
    """

    num_states = Q.shape[0]

    # enforce 0 constraints (e.g. Markov blanket condition)
    if zero_pairs is not None:
        for pair in zero_pairs:
            Pi[np.ix_(pair[0], pair[1])] = 0.0

    # for a fixed Q, D, and Pi, solve for the drift matrix B  
    B = (Q+D).dot(Pi)

    # Drift matrix
    eig_values = np.linalg.eigvals(B)
    n_init = 0

    bad_eigenvalue_condition = (eig_values <= 0).any() or np.iscomplex(eig_values).any()
    try:
        tmp = np.linalg.cholesky(Pi)
        not_pd_condition = False
    except:
        not_pd_condition = True

    if not_pd_condition:
        sylvester_ineq = True
    else:
        S = np.linalg.inv(Pi) # invert Pi to obtain S
        S_validate = OU.sylvester_equation_solve(B, D) # validation solution by solving for the stationary covariance via the Sylvester equation

        sylvester_ineq = not np.isclose(S, S_validate).all()

    while bad_eigenvalue_condition or not_pd_condition or sylvester_ineq:

        n_init +=1 

        if verbose and (n_init == 1 or (n_init % 5) == 0):
            print(f'Conditions not met! Reinitializing...\n')

        desired_eigs = np.random.uniform(low = 0.01, high = 1.0, size = num_states)
        B = OU.initialize_random_friction(desired_eigs)

        S = OU.sylvester_equation_solve(B, D) # solve for the stationary covariance by solving the Sylvester equation
        Pi = np.linalg.inv(S)

        Q = OU.sylvester_equation_solve(B, (B.dot(D) - D.dot(B.T)) / 2.0 ) # solve for the solenoidal flow by solving this Sylvester equation

        # re-enforce 0-constraints
        if zero_pairs is not None:
            for pair in zero_pairs:
                Pi[np.ix_(pair[0], pair[1])] = 0.0

        # new iterative method for solving for B and Q simultaneously
        B, Q = fit_Q(B, Q, Pi, D, n_iter = n_iter)

        eig_values = np.linalg.eigvals(B)

        # Check all the conditions (eigenvalues, invertibility/positive-definiteness, and sylvester solution equality)

        bad_eigenvalue_condition = (eig_values <= 0).any() or np.iscomplex(eig_values).any()

        try:
            tmp = np.linalg.cholesky(Pi)
            not_pd_condition = False
        except:
            not_pd_condition = True
        
        if not_pd_condition:
            sylvester_ineq = True
        else:
            S = np.linalg.inv(Pi) # invert Pi to obtain S
            S_validate = OU.sylvester_equation_solve(B, D) # validation solution by solving for the stationary covariance via the Sylvester equation

            sylvester_ineq = not np.isclose(S, S_validate).all()

    S = np.linalg.inv(Pi)

    if verbose and n_init > 0:
        print(f'Took {n_init} initializations to solve')
    
    return B, Q, Pi, S

def rank(A):  # compute matrix rank
    if A.size == 0:
        return 0
    else:
        return np.linalg.matrix_rank(A)

    
    