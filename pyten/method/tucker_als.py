import numpy as np
import pyten.tenclass


def tucker_als(y, r=20, omega=None, tol=1e-4, max_iter=100, init='random', printitn=100):
    """
    TUCKER_ALS Higher-order orthogonal iteration.
    ----------
     :param
        'y' - Tensor with Missing data
        'r' - Rank of the tensor
        'omega' - Missing data Index Tensor
        'tol' - Tolerance on difference in fit
        'maxiters' - Maximum number of iterations
        'init' - Initial guess ['random'|'nvecs'|'eigs']
        'printitn' - Print fit every n iterations; 0 for no printing
    ----------
      :return
        'T1' - Decompose result.(kensor)
        'X' - Recovered Tensor.
    ----------
    """

    X = y.data.copy()
    X = pyten.tenclass.Tensor(X)
    # Setting Parameters
    # Construct omega if no input
    if omega is None:
        omega = X.data * 0 + 1

    # Extract number of dimensions and norm of X.
    N = X.ndims
    dimorder = range(N)  # 'dimorder' - Order to loop through dimensions {0:(ndims(A)-1)}

    # Define convergence tolerance & maximum iteration
    fitchangetol = tol
    maxiters = max_iter

    # Define size for factorization matrices
    if type(r) == int:
        r = r * np.ones(N, dtype=int)

    # Error checking
    # Error checking on maxiters
    if maxiters < 0:
        raise ValueError('OPTS.maxiters must be positive')

    # Set up and error checking on initial guess for U.
    if type(init) == list:
        Uinit = init[:]
        if len(Uinit) != N:
            raise IndexError('OPTS.init does not have %d lists', N)
        for n in dimorder[1:]:
            if Uinit[n].shape != (X.shape[n], r[n]):
                raise IndexError('OPTS.init{%d} is the wrong size', n)
    else:
        # Observe that we don't need to calculate an initial guess for the
        # first index in dimorder because that will be solved for in the first
        # inner iteration
        if init == 'random':
            Uinit = range(N)
            Uinit[0] = []
            for n in dimorder[1:]:
                Uinit[n] = np.random.random([X.shape[n], r[n]])
        elif init == 'nvecs' or init == 'eigs':
            # Compute an orthonormal basis for the dominant
            # Rn-dimensional left singular subspace of
            # X_(n) (0 <= n <= N-1).
            Uinit = range(N)
            Uinit[0] = []
            for n in dimorder[1:]:
                print('  Computing %d leading e-vectors for factor %d.\n', r, n)
                Uinit[n] = X.nvecs(n, r)  # first r leading eigenvecters
        else:
            raise TypeError('The selected initialization method is not supported')

    # Set up for iterations - initializing U and the fit.
    U = Uinit[:]
    fit = 0

    if printitn > 0:
        print('\nTucker Alternating Least-Squares:\n')

    # Set up loop2 for recovery. If loop2=1, then we need to recover the Tensor.
    Loop2 = 0
    if omega.any():
        Loop2 = 1

    # Main Loop: Iterate until convergence
    """Still need some change. Right now, we use two loops to recover a Tensor, one loop is enough."""
    normX = X.norm()
    for iter in range(1, maxiters + 1):
        if Loop2 == 1:
            Xpast = X.data.copy()
            Xpast = pyten.tenclass.Tensor(Xpast)
            fitold = fit
        else:
            fitold = fit

        # Iterate over all N modes of the Tensor
        for n in range(N):
            tempU = U[:]
            tempU.pop(n)
            tempIndex = range(N)
            tempIndex.pop(n)
            Utilde = X
            for k in range(len(tempIndex)):
                Utilde = Utilde.ttm(tempU[k].transpose(), tempIndex[k] + 1)

            # Maximize norm(Utilde x_n W') wrt W and
            # keeping orthonormality of W
            U[n] = Utilde.nvecs(n, r[n])

        # Assemble the current approximation
        core = Utilde.ttm(U[n].transpose(), n + 1)

        # Construct fitted Tensor
        T1 = pyten.tenclass.Ttensor(core, U)
        T = T1.totensor()

        # Compute fitting error
        if Loop2 == 1:
            X.data = T.data * (1 - omega) + X.data * omega
            diff = Xpast.data - X.data
            fitchange = np.linalg.norm(diff) / normX

            normXtemp = X.norm()
            normresidual = np.sqrt(abs(normXtemp ** 2 - core.norm() ** 2))
            fit = 1 - (normresidual / normXtemp)  # fraction explained by model
            fitchange = max(abs(fitold - fit), fitchange)

        else:
            normresidual = np.sqrt(abs(normX ** 2 - core.norm() ** 2))
            fit = 1 - (normresidual / normX)  # fraction explained by model
            fitchange = abs(fitold - fit)

        # Print inner loop fitting change
        if printitn != 0 and iter % printitn == 0:
            print ' Tucker_ALS: iterations={0}, fit = {1}, fit-delta = {2}\n'.format(iter, fit, fitchange)
            # print ' Iter ',iter,': fit = ',fit,'fitdelta = ',fitchange,'\n'
        # Check for convergence
        if (iter > 1) and (fitchange < fitchangetol):
            break

    return T1, X
