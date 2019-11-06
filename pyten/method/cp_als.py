import numpy
from pyten.tools import khatrirao
import pyten.tenclass


def cp_als(y, r=20, omega=None, tol=1e-4, maxiter=100, init='random', printitn=100):
    """ CP_ALS Compute a CP decomposition of a Tensor (and recover it).
    ---------
     :param  'y' - Tensor with Missing data
     :param  'r' - Rank of the tensor
     :param 'omega' - Missing data Index Tensor
     :param 'tol' - Tolerance on difference in fit
     :param 'maxiters' - Maximum number of iterations
     :param 'init' - Initial guess ['random'|'nvecs'|'eigs']
     :param 'printitn' - Print fit every n iterations; 0 for no printing
    ---------
     :return
        'P' - Decompose result.(kensor)
        'X' - Recovered Tensor.
    ---------
    """

    X = y.data.copy()
    X = pyten.tenclass.Tensor(X)
    # Setting Parameters
    # Construct omega if no input
    if omega is None:
        omega = X.data * 0 + 1

    # Extract number of dimensions and norm of X.
    N = X.ndims
    normX = X.norm()
    dimorder = range(N)  # 'dimorder' - Order to loop through dimensions {0:(ndims(A)-1)}

    # Define convergence tolerance & maximum iteration
    fitchangetol = tol
    maxiters = maxiter

    # Recover or just decomposition
    recover = 0
    if 0 in omega:
        recover = 1

    # Set up and error checking on initial guess for U.
    if type(init) == list:
        Uinit = init[:]
        if len(Uinit) != N:
            raise IndexError('OPTS.init does not have %d lists', N)
        for n in dimorder[1:]:
            if Uinit[n].shape != (X.shape[n], r):
                raise IndexError('OPTS.init{%d} is the wrong size', n)
    else:
        # Observe that we don't need to calculate an initial guess for the
        # first index in dimorder because that will be solved for in the first
        # inner iteration.
        if init == 'random':
            Uinit = range(N)
            Uinit[0] = []
            for n in dimorder[1:]:
                Uinit[n] = numpy.random.random([X.shape[n], r])
        elif init == 'nvecs' or init == 'eigs':
            Uinit = range(N)
            Uinit[0] = []
            for n in dimorder[1:]:
                Uinit[n] = X.nvecs(n, r)  # first r leading eigenvecters
        else:
            raise TypeError('The selected initialization method is not supported')

    # Set up for iterations - initializing U and the fit.
    U = Uinit[:]
    fit = 0

    if printitn > 0:
        print('\nCP_ALS:\n')

    # Save hadamard product of each U[n].T*U[n]
    UtU = numpy.zeros([N, r, r])
    for n in range(N):
        if len(U[n]):
            UtU[n, :, :] = numpy.dot(U[n].T, U[n])

    for iter in range(1, maxiters + 1):
        fitold = fit
        oldX = X.data * 1.0

        # Iterate over all N modes of the Tensor
        for n in range(N):
            # Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
            temp1 = [n]
            temp2 = range(n)
            temp3 = range(n + 1, N)
            temp2.reverse()
            temp3.reverse()
            temp1[len(temp1):len(temp1)] = temp3
            temp1[len(temp1):len(temp1)] = temp2
            Xn = X.permute(temp1)
            Xn = Xn.tondarray()
            Xn = Xn.reshape([Xn.shape[0], Xn.size / Xn.shape[0]])
            tempU = U[:]
            tempU.pop(n)
            tempU.reverse()
            Unew = Xn.dot(khatrirao(tempU))

            # Compute the matrix of coefficients for linear system
            temp = range(n)
            temp[len(temp):len(temp)] = range(n + 1, N)
            y = numpy.prod(UtU[temp, :, :], axis=0)
            Unew = Unew.dot(numpy.linalg.inv(y))

            # Normalize each vector to prevent singularities in coefmatrix
            if iter == 1:
                lamb = numpy.sqrt(numpy.sum(numpy.square(Unew), 0))  # 2-norm
            else:
                lamb = numpy.max(Unew, 0)
                lamb = numpy.max([lamb, numpy.ones(r)], 0)  # max-norm

            lamb = [x * 1.0 for x in lamb]
            Unew = Unew / numpy.array(lamb)
            U[n] = Unew
            UtU[n, :, :] = numpy.dot(U[n].T, U[n])

        # Reconstructed fitted Ktensor
        P = pyten.tenclass.Ktensor(lamb, U)
        if recover == 0:
            if normX == 0:
                fit = P.norm() ** 2 - 2 * numpy.sum(X.tondarray() * P.tondarray())
            else:
                normresidual = numpy.sqrt(
                    abs(normX ** 2 + P.norm() ** 2 - 2 * numpy.sum(X.tondarray() * P.tondarray())))
                fit = 1 - (normresidual / normX)  # fraction explained by model
                fitchange = abs(fitold - fit)
        else:
            temp = P.tondarray()
            X.data = temp * (1 - omega) + X.data * omega
            fitchange = numpy.linalg.norm(X.data - oldX)

        # Check for convergence
        if (iter > 1) and (fitchange < fitchangetol):
            flag = 0
        else:
            flag = 1

        if (printitn != 0 and iter % printitn == 0) or ((printitn > 0) and (flag == 0)):
            if recover == 0:
                print 'CP_ALS: iterations={0}, f={1}, f-delta={2}'.format(iter, fit, fitchange)
            else:
                print 'CP_ALS: iterations={0}, f-delta={1}'.format(iter, fitchange)

        # Check for convergence
        if flag == 0:
            break

    return P, X
