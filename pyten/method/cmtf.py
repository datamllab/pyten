import numpy
import pyten.method
import pyten.tenclass


def cmtf(x, y=None, c_m=None, r=2, omega=None, tol=1e-4, maxiter=100, init='random', printitn=100):
    """
    CMTF Compute a Coupled Matrix and Tensor Factorization (and recover the Tensor).
    ---------
    :param   'x'  - Tensor
    :param   'y'  - Coupled Matries
    :param  'c_m' - Coupled Modes
    :param   'r'  - Tensor Rank
    :param  'omega'- Index Tensor of Obseved Entries
    :param  'tol' - Tolerance on difference in fit {1.0e-4}
    :param 'maxiters' - Maximum number of iterations {50}
    :param 'init' - Initial guess [{'random'}|'nvecs'|cell array]
    :param 'printitn' - Print fit every n iterations; 0 for no printing {1}
    ---------
    :return
     P: Decompose result.(kensor)
     x: Recovered Tensor.
     V: Projection Matrix.
    ---------
    """

    # Setting Parameters
    if type(r) == list or type(r) == tuple:
        r = r[0]

    if y is None:
        [P, x] = pyten.method.cp_als(x, r, omega, tol, maxiter, init, printitn)
        V = None
        return P, x, V

    if c_m is None:
        c_m = 0
    elif int == type(c_m):
        c_m = c_m - 1
    else:
        c_m = [i - 1 for i in c_m]

    # Construct omega if no input
    if omega is None:
        omega = x.data * 0 + 1

    # Extract number of dimensions and norm of x.
    N = x.ndims
    normX = x.norm()
    dimorder = range(N)  # 'dimorder' - Order to loop through dimensions {0:(ndims(A)-1)}

    # Define convergence tolerance & maximum iteration
    fitchangetol = 1e-4
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
            if Uinit[n].shape != (x.shape[n], r):
                raise IndexError('OPTS.init{%d} is the wrong size', n)
    else:
        # Observe that we don't need to calculate an initial guess for the
        # first index in dimorder because that will be solved for in the first
        # inner iteration.
        if init == 'random':
            Uinit = range(N)
            Uinit[0] = []
            for n in dimorder[1:]:
                Uinit[n] = numpy.random.random([x.shape[n], r])
        elif init == 'nvecs' or init == 'eigs':
            Uinit = range(N)
            Uinit[0] = []
            for n in dimorder[1:]:
                Uinit[n] = x.nvecs(n, r)  # first r leading eigenvecters
        else:
            raise TypeError('The selected initialization method is not supported')

        # Set up for iterations - initializing U and the fit.
        U = Uinit[:]
        if type(c_m) == int:
            V = numpy.random.random([y.shape[1], r])
        else:
            V = [numpy.random.random([y[i].shape[1], r]) for i in range(len(c_m))]
        fit = 0

        if printitn > 0:
            print('\nCMTF:\n')

    # Save hadamard product of each U[n].T*U[n]
    UtU = numpy.zeros([N, r, r])
    for n in range(N):
        if len(U[n]):
            UtU[n, :, :] = numpy.dot(U[n].T, U[n])

    for iter in range(1, maxiters + 1):
        fitold = fit
        oldX = x.data * 1.0

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
            xn = x.permute(temp1)
            xn = xn.tondarray()
            xn = xn.reshape([xn.shape[0], xn.size / xn.shape[0]])
            tempU = U[:]
            tempU.pop(n)
            tempU.reverse()
            Unew = xn.dot(pyten.tools.khatrirao(tempU))

            # Compute the matrix of coefficients for linear system
            temp = range(n)
            temp[len(temp):len(temp)] = range(n + 1, N)
            B = numpy.prod(UtU[temp, :, :], axis=0)
            if int != type(c_m):
                tempCM = [i for i, a in enumerate(c_m) if a == n]
            elif c_m == n:
                tempCM = [0]
            else:
                tempCM = []
            if tempCM != [] and int != type(c_m):
                for i in tempCM:
                    B = B + numpy.dot(V[i].T, V[i])
                    Unew = Unew + numpy.dot(y[i], V[i])
                    V[i] = numpy.dot(y[i].T, Unew)
                    V[i] = V[i].dot(numpy.linalg.inv(numpy.dot(Unew.T, Unew)))
            elif tempCM != []:
                B = B + numpy.dot(V.T, V)
                Unew = Unew + numpy.dot(y, V)
                V = numpy.dot(y.T, Unew)
                V = V.dot(numpy.linalg.inv(numpy.dot(Unew.T, Unew)))
            Unew = Unew.dot(numpy.linalg.inv(B))
            U[n] = Unew
            UtU[n, :, :] = numpy.dot(U[n].T, U[n])

        # Reconstructed fitted Ktensor
        lamb = numpy.ones(r)
        P = pyten.tenclass.Ktensor(lamb, U)
        if recover == 0:
            if normX == 0:
                fit = P.norm() ** 2 - 2 * numpy.sum(x.tondarray() * P.tondarray())
            else:
                normresidual = numpy.sqrt(normX ** 2 + P.norm() ** 2 - 2 * numpy.sum(x.tondarray() * P.tondarray()))
                fit = 1 - (normresidual / normX)  # fraction explained by model
                fitchange = abs(fitold - fit)
        else:
            temp = P.tondarray()
            x.data = temp * (1 - omega) + x.data * omega
            fitchange = numpy.linalg.norm(x.data - oldX)

        # Check for convergence
        if (iter > 1) and (fitchange < fitchangetol):
            flag = 0
        else:
            flag = 1

        if (printitn != 0 and iter % printitn == 0) or ((printitn > 0) and (flag == 0)):
            if recover == 0:
                print 'CMTF: iterations={0}, f={1}, f-delta={2}'.format(iter, fit, fitchange)
            else:
                print 'CMTF: iterations={0}, f-delta={1}'.format(iter, fitchange)
        if flag == 0:
            break

    return P, x, V
