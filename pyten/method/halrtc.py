import numpy as np
import math
import pyten.tenclass


def pro_to_trace_norm(z, tau):
    m = z.shape[0]
    n = z.shape[1]
    if 2 * m < n:
        [U, Sigma2, V] = np.linalg.svd(np.dot(z, z.T))
        S = np.sqrt(Sigma2)
        tol = np.max(z.shape) * (2 ** int(math.log(max(S), 2))) * 2.2204 * 1E-16
        k = np.sum(S > max(tol, tau))
        mid = [max(S[i] - tau, 0) * 1.0 / S[i] for i in range(k)]
        X = np.dot(np.dot(U[:, 0:k], np.dot(np.diag(mid), U[:, 0:k].T)), z)
        return X, k, Sigma2
    if m > 2 * n:
        z = z.T
        [U, Sigma2, V] = np.linalg.svd(np.dot(z, z.T))
        S = np.sqrt(Sigma2)
        tol = np.max(z.shape) * (2 ** int(math.log(max(S), 2))) * 2.2204 * 1E-16
        k = np.sum(S > max(tol, tau))
        mid = [max(S[i] - tau, 0) * 1.0 / S[i] for i in range(k)]
        X = np.dot(np.dot(U[:, 0:k], np.dot(np.diag(mid), U[:, 0:k].T)), z)
        return X.T, k, Sigma2

    [U, S, V] = np.linalg.svd(z)
    Sigma2 = S ** 2
    k = sum(S > tau)
    X = np.dot(U[:, 0:k], np.dot(np.diag(S[0:k] - tau), V[0:k, :]))
    return X, n, Sigma2


def halrtc(x, omega=None, alpha=None, beta=None, max_iter=100, epsilon=1e-5, printitn=100):
    """
    High Accuracy Low Rank Tensor Completion (HaLRTC).
    Reference: "Tensor Completion for Estimating Missing Values. in Visual Data", PAMI, 2012.
    """

    T = x.data.copy()
    N = x.ndims
    dim = x.shape
    normX = x.norm()

    if printitn == 0:
        printitn = max_iter
    if omega is None:
        omega = x.data * 0 + 1

    if alpha is None:
        alpha = np.ones([N])
        alpha = alpha / sum(alpha)

    if beta is None:
        beta = 1e-6

    # Initialization
    x.data[omega == 0] = np.mean(x.data[omega == 1])
    errList = np.zeros([max_iter, 1])

    Y = range(N)
    M = range(N)

    for i in range(N):
        Y[i] = x.data
        M[i] = np.zeros(dim)

    Msum = np.zeros(dim)
    Ysum = np.zeros(dim)

    for k in range(max_iter):

        if (k + 1) % printitn == 0 and k != 0 and printitn != max_iter:
            print 'HaLRTC: iterations = {0}  difference = {1}\n'.format(k, errList[k - 1])

        beta = beta * 1.05

        # Update Y
        Msum = 0 * Msum
        Ysum = 0 * Ysum
        for i in range(N):
            A = pyten.tenclass.Tensor(x.data - M[i] / beta)
            temp = pyten.tenclass.Tenmat(A, i + 1)
            [temp1, tempn, tempSigma2] = pro_to_trace_norm(temp.data, alpha[i] / beta)
            temp.data = temp1
            Y[i] = temp.totensor().data
            Msum = Msum + M[i]
            Ysum = Ysum + Y[i]

        # update x
        Xlast = x.data.copy()
        Xlast = pyten.tenclass.Tensor(Xlast)
        x.data = (Msum + beta * Ysum) / (N * beta)
        x.data = T * omega + x.data * (1 - omega)

        # Update M
        for i in range(N):
            M[i] = M[i] + beta * (Y[i] - x.data)

        # Compute the error
        diff = x.data - Xlast.data
        errList[k] = np.linalg.norm(diff) / normX


        if errList[k] < epsilon:
            errList = errList[0:(k + 1)]
            break


    print 'HaLRTC ends: total iterations = {0}   difference = {1}\n\n'.format(k + 1, errList[k])
    return x
