import numpy as np
import math
import pyten.tenclass


def truncate(z, tau):
    # z is a Tenmat
    m = z.shape[0]
    n = z.shape[1]
    if 2 * m < n:
        [U, Sigma2, V] = np.linalg.svd(np.dot(z, z.T))
        S = np.sqrt(Sigma2)
        tol = np.max(z.shape) * (2 ** int(math.log(max(S), 2))) * 2.2204 * 1E-16
        k = np.sum(S > max(tol, tau))
        mid = [max(S[i] - tau, 0) * 1.0 / S[i] for i in range(k)]
        X = np.dot((np.eye(m) - np.dot(U[:, 0:k], np.dot(np.diag(mid), U[:, 0:k].T))), z)
        return X, Sigma2

    if 2 * m > n:
        z = z.T
        [U, Sigma2, V] = np.linalg.svd(np.dot(z, z.T))
        S = np.sqrt(Sigma2)
        tol = np.max(z.shape) * (2 ** int(math.log(max(S), 2))) * 2.2204 * 1E-16
        k = np.sum(S > max(tol, tau))
        mid = [max(S[i] - tau, 0) * 1.0 / S[i] for i in range(k)]
        X = np.dot((np.eye(n) - np.dot(U[:, 0:k], np.dot(np.diag(mid), U[:, 0:k].T))), z)
        return X.T, Sigma2

    [U, sigma, V] = np.linalg.svd(z)
    Sigma2 = sigma ** 2
    n = np.sum(sigma > tau)
    X = z - np.dot(U[:, 0:n], np.dot(np.diag(sigma[0:n] - tau), V[0:n, :]))
    return X, Sigma2


def falrtc(x, omega=None, alpha=None, mu=None, l=1e-5, c=0.6, max_iter=100, epsilon=1e-5, printitn=100):
    """
    Fast Low Rank Tensor Completion (FaLRTC).
    Reference: "Tensor Completion for Estimating Missing Values in Visual Data", PAMI, 2012.
    """
    N = x.ndims
    dim = x.shape
    if printitn == 0:
        printitn = max_iter
    if omega is None:
        omega = x.data * 0 + 1

    if alpha is None:
        alpha = np.ones([N])
        alpha = alpha / sum(alpha)

    if mu is None:
        mu = 5.0 * alpha / np.sqrt(dim)

    normX = x.norm()
    # initialization
    x.data[omega == 0] = np.mean(x.data[omega == 1])

    Y = x.data.copy()
    Y = pyten.tenclass.Tensor(Y)
    Z = x.data.copy()
    Z = pyten.tenclass.Tensor(Z)
    B = 0

    Gx = np.zeros(dim)
    errList = np.zeros([max_iter, 1])

    Lmax = 10 * np.sum(1.0 / mu)

    tmp = np.zeros([N])
    for i in range(N):
        tempX = pyten.tenclass.Tenmat(x, i + 1)
        [U, sigma, V] = np.linalg.svd(tempX.data)
        tmp[i] = np.max(sigma) * alpha[i] * 0.3

    P = 1.15
    flatNum = 15
    slope = (tmp - mu) / (1 - (max_iter - flatNum) ** (-P))
    offset = (mu * (max_iter - flatNum) ** P - tmp) / ((max_iter - flatNum) ** P - 1)

    mu0 = mu * 1.0
    for k in range(max_iter):
        if (k + 1) % printitn == 0 and k != 0 and printitn != max_iter:
            print 'FaLRTC: iterations = {0}   difference = {1}\n'.format(k, errList[k - 1])

        # update mu
        t = slope * 1.0 / (k + 1) ** P + offset
        mu = [max(t[j], mu0[j]) * 1.0 for j in range(N)]

        a2m = alpha ** 2 / mu
        ma = mu / alpha

        Ylast = Y.data.copy()
        Ylast = pyten.tenclass.Tensor(Ylast)
        while True:
            b = (1 + np.sqrt(1 + 4 * l * B)) * 1.0 / (2.0 * l)
            x.data = b * 1.0 / (B + b) * Z.data + B * 1.0 / (B + b) * Ylast.data

            # compute f'(x) namely "Gx" and f(x) namely "fx"
            Gx = Gx * 0
            fx = 0
            for i in range(N):
                temp = pyten.tenclass.Tenmat(x, i + 1)
                [tempX, sigma2] = truncate(temp.data, ma[i])
                temp.data = tempX
                temp = temp.totensor()
                Gx = Gx + a2m[i] * temp.data
                fx = fx + a2m[i] * (sum(sigma2) - sum([(max(np.sqrt(a) - ma[i], 0)) ** 2 for a in sigma2]))
            Gx[omega == 1] = 0

            # compute f(Ytest) namely fy
            Y.data = x.data - Gx / l
            fy = 0
            for i in range(N):
                tempY = pyten.tenclass.Tenmat(Y, i + 1)
                [U, sigma, V] = np.linalg.svd(tempY.data)
                fy = fy + a2m[i] * (sum(sigma ** 2) - sum(([(max(q - ma[i], 0)) ** 2 for q in sigma])))

            # test if l(fx-fy) > \|Gx\|^2
            if (fx - fy) * l < np.sum(Gx[:] ** 2):
                if l > Lmax:
                    errList = errList[0:(k + 1)]
                    print 'FaLRTC: iterations = {0}   difference = {1}\n Exceed the Maximum ' \
                          'Lipschitiz Constan\n\n'.format(k + 1, errList[k])
                    return Y
                l = l / c
            else:
                break

        # Check Convergence
        diff = Y.data - Ylast.data
        errList[k] = np.linalg.norm(diff) / normX
        if errList[k] < epsilon:
            break

        # update Z, Y, and B
        Z.data = Z.data - b * Gx
        B = B + b

    errList = errList[0:(k + 1)]
    print 'FaLRTC ends: total iterations = {0}   difference = {1}\n\n'.format(k + 1, errList[k])
    return Y
