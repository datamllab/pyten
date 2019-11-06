# coding=utf-8
import numpy as np
import pyten.tenclass
import pyten.tools


class PARAFAC2(object):
    """
    ALS Algorithm for PARAFAC2 proposed in :
    PARAFAC2-part i. A direct fitting algorithm for the PARAFAC2 model. Journal of Chemometrics, 13(3-4):275–294, 1999.
    Henk AL Kiers, Jos MF Ten Berge, and Rasmus Bro.
    """

    def __init__(self, x, rank=20, tol=1e-5, maxiter=500, printitn=100):
        """Initialization stage of Parafac2 method."""
        if not x:
            raise ValueError("PARAFAC2: the multiset cannot be empty!")
        elif type(x) != list:
            raise ValueError("PARAFAC2: cannot recognize the format of this multiset!")
        else:
            self.X = x

        self.K = len(self.X)
        self.L = self.X[0].shape[1]
        for k in range(1, self.K):
            if self.X[k].shape[1] != self.L:
                raise ValueError("PARAFAC2: the column size of multiset slices should be the same!")

        if printitn == 0:
            printitn = maxiter

        self.rank = rank
        self.maxiter = maxiter
        self.tol = tol
        self.printitn = printitn
        self.U = None
        self.H = None
        self.S = None
        self.V = None
        self.fit = range(self.K)
        self.errList = []
        self.sigma_new = 0
        self.sigma_old = 1e-6

    def initialize(self):
        """Random initialization of all decomposition matrices and tensors"""
        self.U = range(self.K)
        self.H = np.identity(self.rank)
        temp = 0
        self.S = np.zeros([self.rank, self.rank, self.K])
        for k in range(self.K):
            self.S[:, :, k] = np.identity(self.rank)
            temp += self.X[k].T.dot(self.X[k])
        [eigval, eigvec] = np.linalg.eig(temp)
        self.V = eigvec[:, range(self.rank)]

    #        for k in range(self.K):
    #            self.normX += np.linalg.norm(self.X[k]) ** 2
    #        self.normX = self.normX ** 0.5

    def run(self):
        """Running (Optimization) stage for Parafac2 decomposition."""
        self.errList = []
        self.initialize()

        for i in range(self.maxiter):
            # update U
            for k in range(self.K):
                [p, sigma, q] = np.linalg.svd(self.H.dot(self.S[:, :, k]).dot(self.V.T).dot(self.X[k].T),
                                              full_matrices=False)
                self.U[k] = q.T.dot(p.T)
                self.U[k] = self.U[k].real

            # calculate temporal variable y
            y = np.zeros([self.rank, self.L, self.K])
            for k in range(self.K):
                y[:, :, k] = self.U[k].T.dot(self.X[k])

            # get H, V, and temps by running a single iteration of CP_ALS
            if i == 0:
                [cp, rec] = pyten.method.cp_als(pyten.tenclass.Tensor(y),
                                                self.rank, tol=self.tol, maxiter=1, printitn=0)
            else:
                [cp, rec] = pyten.method.cp_als(pyten.tenclass.Tensor(y),
                                                self.rank, tol=self.tol, maxiter=1,
                                                init=[self.H, self.V, temps], printitn=0)
            self.H = cp.Us[0]
            self.V = cp.Us[1]
            temps = cp.Us[2].dot(np.diag(cp.lmbda))

            # update S
            for k in range(self.K):
                self.S[:, :, k] = np.diag(temps[k, :])

            # checking the stop criteria
            # error = 0
            for k in range(self.K):
                temp = self.U[k].dot(self.H).dot(self.S[:, :, k]).dot(self.V.T)
                self.sigma_new += np.linalg.norm(temp - self.X[k]) ** 2

            error = abs(self.sigma_new - self.sigma_old) #/ self.sigma_old
            self.errList.append(error)
            if (i + 1) % self.printitn == 0:
                print 'PARAFAC2: iterations={0}, difference={1}, fit_difference={2}'.format(i + 1, self.errList[-1],
                                                                                            self.sigma_new)
            elif error < self.tol:
                print 'PARAFAC2: iterations={0}, difference={1}, fit_difference={2}'.format(i + 1, self.errList[-1],
                                                                                            self.sigma_new)

            if error < self.tol:
                break
            else:
                self.sigma_old = self.sigma_new
                self.sigma_new = 0

        for k in range(self.K):
            self.fit[k] = self.U[k].dot(self.H).dot(self.S[:, :, k]).dot(self.V.T)
