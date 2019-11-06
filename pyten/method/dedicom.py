import numpy as np
import pyten.tenclass
import pyten.tools


class DEDICOM(object):
    """
    This route has some problem and will be changed in the future.
    ASALSAN Algorithm for DEDICOM proposed in :
    Bader, Brett W., Richard A. Harshman, and Tamara G. Kolda. "Temporal
    analysis of semantic graphs using ASALSAN." IEEE ICDM 2007.
    """

    def __init__(self, x, rank=20, gamma=1e-1, lamb=1e-3, tol=1e-5, maxiter=500, printitn=100):
        """Initialization stage of Dedicom method."""
        if not x:
            raise ValueError("DEDICOM: this tensor cannot be empty!")
        elif type(x) != pyten.tenclass.Tensor and type(x) != np.ndarray:
            raise ValueError("DEDICOM: cannot recognize the format of this tensor!")
        elif type(x) == np.ndarray:
            self.T = pyten.tenclass.Tensor(x)
            self.X = self.T.data
        else:
            self.T = x
            self.X = self.T.data

        self.ndims = self.T.ndims
        self.shape = self.T.shape

        if self.shape[0] != self.shape[1]:
            raise ValueError("DEDICOM: the first and second modes of this tensor should be the same!")

        if self.ndims != 3:
            raise ValueError("DEDICOM: this tensor should be a third-order tensor!")

        if printitn == 0:
            printitn = maxiter

        self.rank = rank
        self.maxiter = maxiter
        self.tol = tol
        self.printitn = printitn
        self.A = None
        self.R = None
        self.D = None
        self.fit = None
        self.errList = []
        self.gamma = gamma
        self.lamb = lamb
        self.sigma_new = 0
        self.sigma_old = 1e-6

    def initialize(self):
        """Random initialization of all decomposition matrices and tensors."""
        self.A = np.random.rand(self.shape[0], self.rank)
        self.R = np.random.rand(self.rank, self.rank)
        self.D = np.zeros([self.rank, self.rank, self.shape[2]])
        diagonal = np.random.rand(self.rank, self.shape[2])
        for i in range(self.shape[2]):
            self.D[:, :, i] = np.diag(diagonal[:, i])

    def run(self):
        """Running (Optimization) stage for Dedicom decomposition. """
        self.errList = []
        self.initialize()
        for k in range(self.maxiter):
            # update A
            temp1 = 0
            temp2 = 0
            for i in range(self.shape[2]):
                temp_a = self.A.dot(self.D[:, :, i])
                temp_r = self.R.dot(self.D[:, :, i])
                temp_rt = self.R.T.dot(self.D[:, :, i])
                temp1 += self.X[:, :, i].dot(temp_a).dot(temp_rt) + self.X[:, :, i].T.dot(temp_a).dot(temp_r)
                temp_datad = temp_a.T.dot(temp_a)
                temp2 += temp_rt.T.dot(temp_datad).dot(temp_rt) + temp_r.T.dot(temp_datad).dot(temp_r)
                self.A = temp1.dot(np.linalg.inv(temp2))

            # update R
            temp1 = 0
            temp2 = 0
            for i in range(self.shape[2]):
                temp_a = self.A.dot(self.D[:, :, i])
                temp_at = self.D[:, :, i].dot(self.A.T)
                temp_ata = temp_at.dot(temp_a)
                temp1 += np.kron(temp_ata, temp_ata)
                temp_daxad = temp_at.dot(self.X[:, :, i]).dot(temp_a)
                temp2 += np.reshape(temp_daxad, np.prod(temp_daxad.shape))
                self.R = np.reshape(np.linalg.inv(temp1).dot(temp2), [self.rank, self.rank])

            # solve D using Newton's method
            for i in range(self.shape[2]):
                hessian = np.zeros([self.rank, self.rank])
                gradient = np.zeros(self.rank)
                temp_ad = self.A.dot(self.D[:, :, i])
                # temp_da = self.D[:, :, i].dot(self.A.T)
                temp_da = temp_ad.T
                temp1 = self.X[:, :, i]-temp_ad.dot(self.R).dot(temp_da)
                for s in range(self.rank):
                    temp2 = (np.outer(temp_ad.dot(self.R[:, s]), self.A[:, s].T)
                             + np.outer(self.A[:, s], self.R[s, :]).dot(temp_da))
                    gradient[s] = -2*np.sum(temp1*temp2)
                    for t in range(self.rank):
                        temp_asat = np.outer(self.A[:, s], self.A[:, t].T)
                        temp3 = self.R[s, t]*temp_asat+self.R[t, s]*temp_asat.T
                        temp4 = (np.outer(temp_ad.dot(self.R[:, t]), self.A[:, t].T)
                                 + np.outer(self.A[:, t], self.R[t, :]).dot(temp_da))
                        hessian[s, t] = -2*np.sum(temp1*temp3-temp2*temp4)
                e = abs(np.linalg.eigvals(hessian).min())
                hessian = hessian + (np.eye(hessian.shape[0]) * e)
                print np.linalg.eigvals(hessian)
                self.D[:, :, i] = (np.diag(np.diag(self.D[:, :, i])
                                   - self.gamma*np.linalg.inv(hessian+self.lamb*np.identity(self.rank)).dot(gradient)))
                self.sigma_new += np.linalg.norm(temp1)**2

            # checking the stop criteria
            error = abs(self.sigma_new-self.sigma_old) #/self.sigma_old
            self.errList.append(error)
            if (k + 1) % self.printitn == 0:
                print 'DEDICOM: iterations={0}, difference={1}, fit_difference={2}'.format(k + 1, self.errList[-1],
                                                                                           self.sigma_new)
            elif error < self.tol:
                print 'DEDICOM: iterations={0}, difference={1}, fit_difference={2}'.format(k + 1, self.errList[-1],
                                                                                           self.sigma_new)

            if error < self.tol:
                break
            else:
                self.sigma_old = self.sigma_new
                self.sigma_new = 0

        self.fit = self.X
        for i in range(self.shape[2]):
            temp_ad = self.A.dot(self.D[:, :, i])
            temp_da = self.D[:, :, i].dot(self.A.T)
            self.fit[:, :, i] = temp_ad.dot(self.R).dot(temp_da)
        self.fit = pyten.tenclass.Tensor(self.fit)
