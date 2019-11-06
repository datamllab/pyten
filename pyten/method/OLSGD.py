import copy

import numpy as np

import pyten.tenclass


class OLSGD(object):
    """
    This routine solves the online tensor completion problem
    with CP decomposition and online SGD optimization scheme
    """

    def __init__(self, rank=20, mu=0.01, lmbda=0.1):
        """ Initialization stage of OLSGD"""
        self.mu = mu
        if type(rank) == list or type(rank) == tuple:
            rank = rank[0]
        self.rank = rank
        self.lmbda = lmbda
        self.A = None
        self.B = None
        self.X = None
        self.timestamp = 0
        self.recx = None
        self.fitx = None
        self.rec = None
        self.omega = None
        self.shape = None

    def update(self, new_x, omega=None, mut=0.01, lmbdat=0.1):
        """
        Update stage of OLSGD
        --------
        :param
                 new_x: the new incoming data Tensor
                 As: a cell array contains the previous loading matrices of initX
        :return
                As: a cell array contains the updated loading matrices of initX.
                     To save time, As(N) is not modified, instead, projection of
                     new_x on time mode (alpha) is given in the output
                Ps, Qs: cell arrays contain the updated complementary matrices
                alpha: coefficient on time mode of new_x
        --------
        """

        self.mu = mut
        self.lmbda = lmbdat

        if type(new_x) != pyten.tenclass.Tensor and type(new_x) != np.ndarray:
            raise ValueError("OLSGD: cannot recognize the format of observed Tensor!")
        elif type(new_x) == np.ndarray:
            new_x = pyten.tenclass.Tensor(new_x)

        self.T = new_x
        if omega is None:
            self.omega = self.T.data * 0 + 1
        elif type(omega) != pyten.tenclass.Tensor and type(omega) != np.ndarray:
            raise ValueError("OLSGD: cannot recognize the format of indicator Tensor!")
        elif type(omega) == np.ndarray:
            self.omega = pyten.tenclass.Tensor(omega)
        else:
            self.omega = omega

        dims = list(new_x.shape)
        if len(dims) == 2:
            dims.insert(0, 1)
            self.T.data = self.T.data.reshape(dims)
            self.omega.data = self.omega.data.reshape(dims)

        if self.A is None:
            self.shape = copy.deepcopy(dims)
            self.A = np.random.rand(self.shape[1], self.rank)
            self.B = np.random.rand(self.shape[2], self.rank)
            self.C = np.zeros([self.shape[0], self.rank])
        else:
            self.shape[0] += dims[0]
            self.C = np.row_stack((self.C, np.zeros([dims[0], self.rank])))

        for i in range(dims[0]):
            self.timestamp += 1
            omg = self.omega.data[i, :, :]
            index = omg.nonzero()
            m = index[0]
            n = index[1]
            NNZ = len(m)
            temp1 = self.lmbda * np.identity(self.rank)
            temp2 = np.zeros([self.rank, 1])
            for j in range(NNZ):
                temp3 = self.A[m[j], :] * self.B[n[j], :]
                temp3 = temp3.reshape([temp3.shape[0], 1])
                temp1 += temp3.dot(temp3.T)
                temp2 += self.T.data[i, m[j], n[j]] * temp3
            temp = np.dot(np.linalg.inv(temp1), temp2)
            self.C[self.timestamp - 1, :] = temp.T
            tempA = copy.deepcopy(self.A)
            temp = temp.reshape(temp.shape[0])
            Err = omg * (self.T.data[i, :, :] - np.dot(np.dot(self.A, np.diag(temp)), self.B.T))
            self.A = (1 - self.lmbda / self.timestamp / self.mu) * self.A + \
                     1 / self.mu * np.dot(np.dot(Err, self.B), np.diag(temp))
            self.B = (1 - self.lmbda / self.timestamp / self.mu) * self.B + \
                     1 / self.mu * np.dot(np.dot(Err.T, tempA), np.diag(temp))

            tempfitx = np.dot(np.dot(self.A, np.diag(temp)), self.B.T)
            tempfitx = tempfitx.reshape([1, dims[1], dims[2]])
            if not self.X:
                self.X = pyten.tenclass.Tensor(tempfitx)
                temprecx = self.T.data[0, :, :] * self.omega.data[0, :, :] + self.X.data * (
                    1 - self.omega.data[0, :, :])
                self.rec = pyten.tenclass.Tensor(temprecx)
            else:
                self.X = pyten.tenclass.Tensor(np.row_stack((self.X.data, tempfitx)))
                temprecx = self.T.data[i, :, :] * self.omega.data[i, :, :] + tempfitx * (1 - self.omega.data[i, :, :])
                self.rec = pyten.tenclass.Tensor(np.row_stack((self.rec.data, temprecx)))
        self.fitx = pyten.tenclass.Tensor(self.X.data[-dims[0]:, :, :])
        self.recx = pyten.tenclass.Tensor(self.rec.data[-dims[0]:, :, :])
