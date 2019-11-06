import numpy as np
import pyten.tenclass
import pyten.tools


class MAST(object):
    """
    This routine solves the multi-aspect streaming tensor completion problem
    with CP decomposition and ADMM optimization scheme, (Only at current timestep T, T is not 1)
    Reference:  Song, Qingquan, et al. "Multi-Aspect Streaming Tensor Completion."
    Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2017.
    """

    def __init__(self, init=None, omega=None, decomp=None, olddims=np.zeros(3), newdims=np.zeros(3),
                 idx=np.ones([8, 3]), rank=2, lmbda=1, alpha=np.ones(3) / 3, maxiter=500, tol=1e-5, mu=1,
                 eta=1e-4, rho=1.05, printitn=500):
        """
        Initialization stage of MAST
        ----------
        :param
            init: A dict to save seperate tensors. (dim=N, size=2^N)
            omega: A dict to save observation index subtensors.
            decomp: Old decomposition matrices.
            olddims: Dimension of the tensor in former time step
            newdims: Dimension of the tensor in current time step
            idx: Block (subtensor) index.
            rank: Tensor Rank.
            lmbda: Regularization penalty for factorization (this penalty is incorported into 'alpha' in paper)
            alpha: Regularization penalty for trace norm term
            maxiter: Max iteration.
            tol: Convergence tolerance.
            mu:  Forgetting factor
            eta: step size
            rho: increasing factor
            printitn: is the printing control variable
        ----------
        """
        self.C = init
        self.pre_C = self.C
        self.OmegaC = omega
        self.As = decomp
        self.olddims = olddims
        self.newdims = newdims
        self.idx = idx
        if type(rank) == list or type(rank) == tuple:
            rank = rank[0]
        self.rank = rank
        self.lmbda = lmbda
        self.alpha = alpha
        self.maxIter = maxiter
        self.tol = tol
        self.mu = mu
#        dimchange = np.zeros(3)
#        timestep = 1
#        self.dimchange = dimchange
#        self.timestamp = timestep
        self.N = idx.shape[1]  # Tensor dim
        self.eta = eta
        self.rho = rho
        self.Inc_size = self.newdims - self.olddims  # Size of the dim increase
        self.stopC1 = None
        self.stopC2 = None
        self.stopC = None
        self.errList = []  # Initialize error list used to record training errors

        self.printitn = printitn
        # Initialize latent matrices
        # CAs{1,:} old decomposition (=As), CAs{2,:} new part (randomly initialize)
        # Zs  with zeros and Ys with zeros.
        self.CAs1 = range(self.N)
        self.CAs2 = range(self.N)
        self.Zs = range(self.N)
        self.Ys = range(self.N)
        for i in range(self.N):
            self.CAs1[i] = self.As[i]
            self.CAs2[i] = np.random.random([self.Inc_size[i], self.rank])
            self.Ys[i] = np.zeros([self.newdims[i], self.rank])
            self.Zs[i] = np.zeros([self.newdims[i], self.rank])

        self.finalAs = self.As

    #        self.recx = None
    #        self.fitx = None
    #        self.rec = None
    #        self.shape = None

    def update(self):
        """
        Update stage of MAST
        """
        for k in range(self.maxIter):
            self.stopC1 = 0
            self.stopC2 = 0

            # Update step eta
            self.eta *= self.rho

            # update Zs
            for i in range(self.N):
                temp_1 = np.row_stack((self.CAs1[i], self.CAs2[i])) - self.Ys[i] / self.eta
                u, s, v = np.linalg.svd(temp_1)
                for j in range(s.size):
                    s[j] = max(s[j], self.alpha[i] / self.eta)
                [m, n] = temp_1.shape
                if m > n:
                    s = np.dot(np.eye(m, n), np.diag(s))
                else:
                    s = np.dot(np.diag(s), np.eye(m, n))
                self.Zs[i] = np.dot(np.dot(u, s), v)

            # Update CAs
            for n in range(self.N):
                tempu = range(self.N)
                temp_d1 = 0
                temp_d2 = 0
                for j in range(1, 2 ** self.N):
                    if self.C[str(self.idx[2, ])].all():
                        continue
                    else:
                        for p in range(self.N):
                            if self.idx[j, p] == 1:
                                tempu[p] = self.CAs1[p]
                            else:
                                tempu[p] = self.CAs2[p]
                        if self.idx[j, n] == 1:
                            temp_d1 += pyten.tools.mttkrp(pyten.tenclass.Tensor(self.C[str(self.idx[j, ])]), tempu, n)
                        else:
                            temp_d2 += pyten.tools.mttkrp(pyten.tenclass.Tensor(self.C[str(self.idx[j, ])]), tempu, n)

                # Core idea
                had = np.array([])
                index = range(n, self.N)
                index[0:0] = range(n - 1)
                for i in index:
                    if had.all():
                        had = np.dot(self.As[i].T, self.CAs1[i])
                    else:
                        had = had * (np.dot(self.As[i].T, self.CAs1[i]))

                temp_d1 = temp_d1 + np.dot(self.As[n], had) * self.mu
                temp_z1 = self.eta * self.Zs[n][0:self.olddims[n], ] + self.Ys[n][0:self.olddims[n], ]
                temp_z2 = self.eta * self.Zs[n][self.olddims[n]:, ] + self.Ys[n][self.olddims[n]:, ]

                had = np.array([])
                had1 = np.array([])
                for i in index:
                    if had.all():
                        had = np.dot(self.CAs1[i].T, self.CAs1[i]) + np.dot(self.CAs2[i].T, self.CAs2[i])
                        had1 = np.dot(self.CAs1[i].T, self.CAs1[i])
                    else:
                        had = had * (np.dot(self.CAs1[i].T, self.CAs1[i]) + np.dot(self.CAs2[i].T, self.CAs2[i]))
                        had1 = had1 * (np.dot(self.CAs1[i].T, self.CAs1[i]))

                temp_b_1 = self.lmbda * (had - (1 - self.mu) * had1) + self.eta * np.eye(self.rank)
                temp_b_2 = self.lmbda * had + self.eta * np.eye(self.rank)
                self.CAs1[n] = np.dot((self.lmbda * temp_d1 + temp_z1), np.linalg.inv(temp_b_1))
                if self.newdims[n] > self.olddims[n]:
                    self.CAs2[n] = np.dot((self.lmbda * temp_d2 + temp_z2), np.linalg.inv(temp_b_2))

            # Update C (Original block tensors)
            for j in range(1, 2 ** self.N):
                for p in range(self.N):
                    if self.idx[j, p] == 0:
                        tempu[p] = self.CAs1[p]
                    else:
                        tempu[p] = self.CAs2[p]
                tempt = pyten.tenclass.Ktensor(np.ones(self.rank), tempu).tondarray()

                # Can be improved here
                if self.C[str(self.idx[j, ])].all():
                    continue
                else:
                    if type(self.C[str(self.idx[j, ])]) == 'pyten.tenclass.tensor.Tensor':
                        self.C[str(self.idx[j, ])] = self.C[str(self.idx[j, ])].tondarray()
                        self.C[str(self.idx[j, ])] = tempt * self.OmegaC[str(self.idx[j, ])].data + \
                                                    self.C[str(self.idx[j, ])] * (
                                                    1 - self.OmegaC[str(self.idx[j, ])].data)
                    # tempidx=find((1-OmegaC{idx{j,:}})==1)
                    #                        C{idx{j,:}}(reshape(tempidx,[length(tempidx),1]))=tempt*self.OmegaC[str(self.idx[j,])].data
                    else:
                        self.C[str(self.idx[j, ])] = tempt * self.OmegaC[str(self.idx[j, ])] + \
                                                    self.C[str(self.idx[j, ])] * (
                                                    1 - self.OmegaC[str(self.idx[j, ])])
                    self.stopC1 += pyten.tenclass.Tensor(self.pre_C[str(self.idx[j, ])]
                                                         - self.C[str(self.idx[j, ])]).norm() ** 2
                    self.stopC2 += pyten.tenclass.Tensor(self.C[str(self.idx[j, ])]).norm() ** 2

            self.stopC = np.sqrt(self.stopC1) / np.sqrt(self.stopC2)
            # update Lagrange multiper Ys
            for i in range(self.N):
                self.Ys[i] += self.eta * (self.Zs[i] - np.row_stack((self.CAs1[i], self.CAs2[i])))

            # checking the stop criteria
            self.errList.append(self.stopC)
            self.pre_C = self.C.copy()

            if (k + 1) % self.printitn == 0:
                print 'MAST: iterations={0}, difference={1}'.format(k + 1, self.errList[-1])
            elif self.stopC < self.tol:
                print 'MAST: iterations={0}, difference={1}'.format(k + 1, self.errList[-1])

            if self.stopC < self.tol:
                break

        for i in range(self.N):
            self.finalAs[i] = np.row_stack((self.CAs1[i], self.CAs2[i]))

        print 'MAST: iterations={0}, difference={1}'.format(k + 1, self.errList[-1])
