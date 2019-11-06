import sys
sys.path.append('/path/to/pyten/tenclass')
import tensor, sptenmat
import numpy as np
import scipy as sp
from pyten.tools import tools


class Sptensor(object):
    """
    Store a Sparse Tensor object.
    """

    def __init__(self, subs, vals, shape):
        """
        Constructor for Sptensor object.
        ----------
        :param subs: subscripts (position) of each entry
        :param vals: a list of value of each entry in the tensor
        :param shape: can be tuple, numpy.array, or list of integers
        :return: constructed Sparse Tensor object.
        ----------

        """
        self.subs = None
        self.vals = None
        self.shape = None
        if (subs.__class__ == list):
            subs = np.array(subs)
        if (vals.__class__ == list):
            vals = np.array(vals)
        if (shape.__class__ == list):
            shape = np.array(shape)

        if not (tools.tt_subscheck(subs)):
            raise ValueError("Sptensor: error in subscripts.")
        if not (tools.tt_valscheck(vals)):
            raise ValueError("Sptensor: error in values.")
        if shape is not None:
            if not tools.tt_sizecheck(shape):
                raise ValueError("Sptensor: error in shape.")

        if len(vals) != len(subs):
            raise ValueError("Sptensor: number of subscripts and values must be equal.")

        self.subs = subs
        self.vals = vals
        self.shape = shape
        self.ndims = len(shape)

    def nnz(self):
        # returns the number of non-zero elements in the Sptensor
        return len(self.vals)

    def copy(self):
        # returns a deepcpoy of Sptensor object
        return Sptensor(self.subs.copy(), self.vals.copy(), self.shape)

    def dimsize(self, idx=None):
        # returns the size of the dimension specified by index
        if idx is None:
            raise ValueError("Sptensor: index of a dimension cannot be empty.")
        if idx.__class__ != int or idx < 1 or idx > self.ndims:
            raise ValueError("Sptensor: index of a dimension is an integer between 1 and NDIMS(Tensor).")
        idx = idx - 1
        return self.shape[idx]

    def totensor(self):
        # returns a new Tensor object that contains the same values
        data = np.zeros(self.shape)
        for i in range(0, len(self.vals)):
            data.put(int(tools.sub2ind(self.shape, self.subs[i])), self.vals[i])
        return tensor.Tensor(data)

    def permute(self, order=None):
        # returns a new Sptensor permuted by the given order
        if order is None:
            raise ValueError("Sptensor: order in permute cannot be empty.")
        if order.__class__ == list:
            order = np.array(order)

        order = order - 1
        if (self.ndims != len(order)) or (sorted(order) != range(0, self.ndims)):
            raise ValueError("Sptensor: invalid permute order.")

        newsubs = self.subs[:, order.tolist()]
        newvals = self.vals.copy()
        newsize = self.shape[order.tolist()]

        return Sptensor(newsubs, newvals, newsize)

    def ttm(self, mat=None, mode=None, option=None):
        if (mat is None):
            raise ValueError('Sptensor/TTM: matrix (mat) needs to be specified.')

        if (mode is None or mode.__class__ != int or mode > self.ndims() or mode < 1):
            raise ValueError('Sptensor/TTM: mode must be between 1 and NDIMS(Tensor).')

        if (mat.__class__ == list):
            matrix = np.array(mat)
        elif (mat.__class__ == np.ndarray):
            matrix = mat
        else:
            raise ValueError('Sptensor/TTM: matrix must be a list or a numpy.ndarray.')

        if (len(matrix.shape) != 2):
            raise ValueError('Sptensor/TTM: first argument must be a matrix.')

        if (matrix.shape[1] != self.shape[mode - 1]):
            raise ValueError('Sptensor/TTM: matrix dimensions must agree.')

        dim = mode - 1
        newsize = self.shape
        newsize[dim] = matrix.shape[1]

        rdim = [x for x in range(0, self.ndims) if x != dim]
        cdim = [dim]
        Xnt = sptenmat(self, rdim, cdim)

        rsubs = Xnt.subs[:, 0]
        csubs = Xnt.subs[:, 1]
        rsize = tools.prod(self.shape[rdim])
        csize = tools.prod(self.shape[cdim])
        XntDense = sp.sparse.coo_matrix((Xnt.vals, (rsubs, csubs)), shape=(rsize, csize))
        Z = XntDense * matrix.transpose()
        Z = tensor.Tensor(Z, newsize)
        if (Z.nnz() <= 0.5 * newsize.prod()):
            return Z.tosptensor()
        else:
            return Z
