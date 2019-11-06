import pyten.tenclass
import numpy as np
from pyten.tools import tools
import pyten.tenclass


class Tenmat(object):
    """
    Store a Matricization of a Tensor object.
    """

    def __init__(self, x=None, rdim=None, cdim=None, tsize=None):
        """
        Create a Tenmat object from a given Tensor X
         ----------
        :param x: dense Tensor object.
        :param rdim: an one-dim array representing the arranged dimension index for the matrix column
        :param cdim: an one-dim array representing the arranged dimension index for the matrix row
        :param tsize: a tuple denoting the size of the original tensor
        :return: constructed Matricization of a Tensor object.
        ----------
        """

        if x is None:
            raise ValueError('Tenmat: first argument cannot be empty.')

        if x.__class__ == pyten.tenclass.Tensor:
            # convert a Tensor to a matrix
            if rdim is None:
                raise ValueError('Tenmat: second argument cannot be empty.')

            if rdim.__class__ == list or rdim.__class__ == int:
                rdim = np.array(rdim) - 1

            self.shape = x.shape

            ##################
            if cdim is None:
                cdim = np.array([y for y in range(0, x.ndims) if y not in np.zeros(x.ndims - 1) + rdim])
            elif cdim.__class__ == list or cdim.__class__ == int:
                cdim = np.array(cdim) - 1
            else:
                raise ValueError("Tenmat: incorrect specification of dimensions.")
            ##################

            if not (range(0, x.ndims) == sorted(np.append(rdim, cdim))):
                raise ValueError("Tenmat: second argument must be a list or an integer.")

            self.rowIndices = rdim
            self.colIndices = cdim

            x = x.permute(np.append(rdim, cdim))

            ##################
            if type(rdim) != np.ndarray:
                row = tools.prod([self.shape[y] for y in [rdim]])
            else:
                row = tools.prod([self.shape[y] for y in rdim])

            if type(cdim) != np.ndarray:
                col = tools.prod([self.shape[y] for y in [cdim]])
            else:
                col = tools.prod([self.shape[y] for y in cdim])
            ##################

            self.data = x.data.reshape([row, col], order='F')
        elif x.__class__ == np.ndarray:
            # copy a matrix to a Tenmat object
            if len(x.shape) != 2:
                raise ValueError("Tenmat: first argument must be a 2-D numpy array when converting a matrix to Tenmat.")

            if tsize is None:
                raise ValueError("Tenmat: Tensor size must be specified as a tuple.")
            else:
                if rdim is None or cdim is None or rdim.__class__ != list or cdim.__class__ != list:
                    raise ValueError("Tenmat: second and third arguments must be specified with list.")
                else:
                    rdim = np.array(rdim) - 1
                    cdim = np.array(cdim) - 1
                    if tools.prod([tsize[idx] for idx in rdim]) != x.shape[0]:
                        raise ValueError("Tenmat: matrix size[0] does not match the Tensor size specified.")
                    if tools.prod([tsize[idx] for idx in cdim]) != x.shape[1]:
                        raise ValueError("Tenmat: matrix size[1] does not match the Tensor size specified.")
            self.data = x
            self.rowIndices = rdim
            self.colIndices = cdim
            self.shape = tsize

    def copy(self):
        # returns a deepcpoy of Tenmat object
        return Tenmat(self.data, self.rowIndices, self.colIndices, self.shape)

    def totensor(self):
        # returns a Tensor object based on a Tenmat
        order = np.append(self.rowIndices, self.colIndices)
        data = self.data.reshape([self.shape[idx] for idx in order], order='F')
        t_data = pyten.tenclass.Tensor(data).ipermute(list(order))
        return t_data

    def tondarray(self):
        # returns data of a Tenmat with a numpy.ndarray object
        return self.data

    def __str__(self):
        ret = ""
        ret += "Matrix corresponding to a Tensor of size {0}\n".format(self.shape)
        ret += "Row Indices {0}\n".format(self.rowIndices + 1)
        ret += "Column Indices {0}\n".format(self.colIndices + 1)
        return ret


if __name__ == '__main__':
    X = pyten.tenclass.Tensor(range(1, 25), [3, 2, 2, 2])
    print X.data[:, :, 0, 0]
    A = Tenmat(X, [1, 2], [4, 3])
    print A.data
    print A.totensor().data[:, :, 0, 0]
    print A.__str__()
