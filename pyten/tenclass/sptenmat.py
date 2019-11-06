import numpy as np
import pyten.tools
import pyten.tenclass


class Sptenmat(object):
    """
    Store a Sparse Matricization of a Tensor object.
    """

    def __init__(self, x=None, rdim=None, cdim=None, tsize=None):
        """
        Create a Sptenmat object from a given Sptensor x
         ----------
        :param x: dense Tensor object.
        :param rdim: an one-dim array representing the arranged dimension index for the matrix column
        :param cdim: an one-dim array representing the arranged dimension index for the matrix row
        :param tsize: a tuple denoting the size of the original tensor
        :return: constructed Sparse Matricization of a Tensor object.
        ----------
        """

        if x is None:
            raise ValueError('Sptenmat: first argument cannot be empty.')

        if x.__class__ != pyten.tenclass.Sptensor:
            raise ValueError("Sptenmat: Tensor must be a Sptensor object.")

        if rdim is None:
            raise ValueError('Sptenmat: second argument cannot be empty.')

        if rdim.__class__ == list or rdim.__class__ == int:
            rdim = np.array(rdim) - 1

        if cdim is None:
            cdim = np.array([x for x in range(0, x.ndims) if x not in rdim])
        elif cdim.__class__ == list or cdim.__class__ == int:
            cdim = np.array(cdim) - 1
        else:
            raise ValueError("Sptenmat: incorrect specification of dimensions.")

        if not (range(0, x.ndims) == sorted(np.append(rdim, cdim))):
            print (range(0, x.ndims) == sorted(np.append(rdim, cdim)))
            raise ValueError("Tenmat: second argument must be a list or an integer.")

        self.shape = x.shape
        rsize = self.shape[rdim.tolist()]
        csize = self.shape[cdim.tolist()]

        rsubs = x.subs[:, rdim.tolist()]
        csubs = x.subs[:, cdim.tolist()]

        ridxs = np.array([pyten.tools.tools.sub2ind(rsize, rsub) for rsub in rsubs])
        cidxs = np.array([pyten.tools.tools.sub2ind(csize, csub) for csub in csubs])

        self.subs = np.concatenate((ridxs[np.newaxis].transpose(), cidxs[np.newaxis].transpose()), axis=1)
        self.vals = x.vals
        self.rdim = rdim
        self.cdim = cdim
        self.shape = tsize

    def tosptensor(self):
        # returns a Sptensor object based on a Sptenmat
        rsize = self.shape[self.rdim.tolist()]
        csize = self.shape[self.cdim.tolist()]

        ridxs = self.subs[:, 0]
        cidxs = self.subs[:, 1]

        rsubs = np.array([pyten.tools.tools.ind2sub(rsize, ridx) for ridx in ridxs])
        csubs = np.array([pyten.tools.tools.ind2sub(csize, cidx) for cidx in cidxs])

        newsubs = np.concatenate((rsubs, csubs), axis=1)

        order = np.concatenate((self.rdim, self.cdim), axis=0)

        newsubs = newsubs[:, order]

        return pyten.tenclass.Sptensor(newsubs, self.vals, self.shape)

    def __str__(self):
        ret = ""
        ret += "Sptenmat from an Sptensor of size {0} with {1} nonzeros\n".format(self.shape, len(self.vals))
        return ret
