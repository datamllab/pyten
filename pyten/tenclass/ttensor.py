import pyten.tenclass
import numpy as np
from pyten.tools import tools


class Ttensor(object):
    """
    Tensor stored in decomposed form as a Tucker Decomposition.
    ----------
    Intended Usage
        Store the results of a CP decomposition.
    Parameters
    ----------
    """
    def __init__(self, core, us):
        """
        Constructor for Ttensor (Tucker Tensor) object with the core and latent matrices.
        ----------
        :type self: object
        :param core : core tensor in Tucker decomposition which is of
           the same size as original tensor

        :param us : list of ndarrays
           Factor matrices from which the Tensor representation
           is created. All factor matrices ``U[i]`` must have the
           same number of columns, but can have different
           number of rows.
        ----------
        """
        if core.__class__ != pyten.tenclass.Tensor:
            raise ValueError("Ttensor: core must a Tensor object.")

        if us.__class__ != list:
            raise ValueError("Ttensor: latent matrices should be a list of matrices.")
        else:
            for i,U in enumerate(us):
                if np.array(U).ndim != 2:
                    raise ValueError("Ttensor: latent matrix U{0} must be a 2-D matrix.".format(i))

        if core.ndims != len(us):
            raise ValueError("Ttensor: number of dimensions of the core Tensor is different with number of latent matrices.")

        k = core.shape

        for i in range(0, core.ndims):
            if k[i] != np.array(us[i]).shape[1]:
                raise ValueError("Ttensor: latent matrix U{0} does not have {1} columns.".format(i, k[i]))

        self.core = core.copy()
        self.us = us

        shape = []
        for i in range(0, core.ndims):
            shape.append(len(us[i]))
        self.shape = tuple(shape)
        self.ndims = core.ndims

    def size(self):
        """
        Returns the size of this tucker Tensor
        """
        return tools.prod(self.shape)

    def dimsize(self, idx = None):
        """
        Returns the size of the dimension specified by index
        """
        if idx is None:
            raise ValueError("Ttensor: index of a dimension cannot be empty.")
        if idx.__class__ != int or idx < 1 or idx > self.ndims:
            raise ValueError("Ttensor: index of a dimension is an integer between 1 and NDIMS(Tensor).")
        idx = idx - 1
        return self.shape[idx]

    def copy(self):
        """
        Returns a deepcpoy of tucker Tensor object
        """
        return Ttensor(self.core, self.us)

    def totensor(self):
        """
        Returns a Tensor object that is represented by the tucker Tensor
        """
        X = self.core.copy()
        for i in range(0, self.ndims):
            X=X.ttm(self.us[i], i+1)
        return X

    def __str__(self):
        ret = "Ttensor of size {0}\n".format(self.shape)
        ret += "Core = {0} \n".format(self.core.__str__())
        for i in range(0, self.ndims):
            ret += "u[{0}] =\n{1}\n".format(i, self.us[i])
        return ret
