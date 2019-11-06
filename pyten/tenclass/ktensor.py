import numpy as np
import pyten.tenclass
import pyten.tools


class Ktensor(object):
    """
    Tensor stored in decomposed form as a Kruskal operator (CP decomposition).
    ----------
    Intended Usage
        Store the results of a CP decomposition.
    Parameters
    ----------
    """

    def __init__(self, lmbda=None, us=None):
        """
        Constructor for Ktensor (CP Tensor) object with the weights and latent matrices.
        ----------
        :type self: object
        :param lmbda : array_like of floats, optional
           Weights for each dimension of the Kruskal operator.
           ``len(lambda)`` must be equal to ``U[i].shape[1]``

        :param us : list of ndarrays
           Factor matrices from which the Tensor representation
           is created. All factor matrices ``U[i]`` must have the
           same number of columns, but can have different
           number of rows.
        ----------
        """
        if us is None:
            raise ValueError("Ktensor: first argument cannot be empty.")
        else:
            self.Us = np.array(us)
        self.shape = tuple(Ui.shape[0] for Ui in us)
        self.ndim = len(self.Us)
        self.rank = self.Us[0].shape[1]
        if lmbda is None:
            self.lmbda = np.ones(len(self.rank))
        else:
            self.lmbda = np.array(lmbda)
        if not all(np.array([Ui.shape[1] for Ui in us]) == self.rank):
            raise ValueError('Ktensor: dimension mismatch of factor matrices')

    def norm(self):
        """
        Efficient computation of the Frobenius norm for ktensors
        Returns: None
        -------
        norm : float
               Frobenius norm of the Ktensor
        """
        coefmatrix = np.dot(self.Us[0].T, self.Us[0])
        for i in range(1, self.ndim):
            coefmatrix = coefmatrix * np.dot(self.Us[i].T, self.Us[i])
        coefmatrix = np.dot(np.dot(self.lmbda.T, coefmatrix), self.lmbda)
        return np.sqrt(coefmatrix.sum())

    def tondarray(self):
        """
        Converts a Ktensor into a dense multidimensional ndarray

        Returns: None
        -------
        arr : np.ndarray
            Fully computed multidimensional array whose shape matches
            the original Ktensor.
        """
        a = np.dot(self.lmbda.T, pyten.tools.khatrirao(self.Us).T)
        return a.reshape(self.shape)

    def totensor(self):
        """
        Converts a Ktensor into a dense Tensor
        Returns
        -------
        arr : Tensor
            Fully computed multidimensional array whose shape matches
            the original Ktensor.
        """
        return pyten.tenclass.Tensor(self.tondarray())
