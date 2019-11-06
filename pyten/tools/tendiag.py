import numpy as np
import pyten.tenclass
import numpy.matlib


def tendiag(v, sz):
    """
    Create a Diagonal Tensor of Size 'sz' with Diagnal Values 'v'
    :param v: a list/array which defines the size of diagonal tensor
    :param sz: a list/array which defines the size of diagonal tensor
    """

    v = np.array(v)
    n = v.size
    v = v.reshape((n, 1))
    x = np.zeros(sz)
    subs = np.matlib.repmat(np.array(range(n)).reshape(n, 1), 1, len(sz))

    for i in range(n):
        x[subs[i][0], subs[i][1], subs[i][2]] = v[i]

    return pyten.tenclass.Tensor(x)
