import numpy as np


def khatrirao(u):
    """
    Calculate The Khatrirao Product
    :param u: a list of 2-D arrays
    """

    r = u[0].shape[1]
    k = []
    n = len(u)
    for j in range(r):
        temp = 1
        for i in range(n):
            temp1 = np.outer(temp, u[i][:, j])
            temp = temp1.reshape([1, temp1.size])
        k = np.append(k, temp)
    k = (k.reshape([r, len(k) / r])).transpose()
    return k
