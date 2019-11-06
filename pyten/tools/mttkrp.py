from pyten.tools import khatrirao


def mttkrp(x, u, n):
    """
    Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
    :param x: a tensor X.
    :param u: a list of 2-D arrays
    :param n: the except dimension
    """

    dim = x.ndims
    temp1 = [n]
    temp2 = range(n)
    temp3 = range(n + 1, dim)
    temp2.reverse()
    temp3.reverse()
    temp1[len(temp1):len(temp1)] = temp3
    temp1[len(temp1):len(temp1)] = temp2
    xn = x.permute(temp1)
    xn = xn.tondarray()
    xn = xn.reshape([xn.shape[0], xn.size / xn.shape[0]])
    tempu = u[:]
    tempu.pop(n)
    tempu.reverse()
    return xn.dot(khatrirao(tempu))
