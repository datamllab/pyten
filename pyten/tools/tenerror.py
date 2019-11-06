import numpy as np
import pyten.tenclass


def tenerror(fitx, realx, omega):
    """
    Calculate Three Kinds of Error
    :param fitx: fitted tensor
    :param realx: ground-truth tensor
    :param omega: index tensor of observed entries
    """

    if type(omega) != np.ndarray and type(omega) != pyten.tenclass.Tensor:
        raise ValueError("AirCP: cannot recognize the format of observed Tensor!")
    elif type(omega) == pyten.tenclass.Tensor:
        omega = omega.tondarray
    if type(realx) == np.ndarray:
        realx = pyten.tenclass.Tensor(realx)
    if type(fitx) == np.ndarray:
        realx = pyten.tenclass.Tensor(realx)
    norm1 = np.linalg.norm(realx.data)
    norm2 = np.linalg.norm(realx.data * (1 - omega))
    err1 = np.linalg.norm(fitx.data - realx.data)  # Absolute Error
    err2 = np.linalg.norm((fitx.data - realx.data) * (1 - omega))
    re_err1 = err1 / norm1  # Relative Error 1
    re_err2 = err2 / norm2  # Relative Error 2
    return err1, re_err1, re_err2
