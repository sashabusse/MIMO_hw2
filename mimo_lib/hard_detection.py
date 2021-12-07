import numpy as np
from numba import jit


@jit(nopython=True)
def hard_detection(s, mod):
    res = 0

    if mod == "BPSK":
        res = np.where(np.real(s) >= 0, 1., -1.)

    elif mod == "QPSK":
        res = np.zeros(2*len(s))
        res[::2] = np.where(np.real(s) >= 0, 1., -1.)
        res[1::2] = np.where(np.imag(s) >= 0, 1., -1.)

    else:
        assert False, "wrong mode"

    return res

