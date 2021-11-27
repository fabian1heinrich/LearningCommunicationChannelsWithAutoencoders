import numpy as np

from torch import from_numpy, stack


def constellation_4QAM():

    c = np.array([+1, -1, +1j, -1j])

    c_real = np.real(c)
    c_imag = np.imag(c)

    const = stack((from_numpy(c_real), from_numpy(c_imag)), dim=1)

    return const.float()
