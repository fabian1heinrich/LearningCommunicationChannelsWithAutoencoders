import numpy as np

from torch import from_numpy, stack


def constellation_16QAM():

    c = np.array([
        1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j,
        1 + 3j, 1 - 3j, -1 + 3j, -1 - 3j,
        3 + 1j, 3 - 1j, -3 + 1j, -3 - 1j,
        3 + 3j, 3 - 3j, -3 + 3j, -3 - 3j])
    c = c / np.sqrt(10)

    c_real = np.real(c)
    c_imag = np.imag(c)

    const = stack((from_numpy(c_real), from_numpy(c_imag)), dim=1)

    return const.float()
