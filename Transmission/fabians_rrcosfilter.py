import numpy as np

from torch import from_numpy
from numpy import pi, abs, sqrt, sin, cos


def fabians_rrcosfilter(beta, span, sps):

    # init
    filter_order = span * sps
    delay = int(filter_order / 2)
    t = np.linspace(-delay, delay, filter_order + 1, endpoint=True) / sps
    t = t[..., np.newaxis]
    b = np.zeros((filter_order + 1, 1))

    # calc values
    index1 = delay
    # b[index1] = -1 / (pi * sps) * (pi * (beta - 1) - 4 * beta)
    b[index1] = rrcosfilter1(beta, sps)

    eps = 7./3 - 4./3 - 1
    index2 = np.where(abs(abs(4 * beta * t) - 1.0) < sqrt(eps))[0]
    b[index2] = rrcosfilter2(beta, sps)

    index3 = np.arange(0, len(t))
    index3 = np.delete(index3, np.append(index2, index1))
    b[index3] = rrcosfilter3(beta, sps, t[index3])

    # normalize
    b = b / sqrt(sum(b ** 2))
    return from_numpy(b)


def rrcosfilter1(beta, sps):

    return -1 / (pi * sps) * (pi * (beta - 1) - 4 * beta)


def rrcosfilter2(beta, sps):

    r2 = 1 / (2 * pi * sps) * (pi * (beta + 1) * sin(pi * (beta + 1) / (4 * beta)) - 4 * beta * sin(pi * (beta - 1) / (4 * beta)) + pi * (beta - 1) * cos(pi * (beta - 1) / (4 * beta)))

    return r2


def rrcosfilter3(beta, sps, nind):

    beta1 = 1 + beta
    beta2 = 1 - beta
    beta4 = 4 * beta

    r3 = - beta4 / sps * (cos((beta1) * pi * nind) + sin((beta2) * pi * nind) / (4 * beta * nind)) / (pi * ((beta4 * nind) ** 2 - 1))

    return r3
