import torch
from torch import sum, mean, square

from get_dev import get_dev


def get_energy(encoder, M):

    dev = get_dev()

    o = torch.eye(M, dtype=torch.float)
    o = o.to(dev)
    o = encoder(o)
    e = sum(mean(square(o)))
    return e
