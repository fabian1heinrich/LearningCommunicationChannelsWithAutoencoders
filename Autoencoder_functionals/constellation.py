import torch

from get_dev import get_dev


def constellation(autoencoder):

    dev = get_dev()

    encoder = autoencoder.encoder
    M = autoencoder.M

    x = torch.eye(M, dtype=torch.float)
    x = x.to(dev)
    x = encoder(x)
    x = x.to('cpu')
    x = x.detach().numpy()

    return x
