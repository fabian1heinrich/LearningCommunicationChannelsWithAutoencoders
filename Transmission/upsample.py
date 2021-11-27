import torch


def upsample(x, r):
    # torch implementation
    m = len(x)

    if x.is_cuda:
        dev = 'cuda:0'
    else:
        dev = 'cpu'

    d = torch.zeros((m * r, 2), device=dev)
    d[::r] = x

    return d
