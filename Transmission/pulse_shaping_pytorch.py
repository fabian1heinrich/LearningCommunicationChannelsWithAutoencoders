import torch

from torch.nn.functional import conv1d
from numpy import floor, ceil

from filter_pytorch import filter_pytorch


def pulse_shaping_pytorch(tx_symbols, filter):

    if tx_symbols.is_cuda:
        dev = 'cuda:0'
    else:
        dev = 'cpu'

    tx_signal = torch.zeros((tx_symbols.shape), device=dev)

    _, dim = tx_symbols.shape

    for i in range(dim):

        cur_tx_symbols = tx_symbols[:, i]

        tx_signal[:, i] = filter_pytorch(cur_tx_symbols, filter)

    return tx_signal

    # for i in range(dim):
    #     # pad for conv pytorch implementation
    #     x_pytorch = torch.zeros((1, 1, len(x)), device=dev)
    #     x_pytorch[0, 0, :] = x[:, i]
    #     pad = torch.zeros((1, 1, length_filter), requires_grad=True, device=dev)
    #     x_padded = torch.cat((pad, x_pytorch, pad), dim=2)

    #     y = conv1d(x_padded, filter_pytorch)
    #     y = y[0, 0, :]
    #     index1 = int(floor(length_filter / 2))
    #     index2 = len(y) - index1 - 1
    #     # index2 = int(len(y)-ceil(length_filter / 2)-1)
    #     y = y[index1:index2]

    #     signal[:, i] = y

    # return signal
