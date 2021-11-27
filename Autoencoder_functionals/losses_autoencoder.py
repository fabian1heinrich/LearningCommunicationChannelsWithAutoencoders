import torch.nn as nn

from torch import arange, zeros


def mse_ae(ouputs, targets, m):

    t = zeros((len(targets), m))
    t[arange(len(targets)), targets] = 1

    softmax = nn.Softmax()
    o = softmax(ouputs)

    # loss = nn.MSELoss(o, t)
    criterion = nn.MSELoss()
    loss = criterion(o, t)

    return loss


def cel_ae(outputs, targets):

    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)

    return loss
