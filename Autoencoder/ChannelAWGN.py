import torch
import torch
import torch.nn as nn

from torch import normal
from numpy import sqrt


class ChannelAWGN(nn.Module):

    def __init__(self, snr):
        super(ChannelAWGN, self).__init__()

        self.snr = snr

        self.mean = 0

        self.std = 1 / (10 ** (self.snr / 10))

        if torch.cuda.is_available():
            self.dev = 'cuda:0'
        else:
            self.dev = 'cpu'
        self.to(self.dev)

    def forward(self, x):

        if x.is_cuda:
            dev = 'cuda:0'
        else:
            dev = 'cpu'

        noise = normal(self.mean, sqrt(self.std/2), x.size(), device=dev)
        x = x + noise
        return x
