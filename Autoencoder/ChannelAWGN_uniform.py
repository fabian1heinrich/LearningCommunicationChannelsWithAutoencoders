import torch
import numpy as np
import torch.nn as nn

from torch import normal
from numpy import sqrt


class ChannelAWGN_uniform(nn.Module):

    def __init__(self, snr_low, snr_high):
        super(ChannelAWGN_uniform, self).__init__()

        self.snr_low = snr_low
        self.snr_high = snr_high

        self.mean = 0

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

        snr_uniform = np.random.uniform(self.snr_low, self.snr_high)
        std = 1 / (10 ** (snr_uniform / 10))

        noise = normal(self.mean, sqrt(std/2), x.size(), device=dev)
        x = x + noise
        return x
