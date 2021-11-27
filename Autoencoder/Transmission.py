import _paths_ae

import torch.nn as nn

from get_dev import get_dev
from transmission_pytorch import transmission_pytorch


class Transmission(nn.Module):

    def __init__(self, tx_parameters):
        super(Transmission, self).__init__()

        self.tx_parameters = tx_parameters

        self.dev = get_dev()

        self.snr = self.tx_parameters.snr

    def forward(self, x):
        x = transmission_pytorch(x, self.tx_parameters)
        return x
