import _paths_ae

import torch.nn as nn

from get_dev import get_dev
from saleh_pytorch import saleh_pytorch


class ChannelSaleh(nn.Module):

    # only for 1 channel use
    def __init__(self, amam_params, ampm_params):
        super(ChannelSaleh, self).__init__()

        self.amam_params = amam_params
        self.ampm_params = ampm_params

        self.dev = get_dev()

    def forward(self, x):
        x = saleh_pytorch(x, self.amam_params, self.ampm_params)
        return x
