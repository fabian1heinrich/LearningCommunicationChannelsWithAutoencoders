# fig:ae_awgn_bler
import os

os.environ["CUDA_VISIBLE_DEVICES"]=""

import _paths_ae_awgn

import torch
import numpy as np
import torch.nn as nn

from ChannelAutoencoder import ChannelAutoencoder
from MINE import MINE
from ChannelAWGN import ChannelAWGN
from NormLayer import NormLayer


print(torch.cuda.is_available())
your_path = os.path.dirname(__file__)

torch.manual_seed(0)
M = 16
N_channel = 1

encoder = nn.Sequential(
    nn.Linear(M, M),
    nn.Tanh(),
    nn.Linear(M, M),
    nn.Tanh(),
    nn.Linear(M, M),
    nn.Tanh(),
    nn.Linear(M, M),
    nn.Tanh(),
    nn.Linear(M, 2 * N_channel),
    nn.Tanh(),
    NormLayer()
)

decoder = nn.Sequential(
    nn.Linear(2 * N_channel, 128),
    nn.Tanh(),
    nn.Linear(128, 128),
    nn.Tanh(),
    nn.Linear(128, 128),
    nn.Tanh(),
    nn.Linear(128, 128),
    nn.Tanh(),
    nn.Linear(128, 128),
    nn.Tanh(),
    nn.Linear(128, M)
)

layers_mine = nn.Sequential(
    nn.Linear(40, 40),
    nn.Tanh(),
    nn.Linear(40, 40),
    nn.Tanh(),
    nn.Linear(40, 1)
)


channel = ChannelAWGN(15)
mine = MINE(2, layers_mine)
ae = ChannelAutoencoder(encoder, channel, decoder)

ae.train_ae_mine_loss(mine, 1000, 0.01, 0.0005)
