# fig:ae_awgn_bler

import _paths_ae_awgn

import os
import torch
import numpy as np
import torch.nn as nn

from ChannelAutoencoder import ChannelAutoencoder
from ChannelGAN import ChannelGAN
from ChannelAWGN import ChannelAWGN
from NormLayer import NormLayer


# os.environ['CUDA_VISIBLE_DEVICES'] = ''
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

discriminator = nn.Sequential(
    nn.Linear(2, 80),
    nn.LeakyReLU(),
    nn.Linear(80, 80),
    nn.LeakyReLU(),
    nn.Linear(80, 80),
    nn.LeakyReLU(),
    nn.Linear(80, 80),
    nn.LeakyReLU(),
    nn.Linear(80, 1),
    nn.Sigmoid()
)

generator = nn.Sequential(
    nn.Linear(22, 40),
    nn.Dropout(0.2),
    nn.Linear(40, 80),
    nn.LeakyReLU(),
    nn.Dropout(0.2),
    nn.Linear(80, 80),
    nn.LeakyReLU(),
    nn.Linear(80, 2)
)

channel = ChannelAWGN(15)
ae = ChannelAutoencoder(encoder, channel, decoder)
constellation = ae.constellation()
gan = ChannelGAN(generator, discriminator, torch.from_numpy(constellation), channel)

ae.train_ae_gan(gan)