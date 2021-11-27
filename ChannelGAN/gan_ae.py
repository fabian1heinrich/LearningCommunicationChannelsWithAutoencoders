import _paths_channelgan

import os
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from Transmission import Transmission
from TxParameters import TxParameters
from ChannelGAN import ChannelGAN
from constellation_16QAM import constellation_16QAM
from get_plot import get_plot
from ChannelAutoencoder import ChannelAutoencoder
from Transmission import Transmission
from NormLayer import NormLayer
from TxParameters import TxParameters
from ChannelAWGN import ChannelAWGN
from ChannelGAN import ChannelGAN
from AutoencoderGAN import AutoencoderGAN

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

channel = ChannelAWGN(12)
ae = ChannelAutoencoder(encoder, channel, decoder)
gan = AutoencoderGAN()
ae.train_ae_gan(gan)

ae.scatter_plot()
plt.show()

a = 1
