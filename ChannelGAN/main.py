import _paths_channelgan

import torch.nn as nn
from constellation_4QAM import constellation_4QAM
from ChannelGAN import ChannelGAN
from ChannelAWGN import ChannelAWGN

constellation = constellation_4QAM()

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

channel = ChannelAWGN(12)

gan = ChannelGAN(generator, discriminator, constellation, channel)

gan.train_channel_gan(1000, 10000)

b = gan.gen_channel_output(100)

a = 1