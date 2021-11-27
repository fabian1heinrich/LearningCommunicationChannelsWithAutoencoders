import _paths_ae_functionals

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from AutoencoderDataset import AutoencoderDataset
from losses_autoencoder import mse_ae, cel_ae
from torch import zeros, ones
from constellation_16QAM import constellation_16QAM


def train_ae_mine_separated(autoencoder, mine, batch_size, lr_ae, lr_mine):

    autoencoder.train()
    mine.train()

    m = autoencoder.M
    dataset = AutoencoderDataset(m, 16000)
    train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

    opt_encoder = torch.optim.Adam(autoencoder.encoder.parameters(), lr=lr_ae, amsgrad=True)
    opt_decoder = torch.optim.Adam(autoencoder.decoder.parameters(), lr=lr_ae, amsgrad=True)

    criterion_decoder = nn.MSELoss()

    for epoch in range(100):

        loss_encoder = 0

        for batch_features, targets in train_loader:

            opt_encoder.zero_grad()
            opt_decoder.zero_grad()

            x = autoencoder.encoder(batch_features)
            y = autoencoder.channel(x)

            # mine opt for encoder
            samples = torch.stack((x,y), dim=0)
            loss_encoder = mine.train_mine(samples, 100, batch_size, lr_mine)
            loss_encoder.backward()
            opt_encoder.step()

            # separated opt for decoder
            s_hat = autoencoder.decoder(y)
            loss_decoder = criterion_decoder(nn.Softmax(s_hat), targets)
            loss_decoder.backward()
            opt_decoder.step()
