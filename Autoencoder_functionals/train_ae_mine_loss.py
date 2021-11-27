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


def train_ae_mine_loss(autoencoder, mine, batch_size, lr_ae, lr_mine):

    autoencoder.train()
    mine.train()

    m = autoencoder.M
    dataset = AutoencoderDataset(m, 16000)
    train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

    opt_encoder = torch.optim.Adam(autoencoder.parameters(), lr=lr_ae, amsgrad=True)

    for epoch in range(100):


        for batch_features, targets in train_loader:

            opt_encoder.zero_grad()
            x = autoencoder.encoder(batch_features)
            x_mine = x.detach()
            y = autoencoder.channel(x)
            y_mine = y.detach()

            # for i in range(0, 10):
            samples = torch.stack((x_mine, y_mine), dim=0)
            mine.train_mine(samples, 50, batch_size, lr_mine)


        if epoch % 10 == 0:
            autoencoder.scatter_plot()
            plt.show()




