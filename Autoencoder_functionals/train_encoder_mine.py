import torch
import matplotlib.pyplot as plt

from AutoencoderDataset import AutoencoderDataset
from torch.utils.data import DataLoader
from torch import log, mean, exp, randperm


def train_encoder_mine(autoencoder, mine, N, N_epochs, lr_enc, lr_mine):

    autoencoder.train()
    mine.train()

    dataset = AutoencoderDataset(16, 16000)
    train_loader_mine = DataLoader(dataset, batch_size=2**12, shuffle=True)
    train_loader_ae = DataLoader(dataset, batch_size=2**10, shuffle=True)

    optimizer1 = torch.optim.Adam(mine.parameters(), lr=lr_mine)
    optimizer2 = torch.optim.Adam(autoencoder.encoder.parameters(), lr=lr_enc)
    for epoch in range(N_epochs):
        for samples, targets in train_loader_mine:
            optimizer1.zero_grad()

            x = autoencoder.encoder(samples)
            y = autoencoder.channel(x)
            y_s = y[randperm(len(y))]

            est1 = mine(x, y)
            est2 = mine(x, y_s)
            loss1 = -(mean(est1) - log(mean(exp(est2))))

            loss1.backward()
            optimizer1.step()

        for samples, targets in train_loader_ae:
            optimizer2.zero_grad()

            x = autoencoder.encoder(samples)
            y = autoencoder.channel(x)
            y_s = y[randperm(len(y))]

            est1 = mine(x, y)
            est2 = mine(x, y_s)
            loss2 = -(mean(est1) - log(mean(exp(est2))))

            loss2.backward()
            optimizer2.step()

        ax = plt.gca()
        autoencoder.scatter_plot(ax)
        plt.axis([-2, 2, -2, 2])
        plt.pause(0.05)
        ax.grid()
        ax.clear()
        print('AE #{}/{} | loss = {:.6f}'.format(epoch + 1, N_epochs, loss2))
