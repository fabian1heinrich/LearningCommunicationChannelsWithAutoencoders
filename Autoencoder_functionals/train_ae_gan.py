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


def train_ae_gan(autoencoder, gan):

    autoencoder.train()
    gan.generator.train()
    gan.discriminator.train()

    opt_g = torch.optim.Adam(gan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_ae = torch.optim.Adam(list(autoencoder.encoder.parameters()) + list(autoencoder.decoder.parameters()), lr=0.01, amsgrad=True)

    criterion_gan = nn.BCELoss()
    criterion_ae = nn.CrossEntropyLoss()

    fake_labels = zeros((5000, 1), device=gan.dev)
    real_labels = ones((5000, 1), device=gan.dev)

    m = autoencoder.M
    dataset = AutoencoderDataset(m, 16000)
    train_loader = DataLoader(dataset, batch_size=5000, drop_last=True)

    fig1, ax1 = plt.subplots()
    ax1.grid()
    ax1.set_xlim([-2, 2])
    ax1.set_ylim([-2, 2])
    ax1.set_aspect('equal', adjustable='box')

    fig2, ax2 = plt.subplots()

    # pretrain ae on awgn channel
    autoencoder.train_ae(16000, 40, 1000, 0.01, True)

    for epoch in range(100):

        loss_ae = 0
        loss_g = 0
        loss_d = 0

        for batch_features, targets in train_loader:

            opt_ae.zero_grad()
            x = autoencoder.encoder(batch_features)
            x_gan = x.detach()
            # gan needs to be trained more intense since it 'learns' slower
            for i in range(0, 5000):

                # train discriminator
                opt_d.zero_grad()
                y_real = autoencoder.channel(x_gan)
                y_real = y_real.detach()
                pred_y_real = gan.discriminator(y_real)
                loss_d_real = criterion_gan(pred_y_real, real_labels)

                noise = gan.gen_noise(5000)
                with torch.no_grad():
                    y_fake = gan.generator(x, noise)
                pred_y_fake = gan.discriminator(y_fake)
                loss_d_fake = criterion_gan(pred_y_fake, fake_labels)

                loss_d = (loss_d_real + loss_d_fake) / 2
                loss_d.backward()
                opt_d.step()

                # train generator
                opt_g.zero_grad()
                noise = gan.gen_noise(5000)
                y_hat1 = gan.generator(x_gan, noise)
                pred_y_hat = gan.discriminator(y_hat1)
                loss_g = criterion_gan(pred_y_hat, real_labels)

                loss_g.backward()
                opt_g.step()
                y_hat2 = gan.generator(x, noise)

                if i % 100 == 0:
                    y_hat1 = y_hat1.cpu()
                    gen_plot = y_hat1.detach().numpy()
                    ax2.hist2d(gen_plot[:, 0], gen_plot[:, 1], bins=100, cmap='plasma', range = np.array([[-2, 2], [-2, 2]]))
                    plt.pause(0.05)
                    ax2.clear()

            s_hat = autoencoder.decoder(y_hat2)
            loss_ae = criterion_ae(s_hat, targets)
            loss_ae.backward()
            opt_ae.step()

            print(epoch)
            ax1.clear()
            ax1.set_xlim([-2, 2])
            ax1.set_ylim([-2, 2])
            autoencoder.scatter_plot(ax1)
            ax1.grid()
            # plt.axis([-3, 3, -3, 3])
            ax1.set_aspect('equal', adjustable='box')
            plt.pause(0.05)
