import _paths_ae_functionals

import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from AutoencoderDataset import AutoencoderDataset
from losses_autoencoder import mse_ae, cel_ae


def train_ae(autoencoder, N, N_epochs, batch_size, lr, plot):

    autoencoder.train()

    m = autoencoder.M
    dataset = AutoencoderDataset(m, N)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    if plot:
        # plots
        fig, ax = plt.subplots()
        ax.grid()
        plt.axis([-3, 3, -3, 3])
        ax.set_aspect('equal', adjustable='box')

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr, amsgrad=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)

    for epoch in range(N_epochs):
        loss = 0
        for batch_features, targets in train_loader:
            optimizer.zero_grad()

            outputs = autoencoder(batch_features)

            # train_loss = mse_ae(outputs, targets, m)
            train_loss = cel_ae(outputs, targets)

            train_loss.backward()
            optimizer.step()
            # scheduler.step()
            loss += train_loss.item()
        loss = loss / len(train_loader)
        # print(scheduler.get_last_lr())
        # for param_group in optimizer.param_groups:
        #     print(param_group['lr'])

        # print('AE #{}/{} | loss = {:.6f}'.format(epoch + 1, N_epochs, loss))

        if plot:
            autoencoder.scatter_plot(ax)
            ax.grid()
            plt.axis([-3, 3, -3, 3])
            ax.set_aspect('equal', adjustable='box')
            plt.pause(0.05)
            ax.clear()

    if plot:
        autoencoder.scatter_plot(ax)
        ax.grid()
        plt.axis([-3, 3, -3, 3])
        ax.set_aspect('equal', adjustable='box')
