import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as D

from torch import sum, mean, square, sqrt
from AutoencoderDataset import AutoencoderDataset


def pretrain_ae(autoencoder):

    M = autoencoder.M

    x = np.linspace(0, 2 * np.pi, M + 1)
    c = np.exp(x[:-1] * 1j)
    c = [np.real(c), np.imag(c)]
    c = np.transpose(c)
    c = torch.from_numpy(c)
    c = c / sqrt(sum(mean(square(c))))
    c = c.float()

    dataset = AutoencoderDataset(M, 16000)
    train_loader = D.DataLoader(dataset, batch_size=1024, shuffle=True)

    criterion1 = nn.MSELoss()
    optimizer1 = torch.optim.Adam(autoencoder.encoder.parameters(), lr=0.01)
    for epoch in range(20):
        for batch_features, targets in train_loader:
            optimizer1.zero_grad()
            t = autoencoder.encoder(batch_features)
            train_loss1 = criterion1(t, c[targets])
            train_loss1.backward()
            optimizer1.step()

    criterion2 = nn.CrossEntropyLoss()
    optimizer2 = torch.optim.Adam(autoencoder.decoder.parameters(), lr=0.01)
    for epoch in range(30):
        for batch_features, targets in train_loader:
            optimizer2.zero_grad()
            t = autoencoder.encoder(batch_features)
            p = autoencoder.decoder(t)
            # p = channel(decoder(t))
            train_loss2 = criterion2(p, targets)
            train_loss2.backward()
            optimizer2.step()
