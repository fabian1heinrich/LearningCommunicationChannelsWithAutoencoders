import torch

from AutoencoderDataset import AutoencoderDataset
from torch.utils.data import DataLoader
from losses_autoencoder import cel_ae

x = torch.randn((16, 2))


def train_decoder(autoencoder, N, N_epochs, lr):

    autoencoder.train()

    M = autoencoder.M
    targets = torch.zeros(N, dtype=torch.int64).random_(M)

    optimizer = torch.optim.Adam(autoencoder.decoder.parameters(), lr=lr)

    for epoch in range(N_epochs):
        loss = 0
        for t in targets:
            optimizer.zero_grad()

            outputs = autoencoder.channel(x[t])

            train_loss = cel_ae(outputs, targets)

            train_loss.backward()
            optimizer.step()

            loss += train_loss.item()
        loss = loss / len(train_loader)
        print('DEC #{}/{} | loss = {:.6f}'.format(epoch + 1, n_epochs, loss))
