import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as D

from autoencoder_dataset import AutoencoderDataset
from autoencoder import Autoencoder
from torch import sum, mean, square, sqrt

# 16qam1
torch.manual_seed(11)
qam_coeffs = np.array([
    1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j,
    1 + 3j, 1 - 3j, -1 + 3j, -1 - 3j,
    3 + 1j, 3 - 1j, -3 + 1j, -3 - 1j,
    3 + 3j, 3 - 3j, -3 + 3j, -3 - 3j])

# 16qam2
# qam_coeffs = qam_coeffs * np.exp(1j * np.pi / 4)

# make coeffs suitable for ml
coeffs = qam_coeffs
coeffs = [np.real(coeffs), np.imag(coeffs)]
coeffs = np.transpose(coeffs)
coeffs = torch.from_numpy(coeffs)
coeffs = coeffs / sqrt(sum(mean(square(coeffs))))
coeffs = coeffs.float()


autoencoder = Autoencoder(16, 1, 5)
encoder = autoencoder.encoder
decoder = autoencoder.decoder
channel = autoencoder.channel

encoder_dataset = AutoencoderDataset(16, 16000)
train_loader = D.DataLoader(encoder_dataset, batch_size=1024, shuffle=True)


# train encoder
criterion1 = nn.MSELoss()
optimizer1 = torch.optim.Adam(encoder.parameters(), lr=0.01)
for epoch in range(20):
    loss = 0
    for batch_features, targets in train_loader:
        optimizer1.zero_grad()

        t = encoder(batch_features)

        train_loss1 = criterion1(t, coeffs[targets])

        train_loss1.backward()
        optimizer1.step()
        loss += train_loss1.item()

    loss = loss / len(train_loader)
    print("epoch #{}/{} | loss = {:.6f}".format(epoch + 1, 20, loss))


# train decoder
criterion2 = nn.CrossEntropyLoss()
optimizer2 = torch.optim.Adam(decoder.parameters(), lr=0.01)
for epoch in range(30):
    loss = 0
    for batch_features, targets in train_loader:
        optimizer2.zero_grad()

        t = encoder(batch_features)
        p = channel(decoder(t))
        train_loss2 = criterion2(p, targets)

        train_loss2.backward()
        optimizer2.step()
        loss += train_loss2.item()

    loss = loss / len(train_loader)
    print("epoch #{}/{} | loss = {:.6f}".format(epoch + 1, 30, loss))


pretrained_ae = Autoencoder(16, 1, 5)
pretrained_ae.encoder = encoder
pretrained_ae.decoder = decoder

path = '/home/fabian/universitaet/master_thesis/code/final/saved_models/16qam1'
torch.save(pretrained_ae.state_dict(), path)

# plots
pretrained_ae.scatter_plot()
pretrained_ae.ber_plot(10000)
plt.show()
