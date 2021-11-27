import torch.nn as nn

from train_mine import train_mine


class MINE(nn.Module):

    def __init__(self, n_dim, layers):
        super(MINE, self).__init__()

        self.input1 = nn.Linear(n_dim, layers[0].in_features)

        self.input2 = nn.Linear(n_dim, layers[0].in_features)

        self.layers = layers

    def forward(self, x, y):
        h1 = self.input1(x) + self.input2(y)
        h2 = self.layers(h1)
        return h2

    def train_mine(self, samples, n_epochs, batch_size, lr, mi_true=None):
        return train_mine(self, samples, n_epochs, batch_size, lr, mi_true)
