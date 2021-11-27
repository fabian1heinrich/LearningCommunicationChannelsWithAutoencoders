import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, layers):
        super(Discriminator, self).__init__()

        self.layers = layers

    def forward(self, x):
        return self.layers(x)
