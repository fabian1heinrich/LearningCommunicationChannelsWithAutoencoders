import torch.nn as nn

from torch import cat


class Generator(nn.Module):

    def __init__(self, layers):
        super(Generator, self).__init__()

        self.layers = layers

    def forward(self, x, noise):
        return self.layers(cat((x, noise), dim=1))
