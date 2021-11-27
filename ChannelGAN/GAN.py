import torch.nn as nn

from get_dev import get_dev

from train_channel_gan import train_channel_gan
from plot_gan_noise import plot_gan_noise
from plot_gan_dist import plot_gan_dist
from plot_gan_gen import plot_gan_gen
from plot_gan import plot_gan


class GAN(nn.Module):

    def __init__(self, *args):
        super(GAN, self).__init__()

        self.dev = get_dev()

    # functionals
    def train_channel_gan(self, batch_size, n_epochs):
        train_channel_gan(self, batch_size, n_epochs)

    def plot_gan_noise(self, ax=None):
        plot_gan_noise(self, ax)

    def plot_gan_dist(self, ax=None):
        plot_gan_dist(self, ax)

    def plot_gan_gen(self, ax=None):
        plot_gan_gen(self, ax)

    def plot_gan(self):
        plot_gan(self)
