import torch
import torch.nn as nn

from GAN import GAN


class AutoencoderGAN():

    def __init__(self):
        super(AutoencoderGAN, self).__init__()

        # input dim must be even

        self.generator = Generator()

        self.generator.to('cuda:0')

        self.discriminator = nn.Sequential(
            nn.Linear(2, 80),
            nn.LeakyReLU(),
            nn.Linear(80, 80),
            nn.LeakyReLU(),
            nn.Linear(80, 80),
            nn.LeakyReLU(),
            nn.Linear(80, 80),
            nn.LeakyReLU(),
            nn.Linear(80, 1),
            nn.Sigmoid()
        )
        self.discriminator.to('cuda:0')

    def gen_noise(self, batch_size):
        noise = torch.randn((batch_size, 20), device='cuda:0')
        return noise


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(22, 40),
            nn.Dropout(0.2),
            nn.Linear(40, 80),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(80, 80),
            nn.LeakyReLU(),
            nn.Linear(80, 2)
        )

    def forward(self, x, noise):
        h = torch.cat((x, noise), dim=1)
        return self.layers(h)
