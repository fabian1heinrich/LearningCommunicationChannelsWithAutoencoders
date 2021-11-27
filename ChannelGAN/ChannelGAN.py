from torch import randint, randn

from GAN import GAN
from Generator import Generator
from Discriminator import Discriminator


class ChannelGAN(GAN):

    def __init__(self, gen, dis, constellation, channel):
        super(ChannelGAN, self).__init__(gen, dis, constellation, channel)

        self.generator = Generator(gen)
        self.generator = self.generator.to(self.dev)

        self.discriminator = Discriminator(dis)
        self.discriminator = self.discriminator.to(self.dev)

        self.constellation = constellation.to(self.dev)

        self.channel = channel
        self.channel.to(self.dev)

    def gen_channel_input(self, batch_size):
        index = randint(0, len(self.constellation), (batch_size, ), device=self.dev)
        return self.constellation[index]

    def gen_noise(self, batch_size):
        return randn((batch_size, 20), device=self.dev)

    def gen_channel_output(self, batch_size):
        return self.channel(self.gen_channel_input(batch_size))
