import torch
from numpy import log2, log10

from Autoencoder import Autoencoder


class ChannelAutoencoder(Autoencoder):
    def __init__(self, encoder, channel, decoder):
        super(Autoencoder, self).__init__()

        self.encoder = encoder

        self.channel = channel

        self.decoder = decoder

        if torch.cuda.is_available():
            dev = 'cuda:0'
        else:
            dev = 'cpu'
        self.to(dev)

        self.M = encoder[0].in_features

        self.N_channel = int(decoder[0].in_features / 2)

        # self.parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        # self.optimizer = torch.optim.Adam(self.parameters, lr=0.01, amsgrad=True)


        # self.k = log2(self.M)
        # self.channel.snr = channel.ebno + 10 * log10(self.k)

        # self.R = self.k * self.N_channel

    def forward(self, x):
        x = self.encoder(x)
        x = self.channel(x)
        x = self.decoder(x)
        return x
