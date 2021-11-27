import _paths_ae

import torch.nn as nn

from numpy import log2

from get_dev import get_dev

from train_ae import train_ae
from calc_ber_awgn import calc_ber_awgn
from calc_ber_hpa import calc_ber_hpa
from plot_ber import plot_ber
from scatter_plot import scatter_plot
from pretrain_ae import pretrain_ae
# from train_decoder import train_decoder
from train_encoder_mine import train_encoder_mine
from init_weights import init_weights_n, init_weights_u
from constellation import constellation
from train_ae_gan import train_ae_gan
from train_ae_mine_separated import train_ae_mine_separated
from train_ae_mine_loss import train_ae_mine_loss


class Autoencoder(nn.Module):
    def __init__(self, *args):
        super(Autoencoder, self).__init__()

    # init
    def init_weights_n(self):
        init_weights_n(self)

    def init_weights_u(self):
        init_weights_u(self)

    # train
    def pretrain_ae(self):
        pretrain_ae(self)

    def train_ae(self, N, N_epochs, batch_size, lr, plot):
        train_ae(self, N, N_epochs, batch_size, lr, plot)

    def train_ae_gan(self, gan):
        train_ae_gan(self, gan)

    def train_ae_mine_separated(self, mine, batch_size, lr_ae, lr_mine):
        train_ae_mine_separated(self, mine, batch_size, lr_ae, lr_mine)

    def train_ae_mine_loss(self, mine, batch_size, lr_ae, lr_mine):
        train_ae_mine_loss(self, mine, batch_size, lr_ae, lr_mine)

    # def train_decoder(self, N, N_epochs, lr):
    #     train_decoder(self, N, N_epochs, lr)

    def train_encoder_mine(self, mine, N, N_epochs, lr_enc, lr_mine):
        train_encoder_mine(self, mine, N, N_epochs, lr_enc, lr_mine)

    # constellation
    def constellation(self):
        x = constellation(self)
        return x

    # test
    def calc_ber_awgn(self, eb_no):
        ber = calc_ber_awgn(self, eb_no)
        return ber

    def calc_ber_hpa(self, eb_no, tx_parameters):
        ber = calc_ber_hpa(self, eb_no, tx_parameters)
        return ber

    # plots
    def plot_ber(self, eb_no, ax=None, c=None):
        plot_ber(self, eb_no, ax, c)

    def scatter_plot(self, ax=None):
        scatter_plot(self, ax)
