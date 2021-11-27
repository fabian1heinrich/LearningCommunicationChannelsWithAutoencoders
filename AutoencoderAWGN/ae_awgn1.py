# fig:ae_awgn_bler

import _paths_ae_awgn

import os
import torch
import numpy as np
import torch.nn as nn

from ChannelAutoencoder import ChannelAutoencoder
from ChannelAWGN import ChannelAWGN
from NormLayer import NormLayer

import matplotlib.pyplot as plt
from get_plot import get_plot

os.environ['CUDA_VISIBLE_DEVICES'] = ''
print(torch.cuda.is_available())
your_path = os.path.dirname(__file__)

torch.manual_seed(0)
M = 16
N_channel = 1

snr_train = np.arange(0, 25)
snr_bler = np.arange(0, 23)
snr_bler_plot = np.array([6, 9, 15, 18, 21])

bler = np.zeros((len(snr_bler), len(snr_train)))

for i, snr in enumerate(snr_train):

    encoder = nn.Sequential(
        nn.Linear(M, M),
        nn.Tanh(),
        nn.Linear(M, M),
        nn.Tanh(),
        nn.Linear(M, M),
        nn.Tanh(),
        nn.Linear(M, M),
        nn.Tanh(),
        nn.Linear(M, 2 * N_channel),
        NormLayer()
    )

    decoder = nn.Sequential(
        nn.Linear(2 * N_channel, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, M)
    )

    channel = ChannelAWGN(snr)

    ae = ChannelAutoencoder(encoder, channel, decoder)
    ae.init_weights_n()

    ae.train_ae(16000, 40, 1000, 0.01, True)
    ae.train_ae(16000, 20, 5000, 0.005, False)
    ae.train_ae(16000, 20, 5000, 0.001, False)

    # constellation plot
    fig1, ax1 = get_plot(0.3, 2)
    ax1.axis([-2, 2, -2, 2])
    plt.xticks([-2, -1, 0, 1, 2])
    plt.yticks([-2, -1, 0, 1, 2])
    constellation = ae.constellation()
    ax1.scatter(constellation[:, 0], constellation[:, 1])
    name1 = 'ae_awgn_constellation_M' + str(M) + '_snr' + str(snr) + '.pdf'
    print(name1)
    fig1.savefig(os.path.join(your_path, 'plots', name1), format='pdf', bbox_inches='tight')
    plt.clf()

    # calc bler
    bler[:, i] = ae.calc_ber_awgn(snr_bler)

# bler plot
fig2, ax2 = get_plot(1, 1)
ax2.set_yscale('log')
ax2.plot(snr_bler, bler[:,[6, 9, 15, 18, 21]], ':', linewidth=3)
plt.ylabel(r'$BLER$')
plt.xlabel(r'$SNR$')
plt.xticks(snr_bler[::2])
ax2.legend((r'$SNR=6$ dB', r'$SNR=9$ dB', r'$SNR=15$ dB', r'$SNR=18$ dB', r'$SNR=21$ dB'), loc='lower left')
name2 = 'ae_awgn_M' + str(16) + '_bler.pdf'
print(name2)
fig2.savefig(os.path.join(your_path, 'plots', name2), format='pdf', bbox_inches='tight')
plt.clf()

# bler plot opitmal snr
for i, cur_snr_bler in enumerate(snr_bler):

    fig3, ax3 = get_plot(0.4, 1)
    ax3.set_yscale('log')
    ax3.plot(snr_train, bler[i, :])
    y_label = r'$BLER$ at ' + str(cur_snr_bler) + r'$dB$'
    plt.ylabel(y_label)
    plt.xlabel(r'$SNR$ training')
    plt.xticks(snr_train[::6])
    name3 = 'ae_awgn_snr_train_vs_bler_at_' + str(cur_snr_bler) + '_log.pdf'
    fig3.savefig(os.path.join(your_path, 'plots', name3), format='pdf', bbox_inches='tight')
    plt.clf()

    fig4, ax4 = get_plot(0.4, 1)
    ax4.plot(snr_train, bler[i, :])
    y_label = r'$BLER$ at ' + str(cur_snr_bler) + r'$dB$'
    plt.ylabel(y_label)
    plt.xlabel(r'$SNR$ training')
    plt.xticks(snr_train[::6])
    name4 = 'ae_awgn_snr_train_vs_bler_at_' + str(cur_snr_bler) + '_lin.pdf'
    fig4.savefig(os.path.join(your_path, 'plots', name4), format='pdf', bbox_inches='tight')
    plt.clf()
