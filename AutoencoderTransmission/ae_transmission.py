# fig:ae_awgn_bler

import _paths_ae_transmission

import os
import torch
import numpy as np
import torch.nn as nn

from ChannelAutoencoder import ChannelAutoencoder
from Transmission import Transmission
from NormLayer import NormLayer
from TxParameters import TxParameters

import matplotlib.pyplot as plt
from get_plot import get_plot

# os.environ['CUDA_VISIBLE_DEVICES'] = ''
print(torch.cuda.is_available())
your_path = os.path.dirname(__file__)

torch.manual_seed(0)
M = 16
N_channel = 1

snr = 0
input_backoff = 10
amam_params = [2.0, 1.0]
ampm_params = [np.pi / 3, 1.0]
rolloff = 0.3
samples_per_symbol = 8
n_symbols_pulse_shaping = 16

tx_parameters = TxParameters(
        snr,
        input_backoff,
        amam_params,
        ampm_params,
        rolloff,
        samples_per_symbol,
        n_symbols_pulse_shaping)

snr_train = np.array([24])
snr_bler = np.arange(0, 22)

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

    tx_parameters.snr = snr
    channel = Transmission(tx_parameters)

    ae = ChannelAutoencoder(encoder, channel, decoder)
    ae.init_weights_n()

    ae.train_ae(16000, 40, 1000, 0.01, False)
    ae.train_ae(16000, 20, 5000, 0.005, False)
    ae.train_ae(16000, 20, 5000, 0.001, False)

    # constellation plot
    fig1, ax1 = get_plot(0.3, 2)
    ax1.axis([-2, 2, -2, 2])
    plt.xticks([-2, -1, 0, 1, 2])
    plt.yticks([-2, -1, 0, 1, 2])
    constellation = ae.constellation()
    ax1.scatter(constellation[:, 0], constellation[:, 1])
    name1 = 'ae_hpa_constellation_M' + str(M) + '_snr' + str(snr) + 'ip10.pdf'
    print(name1)
    fig1.savefig(os.path.join(your_path, 'plots', name1), format='pdf', bbox_inches='tight')
    plt.clf()

    fig2, ax2 = get_plot(0.3, 2)
    ax2.axis([-3, 3, -3, 3])
    plt.xticks([-3, -2, -1, 0, 1, 2, 3])
    plt.yticks([-3, -2, -1, 0, 1, 2, 3])
    ax2.scatter(constellation[:, 0], constellation[:, 1], s=10)
    tx_parameters.snr = 100
    channel = Transmission(tx_parameters)
    constellation_ac = channel(torch.from_numpy(constellation))
    constellation_ac = constellation_ac.detach().numpy()
    ax2.scatter(constellation_ac[:, 0], constellation_ac[:, 1])
    name2 = 'ae_hpa_constellation_M' + str(M) + '_snr' + str(snr) + '_ac_ibp10.pdf'
    print(name2)
    fig2.savefig(os.path.join(your_path, 'plots', name2), format='pdf', bbox_inches='tight')
    plt.clf()

    # calc bler
    # bler[:, i] = ae.calc_ber_hpa(snr_bler, tx_parameters)

# bler plot
fig3, ax3 = get_plot(1, 1)
ax3.set_yscale('log')
ax3.plot(snr_bler, bler[:,[4, 9, 11, 15, 18]], ':', linewidth=3)
plt.ylabel(r'$BLER$')
plt.xlabel(r'$SNR$')
plt.xticks(snr_bler[::2])
ax3.legend((r'$SNR=4$ dB', r'$SNR=9$ dB', r'$SNR=11$ dB', r'$SNR=15$ dB', r'$SNR=18$ dB'), loc='lower left')
name3 = 'ae_hpa_M' + str(M) + '_bler.pdf'
print(name3)
fig3.savefig(os.path.join(your_path, 'plots', name3), format='pdf', bbox_inches='tight')
plt.clf()

# bler plot opitmal snr
for i, cur_snr_bler in enumerate(snr_bler):

    fig4, ax4 = get_plot(0.4, 1)
    ax4.set_yscale('log')
    ax4.plot(snr_train, bler[i, :])
    y_label = r'$BLER$ at ' + str(cur_snr_bler) + r'$dB$'
    plt.ylabel(y_label)
    plt.xlabel(r'$SNR$ training')
    plt.xticks(snr_train[::6])
    name4 = 'ae_hpa_snr_train_vs_bler_at_' + str(cur_snr_bler) + '_log.pdf'
    fig4.savefig(os.path.join(your_path, 'plots', name4), format='pdf', bbox_inches='tight')
    plt.clf()

    fig5, ax5 = get_plot(0.4, 1)
    ax5.plot(snr_train, bler[i, :])
    y_label = r'$BLER$ at ' + str(cur_snr_bler) + r'$dB$'
    plt.ylabel(y_label)
    plt.xlabel(r'$SNR$ training')
    plt.xticks(snr_train[::6])
    name5 = 'ae_hpa_snr_train_vs_bler_at_' + str(cur_snr_bler) + '_lin.pdf'
    fig5.savefig(os.path.join(your_path, 'plots', name5), format='pdf', bbox_inches='tight')
    plt.clf()
