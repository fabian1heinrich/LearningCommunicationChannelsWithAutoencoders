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

snr_train = np.array([3, 6, 12, 15, 18, 24])
snr_bler = np.arange(0, 23)

bler = np.zeros((len(snr_bler), len(snr_train)))

snr = 0
input_backoff = 0
amam_params = [2.0, 1.0]
ampm_params = [np.pi / 3, 1.0]
rolloff = 0.3
samples_per_symbol = 4
n_symbols_pulse_shaping = 16

tx_parameters = TxParameters(
        snr,
        input_backoff,
        amam_params,
        ampm_params,
        rolloff,
        samples_per_symbol,
        n_symbols_pulse_shaping)

for i, snr in enumerate(snr_train):

    encoder = nn.Sequential(
        nn.Linear(M, M),
        nn.Tanh(),
        nn.Linear(M, int(2*M)),
        nn.Tanh(),
        nn.Linear(int(2*M), int(2*M)),
        nn.Tanh(),
        nn.Linear(int(2*M), 2 * N_channel),
        nn.Tanh(),
        # nn.BatchNorm1d(2 * N_channel),
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

    bler[:, i] = ae.calc_ber_hpa(snr_bler, tx_parameters)

# bler plot
fig1, ax1 = get_plot(1, 1)
ax1.set_yscale('log')
ax1.plot(snr_bler, bler, ':', linewidth=3)
plt.ylabel(r'BLER')
plt.xlabel(r'$SNR$')
plt.xticks(snr_bler[::2])
ax1.legend((r'$SNR=3$ dB', r'$SNR=6$ dB', r'$SNR=12$ dB', r'$SNR=15$ dB', r'$SNR=18$ dB', r'$SNR=24$ dB'), loc='lower left')
name1 = 'ae_hpa_M' + str(16) + '_bler.pdf'
print(name1)
fig1.savefig(os.path.join(your_path, 'plots', name1), format='pdf', bbox_inches='tight')
