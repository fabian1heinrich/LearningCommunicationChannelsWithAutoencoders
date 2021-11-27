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

snr_train = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24])

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
        nn.Tanh()
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
    name1 = 'ae_hpa_constellation_M' + str(M) + '_snr' + str(snr) + '.pdf'
    print(name1)
    fig1.savefig(os.path.join(your_path, 'plots', name1), format='pdf', bbox_inches='tight')


    fig2, ax2 = get_plot(0.3, 2)
    ax2.axis([-2, 2, -2, 2])
    plt.xticks([-2, -1, 0, 1, 2])
    plt.yticks([-2, -1, 0, 1, 2])
    constellation = ae.constellation()
    ax2.scatter(constellation[:, 0], constellation[:, 1])
    constellation_ac = ae.channel(torch.from_numpy(constellation))
    ax2.scatter(constellation_ac[:, 0], constellation_ac[:, 1])
    name2 = 'ae_hpa_constellation_M' + str(M) + '_snr' + str(snr) + '_ac.pdf'
    print(name2)
    fig2.savefig(os.path.join(your_path, 'plots', name2), format='pdf', bbox_inches='tight')
