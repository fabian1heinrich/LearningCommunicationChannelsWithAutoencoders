# fig:ae_awgn_bler

import _paths_ae_awgn

import os
import torch
import numpy as np
import torch.nn as nn

from ChannelAutoencoder import ChannelAutoencoder
from ChannelAWGN import ChannelAWGN
from ChannelAWGN_uniform import ChannelAWGN_uniform
from NormLayer import NormLayer

import matplotlib.pyplot as plt
from get_plot import get_plot

# os.environ['CUDA_VISIBLE_DEVICES'] = ''
print(torch.cuda.is_available())
your_path = os.path.dirname(__file__)

torch.manual_seed(0)
M = 16
N_channel = 1

snr_bler = np.arange(0, 23)
bler = np.zeros((len(snr_bler), 2))

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
    nn.Tanh(),
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
    nn.Linear(128, 128),
    nn.Tanh(),
    nn.Linear(128, M)
)
torch.manual_seed(0)
channel1 = ChannelAWGN(12)
ae1 = ChannelAutoencoder(encoder, channel1, decoder)
ae1.init_weights_n()
ae1.train_ae(16000, 40, 1000, 0.01, True)
ae1.train_ae(16000, 20, 5000, 0.005, True)
ae1.train_ae(16000, 20, 5000, 0.001, True)
ae1.scatter_plot()

torch.manual_seed(0)
channel2 = ChannelAWGN_uniform(12, 15)
ae2 = ChannelAutoencoder(encoder, channel2, decoder)
ae2.init_weights_n()
ae2.train_ae(16000, 40, 1000, 0.01, True)
ae2.train_ae(16000, 20, 5000, 0.005, True)
ae2.train_ae(16000, 20, 5000, 0.001, True)
ae2.scatter_plot()
plt.show()
# calc bler
# matlab_bler_16qam = np.loadtxt('matlab_ser_16qam.txt')
# bler[:, 0] = ae1.calc_ber_awgn(snr_bler)
# bler[:, 1] = ae2.calc_ber_awgn(snr_bler)

# constellation plot
fig1, ax1 = get_plot(0.3, 2)
ax1.axis([-2, 2, -2, 2])
plt.xticks([-2, -1, 0, 1, 2])
plt.yticks([-2, -1, 0, 1, 2])
constellation = ae2.constellation()
ax1.scatter(constellation[:, 0], constellation[:, 1])
name1 = 'ae_awgn_constellation_M' + str(M) + '_uniform_snr' + '.pdf'
print(name1)
fig1.savefig(os.path.join(your_path, 'plots', name1), format='pdf', bbox_inches='tight')
plt.clf()


fig2, ax2 = get_plot(1, 1)
ax2.set_yscale('log')
ax2.plot(matlab_bler_16qam[:, 0], matlab_bler_16qam[:, 1], ':', linewidth=3)
ax2.plot(snr_bler, bler, ':', linewidth=3)
plt.ylabel(r'$BLER$')
plt.xlabel(r'$SNR$')
plt.xticks(snr_bler[::2])
ax2.legend((r'$16QAM$', r'$SNR=12$ dB', r'$SNR \sim U(12, 15)$'), loc='lower left')
name2 = 'ae_awgn_M' + str(M) + '_bler_w16QAM.pdf'
print(name1)
fig2.savefig(os.path.join(your_path, 'plots', name2), format='pdf', bbox_inches='tight')
plt.clf()

ae1.scatter_plot()