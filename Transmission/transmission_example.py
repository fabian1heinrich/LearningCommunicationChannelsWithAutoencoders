import _paths_transmission

import torch
import numpy as np
import matplotlib.pyplot as plt

from AutoencoderTransmission import AutoencoderTransmission
from transmission_pytorch import transmission_pytorch
from TxParameters import TxParameters


# constellation
constellation = np.array([
    1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j,
    1 + 3j, 1 - 3j, -1 + 3j, -1 - 3j,
    3 + 1j, 3 - 1j, -3 + 1j, -3 - 1j,
    3 + 3j, 3 - 3j, -3 + 3j, -3 - 3j]) * np.exp(1j * np.pi / 4) / np.sqrt(10)

constellation = np.stack((np.real(constellation), np.imag(constellation)), axis=1)
constellation = torch.from_numpy(constellation)
constellation = constellation.to('cuda:0')

n = 1024
tx_index = torch.randint(0, len(constellation), size=(n,))

# equivalent to output from encoder
tx_symbols = constellation[tx_index]


ebno = 100
input_backoff = 0
amam_params = [2.0, 1.0]
ampm_params = [np.pi / 3, 1.0]
samples_per_symbol = 10  # lower 10
n_symbols_pulse_shaping = 16  # 16

tx_parameters = TxParameters(
    ebno,
    input_backoff,
    amam_params,
    ampm_params,
    samples_per_symbol,
    n_symbols_pulse_shaping
)
print('filter length:', samples_per_symbol * n_symbols_pulse_shaping + 1)

ae = AutoencoderTransmission(16, 1, tx_parameters)

rx_symbols = ae.channel(tx_symbols)

tx_symbols = tx_symbols.to('cpu')
tx_symbols = tx_symbols.detach().numpy()

rx_symbols = rx_symbols.to('cpu')
rx_symbols = rx_symbols.detach().numpy()

plt.scatter(rx_symbols[:, 0], rx_symbols[:, 1])
plt.scatter(tx_symbols[:, 0], tx_symbols[:, 1])
u = rx_symbols - tx_symbols
print(u)
print(np.sum(np.abs(u)))
plt.show()
