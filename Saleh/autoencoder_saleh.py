import _paths
import torch
import numpy as np
import matplotlib.pyplot as plt

from AutoencoderSaleh import AutoencoderSaleh


# matlab default
# amam = [2.1587, 1.1517]
# ampm = [4.0033, 9.1040]

# normalized?
amam = [2.0, 1.0]
ampm = [np.pi / 3, 1.0]

autoencoder = AutoencoderSaleh(16, 1, amam, ampm)

autoencoder.init_weights_n()
autoencoder.train_ae(16000, 100, 0.01)
autoencoder.train_ae(16000, 20, 0.001)

x1 = autoencoder.constellation()
x2 = torch.from_numpy(x1)
x2 = autoencoder.channel(x2)

plt.scatter(x1[:, 0], x1[:, 1], c='b')
plt.scatter(x2[:, 0], x2[:, 1], c='g')

x = np.linspace(0, 2 * np.pi, 17)
c = np.exp(x[:-1] * 1j)
c = [np.real(c), np.imag(c)]
c = np.transpose(c)
plt.scatter(c[:, 0], c[:, 1], c='r', marker='+')
plt.grid()

print(x1)
print(x2)
print(c)
plt.axis([-2, 2, -2, 2])
plt.show()
