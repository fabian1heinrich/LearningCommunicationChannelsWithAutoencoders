import _paths_mine

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from get_plot import get_plot
from MINE import MINE


layers = nn.Sequential(
    nn.Linear(40, 40),
    nn.Tanh(),
    nn.Linear(40, 40),
    nn.Tanh(),
    nn.Linear(40, 1)
)

n_dim = 2
signal_power = 7
noise_power = 1
n_samples = 1000000

mine = MINE(n_dim, layers)
mi_true = n_dim * 0.5 * np.log2(1 + signal_power / noise_power)

x = torch.normal(0, np.sqrt(signal_power), size=(n_samples, n_dim))
noise = torch.normal(0, 1, size=x.size())
y = x + noise
samples = torch.stack((x, y), dim=0)

n_epochs = 100
batch_size = 10000
lr = 0.005
mi_pred = mine.train_mine(samples, n_epochs, batch_size, lr)

fig1, ax1 = get_plot(1, 1)
plt.ylim(0, mi_true + 0.5)

ax1.plot(mi_pred)
ax1.plot(np.arange(n_epochs + 1), mi_true * np.ones(n_epochs + 1))
plt.ylabel(r'Mutual Information in bits')
plt.xlabel(r'epochs')
ax1.legend((r'estimated MI', r'true MI'), loc='lower right')

name1 = 'mine1.pdf'
print(name1)
your_path = os.path.dirname(__file__)
fig1.savefig(os.path.join(your_path, name1), format='pdf', bbox_inches='tight')

plt.show()
