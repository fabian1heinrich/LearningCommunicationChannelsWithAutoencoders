import _paths_mine

import torch
import torch.nn as nn
import numpy as np

from torch import randint, normal

from MINE import MINE
from constellation_4QAM import constellation_4QAM


n_samples = int(1e6)
constellation = constellation_4QAM()
noise_power = 1

x = constellation[randint(0, len(constellation), size=(n_samples,))]
noise = normal(0, np.sqrt(noise_power / 2), size=x.size())
y = x + noise
samples = torch.stack((x, y), dim=0)

layers = nn.Sequential(
    nn.Linear(40, 40),
    nn.Tanh(),
    nn.Linear(40, 40),
    nn.Tanh(),
    nn.Linear(40, 1)
)

mine = MINE(2, layers)

n_epochs = 100
batch_size = 10000
lr = 0.005
mine.train_mine(samples, n_epochs, batch_size, lr, 0.5)

a = 1