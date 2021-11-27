import torch
import torch.nn as nn
import numpy as np

from MINE import MINE


layers = nn.Sequential(
    nn.Linear(20, 20),
    nn.ReLU(),
    nn.Linear(20, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

n_dim = 2
signal_power = 7
noise_power = 1
n_samples = 1000000
mi_true = n_dim * 0.5 * np.log2(1 + signal_power / noise_power)
mine = MINE(n_dim, layers)

x = torch.normal(0, np.sqrt(signal_power), size=(n_samples, n_dim))
noise = torch.normal(0, 1, size=x.size())
y = x + noise

samples = torch.stack((x, y), dim=0)

mine.train_mine(samples, 100, 100000, 0.005, mi_true)

a = 1
