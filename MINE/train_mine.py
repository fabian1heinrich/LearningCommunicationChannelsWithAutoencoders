import torch
import matplotlib.pyplot as plt
from torch import randint, mean, log, exp
import numpy as np


def train_mine(mine, samples, n_epochs, batch_size, lr, mi_true):

    if mi_true is not None:
        fig, ax = plt.subplots()
    optimizer = torch.optim.Adam(mine.parameters(), lr=lr)

    mi_pred = np.zeros(1)
    for epoch in range(n_epochs):

        optimizer.zero_grad()

        x = samples[0, :, :]
        y = samples[1, :, :]

        index_joint = randint(0, len(x), size=(batch_size, ))
        index_marginal = randint(0, len(x), size=(batch_size, ))

        pred_joint = mine(x[index_joint], y[index_joint])
        pred_marginal = mine(x[index_joint], y[index_marginal])

        loss = -(mean(pred_joint) - log(mean(exp(pred_marginal))))
        loss.backward()
        optimizer.step()

        cur_mi_pred = np.log2(np.exp(1)) * loss.item()  # base-2
        print(cur_mi_pred)
        mi_pred = np.append(mi_pred, -cur_mi_pred)

        if mi_true is not None:
            ax.clear()
            ax.grid()
            ax.autoscale(enable=True, axis='both', tight=True)
            plt.ylim(0, mi_true + 0.5)
            ax.plot(mi_pred)
            ax.plot(np.arange(n_epochs + 1), mi_true * np.ones(n_epochs + 1))
            plt.pause(0.01)

    return mi_pred