import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch import normal, mean, log, exp

# torch.manual_seed(0)


class Mine(nn.Module):

    def __init__(self, N_dim):
        super(Mine, self).__init__()

        self.input1 = nn.Linear(N_dim, 20)
        self.input2 = nn.Linear(N_dim, 20)
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x, y):
        h1 = self.input1(x) + self.input2(y)
        h2 = self.layers(h1)
        return h2


signal_power = 7
noise_power = 1
n_dim = 2
n_samples = 1000000
batch_size = 1000000
n_epochs = 200

mi_pred = torch.zeros(n_epochs)


model = Mine(n_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

dev = 'cuda:0'
model.to(dev)

x = normal(0, np.sqrt(signal_power), size=(n_samples, n_dim))
x = x.to(dev)
noise = normal(0, np.sqrt(noise_power), size=x.size())
noise = noise.to(dev)
y = x + noise
for epoch in range(n_epochs):
    optimizer.zero_grad()

    x = normal(0, np.sqrt(signal_power), size=(n_samples, n_dim))
    x = x.to(dev)
    noise = normal(0, np.sqrt(noise_power), size=x.size())
    noise = noise.to(dev)
    y = x + noise

    index1 = torch.randint(0, len(x), size=(batch_size, 1))
    index2 = torch.randint(0, len(x), size=(batch_size, 1))
    index3 = torch.randint(0, len(x), size=(batch_size, 1))
    y_shuffle = y[torch.randperm(len(y))]
    x_shuffle = x[torch.randperm(len(y))]

    pred_xy = model(x[index1], y[index1])
    pred_x_y = model(x[index2], y[index3])

    loss = -(mean(pred_xy) - log(mean(exp(pred_x_y))))
    mi_pred[epoch] = loss

    loss.backward()
    optimizer.step()
    print('epoch #{}/{} | loss = {:.6f}'.format(epoch + 1, n_epochs, loss))


# true mutual information for gaussian
mi_true = n_dim * 0.5 * np.log2(1 + signal_power / noise_power)
mi_pred = mi_pred.detach().numpy()
# mi_pred to base 2
mi_pred = np.log2(np.exp(1)) * np.abs(mi_pred)

plt.figure()
plt.plot(range(n_epochs), np.ones(n_epochs) * mi_true, label='mi_true')
plt.plot(range(n_epochs), mi_pred, label='mi_pred')
plt.grid()
plt.legend()
plt.title('MINE')
plt.show()

print('durch')