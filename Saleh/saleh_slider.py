import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from torch import sqrt, sum, mean, square

from saleh_pytorch import saleh_pytorch


fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.35)
plt.axis([-2, 2, -2, 2])
ax.grid()
ax.set_aspect('equal', 'box')


qam_coeffs = np.array([
    1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j,
    1 + 3j, 1 - 3j, -1 + 3j, -1 - 3j,
    3 + 1j, 3 - 1j, -3 + 1j, -3 - 1j,
    3 + 3j, 3 - 3j, -3 + 3j, -3 - 3j])

coeffs = qam_coeffs
# coeffs = qam_coeffs * np.exp(1j * np.pi/4)
coeffs = [np.real(coeffs), np.imag(coeffs)]
coeffs = np.transpose(coeffs)
coeffs = torch.from_numpy(coeffs)
coeffs = coeffs / sqrt(sum(2 * mean(square(coeffs))))
coeffs = coeffs.float()
coeffs_saleh = saleh_pytorch(coeffs, [2.1587, 1.1517], [4.0033, 9.1040])


# t = np.arange(0.0, 1.0, 0.001)
# a0 = 5
# f0 = 3
# delta_f = 5.0
# s = a0 * np.sin(2 * np.pi * f0 * t)
ax.scatter(coeffs[:, 0], coeffs[:, 1])
ax.scatter(coeffs_saleh[:, 0], coeffs_saleh[:, 1])
# l, = plt.plot(t, s, lw=2)
ax.margins(x=0)

axcolor = 'lightgoldenrodyellow'
amam1 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
amam2 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ampm1 = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
ampm2 = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)

s_amam1 = Slider(amam1, 'alpha_a', 0.1, 10.0, valinit=2, valstep=0.1)
s_amam2 = Slider(amam2, 'beta_a', 0.1, 10.0, valinit=1, valstep=0.1)
s_ampm1 = Slider(ampm1, 'alpha_phi', 0.1, 10.0, valinit=np.pi/3, valstep=0.1)
s_ampm2 = Slider(ampm2, 'beta_phi', 0.1, 10.0, valinit=1, valstep=0.1)


def update(val):
    alpha_a = s_amam1.val
    beta_a = s_amam2.val
    amam_params = [alpha_a, beta_a]

    alpha_phi = s_ampm1.val
    beta_phi = s_ampm2.val
    ampm_params = [alpha_phi, beta_phi]

    y = saleh_pytorch(coeffs, amam_params, ampm_params)
    ax.clear()
    ax.scatter(coeffs[:, 0], coeffs[:, 1])
    ax.scatter(y[:, 0], y[:, 1])
    ax.axis([-2, 2, -2, 2])
    ax.grid()
    ax.set_aspect('equal', 'box')
    fig.canvas.draw_idle()


s_amam1.on_changed(update)
s_amam2.on_changed(update)
s_ampm1.on_changed(update)
s_ampm2.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    s_amam1.reset()
    s_amam2.reset()
    s_ampm1.reset()
    s_ampm2.reset()
button.on_clicked(reset)


rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw_idle()
radio.on_clicked(colorfunc)


plt.show()
