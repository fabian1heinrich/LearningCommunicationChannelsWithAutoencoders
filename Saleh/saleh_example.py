import _paths_saleh

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider

from ChannelSaleh import ChannelSaleh
from constellation_16QAM import constellation_16QAM

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)
ax.grid()
ax.set_aspect('equal', 'box')

# constellation
constetalltion = constellation_16QAM()

# init params
amam = [1, 1]
ampm = [1, 1]

channel = ChannelSaleh(amam, ampm)

amam1 = plt.axes([0.25, 0.25, 0.65, 0.03])
amam2 = plt.axes([0.25, 0.2, 0.65, 0.03])
ampm1 = plt.axes([0.25, 0.15, 0.65, 0.03])
ampm2 = plt.axes([0.25, 0.1, 0.65, 0.03])

s_amam1 = Slider(amam1, 'alpha_a', 0.0, 8.0, valinit=amam[0], valstep=0.1)
s_amam2 = Slider(amam2, 'beta_a', 0.0, 8.0, valinit=amam[1], valstep=0.1)
s_ampm1 = Slider(ampm1, 'alpha_phi', 0.0, 8.0, valinit=ampm[0], valstep=0.1)
s_ampm2 = Slider(ampm2, 'beta_phi', 0.0, 8.0, valinit=ampm[1], valstep=0.1)

plt.show()