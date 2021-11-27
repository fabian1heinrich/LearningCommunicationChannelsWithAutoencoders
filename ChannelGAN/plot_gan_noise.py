import numpy as np

import matplotlib.pyplot as plt
from numpy.random import randint


# import seaborn as sns
# my_cmap = sns.color_palette("crest", as_cmap=True)


def plot_gan_noise(gan, ax=None):

    n = 100000

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        ax = ax

    noise = gan.gen_noise(n)
    noise = noise.to('cpu')
    noise = noise.detach().numpy()
    # shuffle for hist2d
    noise = noise[randint(0, len(noise), size=(n, ))]

    range = np.array([[-2, 2], [-2, 2]])
    ax.hist2d(noise[:, 0], noise[:, 1], bins=100, range=range, cmap='plasma')

    return ax
