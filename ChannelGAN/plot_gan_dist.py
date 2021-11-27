import numpy as np

import matplotlib.pyplot as plt
from numpy.random import randint


# import seaborn as sns
# my_cmap = sns.color_palette("crest", as_cmap=True)


def plot_gan_dist(gan, ax=None):

    n = 100000

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        ax = ax

    dist = gan.gen_channel_output(n)
    dist = dist.to('cpu')
    dist = dist.detach().numpy()
    # shuffle for hist2d
    dist = dist[randint(0, len(dist), size=(n, ))]

    range = np.array([[-2, 2], [-2, 2]])
    ax.hist2d(dist[:, 0], dist[:, 1], bins=100, range=range, cmap='plasma')

    return ax
