import numpy as np
import torch

import matplotlib.pyplot as plt
from numpy.random import randint


# import seaborn as sns
# my_cmap = sns.color_palette("crest", as_cmap=True)


def plot_gan_gen(gan, ax=None):

    n = 100000

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        ax = ax

    noise = gan.gen_noise(n)
    channel_input = gan.gen_channel_input(n)

    generator_input = torch.cat((channel_input, noise), dim=1)
    generated_samples = gan.generator(channel_input, noise)
    gen = generated_samples
    gen = gen.to('cpu')
    gen = gen.detach().numpy()
    # shuffle for hist2d
    gen = gen[randint(0, len(gen), size=(n, ))]

    range = np.array([[-2, 2], [-2, 2]])
    ax.hist2d(gen[:, 0], gen[:, 1], bins=100, range=range, cmap='plasma')

    return ax
