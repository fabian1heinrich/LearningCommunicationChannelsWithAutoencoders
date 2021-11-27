import matplotlib.pyplot as plt

from constellation import constellation


def scatter_plot(autoencoder, ax=None):

    x = constellation(autoencoder)

    if ax is None:
        fig, ax = plt.subplots()
        ax.grid()
        ax.set_aspect('equal', adjustable='box')
        plt.axis([-3, 3, -3, 3])

    else:
        ax = ax

    ax.scatter(x[:, 0], x[:, 1])

    return ax
