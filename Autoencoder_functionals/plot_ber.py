import matplotlib.pyplot as plt

from calc_ber_awgn import calc_ber_awgn


def plot_ber(autoencoder, eb_no, ax=None, line=None):

    snr, ber = calc_ber_awgn(autoencoder, eb_no)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        ax = ax

    if line is None:
        line = 'bo-'
    else:
        line = line

    ax.grid()
    ax.set_yscale('log')
    # ax.plot(snr, ber, line)
    ax.plot(eb_no, ber, line)

    return ax
