import matplotlib.pyplot as plt


def plot_gan(gan):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    gan.plot_gan_noise(ax1)
    gan.plot_gan_gen(ax2)
    gan.plot_gan_dist(ax3)

    plt.show()
