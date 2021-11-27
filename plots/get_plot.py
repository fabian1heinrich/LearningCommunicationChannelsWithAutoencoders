import matplotlib.pyplot as plt
import matplotlib as mpl

from tex_fonts import tex_fonts


mpl.rcParams.update(tex_fonts)
plt.style.use('seaborn-muted')
width = 433
inches_per_pt = 1 / 72
golden_ratio = (5**.5 + 1) / 2


def get_plot(scale, k):

    fig_width = scale * width * inches_per_pt
    fig_height = fig_width / golden_ratio

    if k == 1:
        figsize = (fig_width, fig_height)
    elif k == 2:
        figsize = (fig_width, fig_width)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.grid()
    # ax.autoscale(enable=True, axis='both', tight=True)

    if k == 2:
        ax.set_aspect('equal', adjustable='box')

    return fig, ax


# fig, ax = get_plot(0.5, 2)

# ebno = np.linspace(0, 20, 21, endpoint=True)
# ax.plot(ebno, ebno)

# plt.xlabel(r'some random latex $\omega$')
# plt.xticks(ebno[0:None:4])
# plt.ylabel(r'some random latex $\phi$')
# plt.yticks(ebno[0:None:4])

# # save in directory
# import os
# import numpy as np

# plt.show()
# name = 'test_neu4.pdf'
# your_path = os.path.dirname(__file__)
# fig.savefig(os.path.join(your_path, name), format='pdf', bbox_inches='tight')
