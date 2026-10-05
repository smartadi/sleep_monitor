"""
Print style for the supplementary rate figures.

The supplement's text width is 6.0 in (8.5 in page, 1.25 in margins). Every
figure is drawn 9 in wide and placed at that width, a scale of 2/3, so the font
sizes below land at about 8-9 pt on the printed page, the size of a figure label
in the body of a paper. Height is chosen per figure.

Figures carry no titles: what each panel shows goes in the caption, and panels
are marked with a bold letter in the top-left corner, as in the main text.
"""

import matplotlib.pyplot as plt

WIDTH_IN = 9.0

STYLE = {
    'font.size': 13, 'axes.labelsize': 13.5, 'axes.titlesize': 13.5,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11.5,
    'font.weight': 'normal', 'axes.labelweight': 'normal',
    'axes.titleweight': 'normal',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.9,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'figure.dpi': 110, 'savefig.dpi': 300,
}


def apply():
    plt.rcParams.update(STYLE)


def letter(ax, s, x=-0.10, y=1.04):
    """Bold panel letter just outside the top-left corner of the axes."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=16, fontweight='bold',
            va='bottom', ha='left')
