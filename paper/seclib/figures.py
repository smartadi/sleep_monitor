"""One figure style for the paper code, and one way to save."""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

from .config import FIG_DIR

STYLE = {
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'savefig.dpi': 200,
}
plt.rcParams.update(STYLE)


def save(fig, name: str, stage: str):
    """Save under outputs/figures/<stage>/<name>.png and close the figure."""
    out = FIG_DIR / stage / f'{name}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out
