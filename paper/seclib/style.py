"""Print-size figure style for the paper (journal: SLEEP, Oxford University Press).

Adapted from the Steinmetz Lab style package (Anna Li, SteinmetzLab/steinmetzlab-style,
MIT licence): its 'paper' stylesheet and a few of its axis helpers, copied here so the
pipeline does not depend on a private repository. The lab's figure checks
(`steinmetzlab_style.check_figure`) still run when they are available; see `save`.

Sizes. SLEEP asks for figures at their final size and >= 300 dpi, with legible type and
no excess white space; its instructions give no column width, so we use OUP's usual
86 mm (one column) and 178 mm (two columns).

Text (2026-11-08, closer to the professor's V18 figures): all text bold, 10 pt labels
and titles, 9 pt tick labels and legends, 12 pt panel letters in brackets, "(a)". A 178 mm
figure sits about 6 in wide in the manuscript, so these print at roughly 8.5, 7.7 and
10 pt. The earlier journal sizes were 7 / 6 / 8 pt, regular weight.

Use
    from seclib import style
    @style.paper                       # every rcParam below applies inside the function
    def fig10(...):
        fig, axes = style.subplots(1, 3, style.COL2_MM, 58)
        ...
        style.spines_to_ticks(ax)      # once limits and ticks are final
        style.panel_label(ax, 'a')
        return style.save(fig, 'fig10_reproducibility', STAGE)

`save` writes <name>.png (white, 300 dpi, page not cropped) and <name>.pdf (vector,
editable text). With SEC_FIGCHECK=1 it also runs the lab's checks and writes
<name>.check.txt; set SLS_SRC to the steinmetzlab-style clone's `src` folder if the
package is not installed.
"""

from __future__ import annotations

import functools
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

from .config import FIG_DIR

MM = 1 / 25.4
COL1_MM, COL15_MM, COL2_MM = 86.0, 132.0, 178.0
MAX_H_MM = 230.0
MIN_FONT_PT = 5.0

PAPER_RC = {
    'figure.dpi': 300, 'savefig.dpi': 300,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'savefig.transparent': False,
    'figure.constrained_layout.use': True,
    'figure.constrained_layout.h_pad': 0.02, 'figure.constrained_layout.w_pad': 0.02,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10, 'legend.fontsize': 9,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'font.weight': 'bold', 'axes.labelweight': 'bold', 'figure.titleweight': 'bold',
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial:bold',
    'mathtext.it': 'Arial:italic:bold', 'mathtext.bf': 'Arial:bold',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'lines.linewidth': 1.0, 'lines.markersize': 3,
    'axes.linewidth': 1.0, 'axes.titlepad': 4, 'axes.labelpad': 2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.formatter.useoffset': False, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left',
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
    'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
    'xtick.major.pad': 3, 'ytick.major.pad': 3,
    'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.borderaxespad': 0.3,
    'image.interpolation': 'none',
}

# One style for every reference line (zero, chance, unity, null), as the lab style asks.
REFERENCE_STYLE = dict(color='0.5', lw=0.5, ls=(0, (4, 2)))
PANEL_LABEL_GID = 'steinmetzlab-panel-label'     # the id the lab's checks look for


def paper(func):
    """Decorator: draw (and save) the figure inside the paper rcParams."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with plt.rc_context(PAPER_RC):
            return func(*args, **kwargs)
    return wrapper


def figure(width_mm: float, height_mm: float, **kwargs):
    """A figure at its printed size (mm). Never rescale it afterwards."""
    return plt.figure(figsize=(width_mm * MM, height_mm * MM), **kwargs)


def subplots(nrows, ncols, width_mm: float, height_mm: float, **kwargs):
    return plt.subplots(nrows, ncols, figsize=(width_mm * MM, height_mm * MM), **kwargs)


def _ticks_in_view(axis, lo, hi):
    lo, hi = min(lo, hi), max(lo, hi)
    tol = 1e-9 * (hi - lo) if hi > lo else 1e-12
    return [t for t in axis.get_majorticklocs() if lo - tol <= t <= hi + tol]


def spines_to_ticks(ax, sides=('left', 'bottom')):
    """End each spine on its first and last tick. Call after limits and ticks are final."""
    for side in sides:
        axis, lims = ((ax.yaxis, ax.get_ylim()) if side in ('left', 'right')
                      else (ax.xaxis, ax.get_xlim()))
        ticks = _ticks_in_view(axis, *lims)
        if len(ticks) >= 2:
            ax.spines[side].set_bounds(ticks[0], ticks[-1])
    return ax


def label_end_ticks(ax, axis='y'):
    """Label only the first and last tick (0, 50, 100 shows 0 and 100); marks stay."""
    which = ax.yaxis if axis == 'y' else ax.xaxis
    ticks = _ticks_in_view(which, *(ax.get_ylim() if axis == 'y' else ax.get_xlim()))
    labels = which.get_major_formatter().format_ticks(ticks)
    labels = [lab if i in (0, len(ticks) - 1) else '' for i, lab in enumerate(labels)]
    (ax.set_yticks if axis == 'y' else ax.set_xticks)(ticks, labels)
    return ax


def panel_label(ax, letter, dx_pt=-12, dy_pt=4, fontsize=12):
    """Bold lower-case panel letter in brackets, "(a)", at the top left of `ax`,
    offset in points."""
    return ax.annotate(f'({letter})', gid=PANEL_LABEL_GID, xy=(0, 1), xycoords='axes fraction',
                       xytext=(dx_pt, dy_pt), textcoords='offset points', ha='right',
                       va='bottom', fontsize=fontsize, fontweight='bold',
                       annotation_clip=False)


def reference_line(ax, y=None, x=None, **kw):
    """Zero / chance / null line in the shared style, kept inside the axis lines."""
    st = {**REFERENCE_STYLE, 'zorder': 0.5, **kw}
    if y is not None:
        return ax.axhline(y, **st)
    if x is not None:
        return ax.axvline(x, **st)
    raise ValueError('pass y= or x=')


def _lab_checks(fig, name):
    """steinmetzlab_style.check_figure, if importable; returns lines or None."""
    src = os.environ.get('SLS_SRC')
    if src and src not in sys.path:
        sys.path.insert(0, src)
    try:
        import steinmetzlab_style as sls
    except ImportError:
        return None
    issues = sls.check_figure(fig, min_font_pt=MIN_FONT_PT)
    req = sum(i.required for i in issues)
    return [f'{name}: {req} required, {len(issues) - req} preference'] + \
        [str(i) for i in issues]


def save(fig, name: str, stage: str, check: bool | None = None):
    """Write <name>.png (white, 300 dpi) and <name>.pdf at printed size; close the figure.

    The page is never cropped: cropping would change the printed width.
    """
    out = FIG_DIR / stage / f'{name}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    if check is None:
        check = bool(os.environ.get('SEC_FIGCHECK'))
    # Checks and savefig read rcParams (font family, dpi, embedding) at draw time, so
    # pin them here: the caller may save outside @paper (a figure built in a helper).
    with plt.rc_context(PAPER_RC):
        if check:
            lines = _lab_checks(fig, name)
            if lines is not None:
                out.with_suffix('.check.txt').write_text('\n'.join(lines) + '\n',
                                                         encoding='utf8')
        fig.savefig(out, dpi=300, facecolor='white', transparent=False)
        fig.savefig(out.with_suffix('.pdf'), transparent=True)
    plt.close(fig)
    return out
