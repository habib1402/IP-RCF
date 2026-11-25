"""Shared style and drawing helpers for the IP-RCF TMI figures.

Print sizes: full width 7.0 in (4200 px at 600 dpi); single column 3.5 in (2100 px).
Palette (Okabe-Ito based, colour-blind safe):
  semantic stream  #1F4E79 (navy)      contour stream #B5452B (brick)
  auxiliary/training-only #7A5195 (purple, dashed lines)
  mask errors: TP white, FP #D55E00 (vermillion), FN #56B4E9 (sky blue)
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

PAPER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIGDIR = os.path.join(PAPER, 'figures')
DPI = 600
FULL_W = 7.0
COL_W = 3.5

SEM = '#1F4E79'
SEM_FILL = '#E9EEF4'
CON = '#B5452B'
CON_FILL = '#F5E9E6'
AUX = '#7A5195'
AUX_FILL = '#EFE7F5'
NEU = '#4D4D4D'
NEU_FILL = '#F2F2F2'
FP = '#D55E00'
FN = '#56B4E9'
PENDING = '#9A9A9A'

# Register the actual font faces for both labels and math text in PNG renders.
# Release configuration (environment variables):
#   IPRCF_ROOT            repository root holding runs/, results/, analyses/ (default: repo root)
#   POLYP_DATA_ROOT       folder with TrainDataset/ and TestDataset/ (default: <repo>/data)
#   COMPARATOR_MAPS_ROOT  released comparator prediction maps (optional)
#   TMI_FIGURE_FONT_DIR   folder with Times New Roman .ttf files (optional; serif fallback)
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
STUDY_ROOT = os.environ.get('IPRCF_ROOT', REPO_ROOT)
DATA_ROOT = os.environ.get('POLYP_DATA_ROOT', os.path.join(REPO_ROOT, 'data'))
COMPARATOR_ROOT = os.environ.get('COMPARATOR_MAPS_ROOT', os.path.join(REPO_ROOT, 'comparator_maps'))

FONT = 'Times New Roman'
font_dir = Path(os.environ.get('TMI_FIGURE_FONT_DIR', os.path.join(REPO_ROOT, 'fonts')))
for font_file in sorted(font_dir.glob('times*.ttf')):
    if font_manager.FontProperties(fname=font_file).get_name() == FONT:
        font_manager.fontManager.addfont(font_file)

try:
    for style, weight in [('normal', 'normal'), ('normal', 'bold'),
                          ('italic', 'normal'), ('italic', 'bold')]:
        font_manager.findfont(
            font_manager.FontProperties(family=FONT, style=style, weight=weight),
            fallback_to_default=False,
        )
except ValueError:
    import warnings
    warnings.warn('Times New Roman not found; set TMI_FIGURE_FONT_DIR. Falling back to serif.')
    FONT = 'serif'

plt.rcParams.update({
    'font.family': FONT,
    'font.serif': [FONT],
    'font.size': 7,
    'axes.linewidth': 0.6,
    'axes.edgecolor': NEU,
    'axes.labelsize': 7,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'legend.fontsize': 6.5,
    'mathtext.fontset': 'custom',
    'mathtext.rm': FONT,
    'mathtext.it': FONT + ':italic',
    'mathtext.bf': FONT + ':bold',
    'mathtext.cal': FONT + ':italic',
    'mathtext.sf': FONT,
    'mathtext.tt': FONT,
    'mathtext.fallback': None,
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
})


def canvas(width, height):
    """Figure whose single axes uses inch coordinates (1 unit = 1 inch)."""
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    return fig, ax


def box(ax, cx, cy, w, h, text='', ec=NEU, fc=NEU_FILL, ls='-', lw=0.8,
        fs=7, color=None, weight='normal', r=0.03, z=2, shadow=False):
    if shadow:
        ax.add_patch(FancyBboxPatch((cx - w / 2 + 0.016, cy - h / 2 - 0.016), w, h,
                     boxstyle='round,pad=0,rounding_size={}'.format(r),
                     ec='none', fc='black', alpha=0.16, zorder=z - 0.05))
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle='round,pad=0,rounding_size={}'.format(r),
                       ec=ec, fc=fc, lw=lw, ls=ls, zorder=z)
    ax.add_patch(p)
    if text:
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
                color=color or 'black', weight=weight, zorder=z + 1,
                linespacing=1.15)
    return p


def op(ax, cx, cy, sym='+', ec=NEU, r=0.065, lw=0.8, fs=7.5, z=4):
    ax.add_patch(Circle((cx, cy), r, ec=ec, fc='white', lw=lw, zorder=z))
    ax.text(cx, cy - 0.003, sym, ha='center', va='center', fontsize=fs,
            color=ec, zorder=z + 1, weight='bold')


def path(ax, pts, color=NEU, lw=0.9, ls='-', head=True, z=1, hw=0.045, hl=0.06):
    """Poly-line through pts (inches) with an arrow head on the last segment."""
    xs, ys = zip(*pts)
    if head:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        d = np.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / d, (y1 - y0) / d
        bx, by = x1 - ux * hl, y1 - uy * hl
        ax.plot(list(xs[:-1]) + [bx], list(ys[:-1]) + [by], color=color, lw=lw,
                ls=ls, zorder=z, solid_capstyle='butt', dash_capstyle='butt')
        px, py = -uy, ux
        tri = [(x1, y1), (bx + px * hw / 2, by + py * hw / 2),
               (bx - px * hw / 2, by - py * hw / 2)]
        ax.add_patch(plt.Polygon(tri, closed=True, fc=color, ec=color, lw=0.3,
                                 zorder=z + 0.5))
    else:
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z,
                solid_capstyle='butt')


def dot(ax, x, y, color=NEU, s=0.022, z=3):
    ax.add_patch(Circle((x, y), s, fc=color, ec=color, lw=0, zorder=z))


def panel_label(ax, x, y, s, fs=8):
    ax.text(x, y, s, ha='left', va='top', fontsize=fs, weight='bold')


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, name)
    fig.savefig(out, dpi=DPI, facecolor='white')
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(out).size
    print('wrote {} ({} x {} px)'.format(out, w, h))
    return out


def draft_banner(fig, text):
    fig.text(0.5, 0.995, text, ha='center', va='top', fontsize=6.5,
             color='#7F2704', weight='bold',
             bbox=dict(boxstyle='round,pad=0.25', fc='#FDE7D7', ec='#D55E00',
                       lw=0.6))


def pending_panel(ax, text='pending', fs=6.5):
    ax.set_facecolor('#F4F4F4')
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                           hatch='////', ec='#C8C8C8', lw=0))
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha='center', va='center',
            fontsize=fs, color='#555555', linespacing=1.2,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#BDBDBD',
                      lw=0.5))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#BDBDBD'); s.set_linewidth(0.5)
