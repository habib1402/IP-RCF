"""Fig. 2: IP-RCF overall network (implementation: HMPFormer_Backbone_Study/lib/iprcf.py).

Encoder: timm pvt_v2_b0 (embed dims 32/64/160/256, depths 2/2/2/2,
SR ratios 8/4/2/1, overlapping patch embedding 7x7/4 then 3x3/2).
Input frames are illustrative development images; no prediction is shown.
Operator names follow the manuscript: A_3 = 3x3 local-mean operator,
R = depthwise-separable refinement (depthwise spatial mixing, pointwise channel mixing).
"""
import os
import numpy as np
from PIL import Image
from matplotlib.patches import FancyBboxPatch, Polygon
from figstyle import *

TEST = DATA_ROOT + '/TestDataset/'
FRAMES = [('Kvasir-SEG', 'Kvasir/images/cju1cqc7n4gpy0855jt246k68.png'),
          ('CVC-ClinicDB', 'CVC-ClinicDB/images/{3}'),
          ('CVC-ColonDB', 'CVC-ColonDB/images/120.png'),
          ('ETIS-Larib', 'ETIS-LaribPolypDB/images/105.png')]

W, H = FULL_W, 3.62
fig, ax = canvas(W, H)
ax.set_ylim(-0.04, H - 0.04)
LANE_FILL, LANE_EDGE = '#F6F6F6', '#CFCFCF'
ENC = '#5F7183'
E_DARK, E_MID, E_LIGHT, E_TINT = '#2F6B6E', '#4E8F8C', '#9CC7C2', '#E4F1EF'
ENC = E_DARK
ENC_ST = ['#A9D0CB', '#7EB3AD', '#559390', '#2F6B6E']
R_DARK, R_MID = '#3E4A56', '#6E7C8A'
DARK = '#2F2F2F'


def shade(hexc, f):
    c = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return tuple(np.clip(c * f if f < 1 else c + (1 - c) * (f - 1), 0, 1))


def cuboid(x, y, w, h, d, base, ec='#3A3A3A', grid=0, z=3):
    dx, dy = d * 0.75, d * 0.55
    ax.add_patch(Polygon([(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy),
                          (x + w, y + h)], fc=shade(base, 1.45), ec=ec, lw=0.45, zorder=z))
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy),
                          (x + w, y + h)], fc=shade(base, 0.75), ec=ec, lw=0.45, zorder=z))
    ax.add_patch(Rectangle((x, y), w, h, fc=base, ec=ec, lw=0.45, zorder=z))
    for k in range(1, grid):
        ax.plot([x + w * k / grid] * 2, [y, y + h], color='white', lw=0.3, alpha=0.7, zorder=z + 0.1)
        ax.plot([x, x + w], [y + h * k / grid] * 2, color='white', lw=0.3, alpha=0.7, zorder=z + 0.1)


def load_square(rel):
    if '{3}' in rel:
        d = TEST + rel.split('{3}')[0]
        rel = rel.replace('{3}', sorted(os.listdir(d))[3])
    im = Image.open(TEST + rel).convert('RGB')
    s = min(im.size)
    return np.asarray(im.crop(((im.width - s) // 2, (im.height - s) // 2,
                               (im.width + s) // 2, (im.height + s) // 2)).resize((300, 300)))


def node(cx, cy, text, col):
    box(ax, cx, cy, 0.38, 0.28, text, ec=col, fc=col, fs=7.5, color='white', shadow=True)


# ------------------------------------------------------------ input column
rows = [0.98, 1.55, 2.12, 2.68]                    # encoder stage rows (stride 4..32)
ax.text(0.33, 3.40, 'Input frames', ha='center', va='bottom', fontsize=7, weight='bold')
ax.text(0.33, 3.25, r'3$\times$352$\times$352', ha='center', va='bottom',
        fontsize=6.5, color=NEU)
fsz = 0.50
fy = [2.70, 2.07, 1.44, 0.81]
for (lab, rel), y in zip(FRAMES, fy):
    ax.imshow(load_square(rel), extent=(0.08, 0.08 + fsz, y, y + fsz), zorder=2)
    ax.add_patch(Rectangle((0.08, y), fsz, fsz, fill=False, ec='#3A3A3A', lw=0.5, zorder=3))
    ax.text(0.33, y - 0.03, lab, ha='center', va='top', fontsize=6, color=NEU)
    ax.plot([0.08 + fsz, 0.68], [y + fsz / 2] * 2, color=NEU, lw=0.6)
ax.plot([0.68, 0.68], [fy[-1] + fsz / 2, fy[0] + fsz / 2], color=NEU, lw=0.6)
path(ax, [(0.68, rows[0]), (0.86, rows[0])])

# ------------------------------------------------------------ encoder
ex0, ex1 = 0.80, 2.13
ax.add_patch(FancyBboxPatch((ex0, 0.70), ex1 - ex0, 2.48,
             boxstyle='round,pad=0,rounding_size=0.05', ec=E_LIGHT, fc='#F3F8F7', lw=0.7, zorder=1))
ax.text((ex0 + ex1) / 2, 3.40, 'PVTv2-B0 encoder', ha='center', va='bottom',
        fontsize=7, weight='bold')
ax.text((ex0 + ex1) / 2, 3.25, 'ImageNet-1K init.',
        ha='center', va='bottom', fontsize=6.5, color=NEU)
spatial = [88, 44, 22, 11]
chans = [32, 64, 160, 256]
face = {88: 0.30, 44: 0.24, 22: 0.18, 11: 0.13}
depth = {32: 0.05, 64: 0.08, 160: 0.13, 256: 0.17}
down = ['4', '2', '2', '2']
sr = [8, 4, 2, 1]
cl, cr, ch = 0.865, 2.065, 0.47
for k, y in enumerate(rows):
    # card with shadow and graded accent strip
    ax.add_patch(FancyBboxPatch((cl + 0.016, y - ch / 2 - 0.016), cr - cl, ch,
                 boxstyle='round,pad=0,rounding_size=0.04', ec='none', fc='black',
                 alpha=0.13, zorder=1.5))
    ax.add_patch(FancyBboxPatch((cl, y - ch / 2), cr - cl, ch,
                 boxstyle='round,pad=0,rounding_size=0.04', ec=E_LIGHT, fc='white',
                 lw=0.6, zorder=1.6))
    ax.add_patch(Rectangle((cl, y - ch / 2 + 0.03), 0.045, ch - 0.06, ec='none',
                           fc=ENC_ST[k], zorder=1.7))
    s_ = face[spatial[k]]
    d_ = depth[chans[k]]
    cuboid(0.96, y - s_ / 2 - 0.03, s_, s_, d_, ENC_ST[k], ec='#24484A', grid=4)
    ax.text(1.34, y + 0.14, r'Stage %d  $E_%d$' % (k + 1, k + 1), ha='left', va='center',
            fontsize=6.5, weight='bold', color=E_DARK, zorder=4)
    ax.text(1.34, y, r'OPE$\downarrow$%s, SRA $R{=}%d$' % (down[k], sr[k]),
            ha='left', va='center', fontsize=6, color=NEU, zorder=4)
    ax.text(1.34, y - 0.14, '{0}×{0}×{1}'.format(spatial[k], chans[k]),
            ha='left', va='center', fontsize=6, zorder=4)
    if k < 3:
        path(ax, [(1.10, y + ch / 2), (1.10, rows[k + 1] - ch / 2)], color=E_DARK,
             hw=0.035, hl=0.045, z=2)

# ------------------------------------------------------------ channel projections
px = 2.24
ax.text(2.42, 3.40, 'Projection', ha='center', va='bottom', fontsize=7, weight='bold')
ax.text(2.40, 3.25, r'$P_i$, 64 ch', ha='center', va='bottom', fontsize=6.5,
        color=NEU)
for k, y in enumerate(rows):
    s = face[spatial[k]] * 0.8
    path(ax, [(cr, y), (px - 0.01, y)], hw=0.035, hl=0.045)
    cuboid(px, y - s / 2, s, s, 0.07, '#C3CCD5')
    ax.text(px + s / 2 + 0.03, y + s / 2 + 0.07, r'$F_{}$'.format(k + 1), ha='center',
            va='bottom', fontsize=7)
    ax.plot([px + s + 0.03, 2.70], [y, y], color=NEU, lw=0.6, zorder=2)
bx = 2.70
ax.plot([bx, bx], [rows[0], rows[-1]], color=NEU, lw=0.6)
yS, yR, yC = 2.50, 1.83, 1.16
path(ax, [(bx, yR), (2.78, yR), (2.78, yS - 0.12), (2.86, yS - 0.12)], color=SEM, hw=0.035, hl=0.045)
path(ax, [(2.78, yR), (2.78, yC + 0.12), (2.86, yC + 0.12)], color=CON, hw=0.035, hl=0.045)
dot(ax, 2.78, yR)

# ------------------------------------------------------------ dual-stream decoder
cols = [3.55, 4.33, 5.11, 5.89]
lvl = [4, 3, 2, 1]
sp = [11, 22, 44, 88]
x0, x1 = 2.86, 6.16
for (yc, col) in [(yS, SEM), (yC, CON)]:
    ax.add_patch(FancyBboxPatch((x0, yc - 0.27), x1 - x0, 0.54,
                 boxstyle='round,pad=0,rounding_size=0.04', ec=LANE_EDGE, fc=LANE_FILL,
                 lw=0.6, zorder=0.5))
    ax.add_patch(Rectangle((x0, yc - 0.27), 0.035, 0.54, fc=col, ec='none', zorder=0.6))
ax.text(3.07, yS, 'Semantic\nstream', rotation=90, ha='center', va='center',
        fontsize=6.5, color=SEM, weight='bold', linespacing=1.05)
ax.text(3.07, yC, 'Contour\nstream', rotation=90, ha='center', va='center',
        fontsize=6.5, color=CON, weight='bold', linespacing=1.05)
ax.text(2.90, yR + 0.07, 'Identity-preserving', ha='left', va='center', fontsize=6,
        color=NEU)
ax.text(2.90, yR - 0.07, r'readout $\tilde{S}_i = S_i + g_i C_i$', ha='left', va='center',
        fontsize=6, color=NEU)

path(ax, [(2.95, 3.20), (6.25, 3.20)], lw=0.8)
ax.text(4.60, 3.27, 'Top-down progressive fusion (coarse → fine)', ha='center',
        va='bottom', fontsize=7, weight='bold')
for c, s, st in zip(cols, sp, [32, 16, 8, 4]):
    ax.text(c, 3.06, r'$s{=}%d$, %d$\times$%d$\times$64' % (st, s, s), ha='center',
            va='center', fontsize=6, color=NEU)

for j, (c, i) in enumerate(zip(cols, lvl)):
    for (yc, col, sup, tag, tdy) in [(yS, SEM, 's', 'F', +0.33), (yC, CON, 'c', 'D', -0.33)]:
        node(c, yc, r'$R^{%s}_{%d}$' % (sup, i), col)
        if j == 0:
            ax.text(c, yc + tdy, r'$%s_%d$' % (tag, i), ha='center', va='center',
                    fontsize=7, color=col)
            path(ax, [(c, yc + tdy - np.sign(tdy) * 0.07), (c, yc + np.sign(tdy) * 0.14)],
                 color=col, hw=0.035, hl=0.045)
        else:
            ox = c - 0.36
            op(ax, ox, yc, '+', ec=col, r=0.06)
            ax.text(ox, yc + tdy, r'$%s_%d$' % (tag, i), ha='center', va='center',
                    fontsize=7, color=col)
            path(ax, [(ox, yc + tdy - np.sign(tdy) * 0.07), (ox, yc + np.sign(tdy) * 0.06)],
                 color=col, hw=0.035, hl=0.045)
            prev = cols[j - 1] + 0.19
            path(ax, [(prev, yc), (ox - 0.06, yc)], color=col, hw=0.035, hl=0.045)
            ax.text((prev + ox) / 2 - 0.02, yc + (0.06 if col == SEM else -0.06),
                    r'$U_{%d}$' % i, ha='center', va='bottom' if col == SEM else 'top',
                    fontsize=6, color=col)
            path(ax, [(ox + 0.06, yc), (c - 0.19, yc)], color=col, hw=0.03, hl=0.04)
    if i <= 3:
        bxr = c + 0.08
        op(ax, bxr, yR, '+', ec='black', r=0.06)
        path(ax, [(bxr, yS - 0.14), (bxr, yR + 0.06)], color=SEM, hw=0.035, hl=0.045)
        ax.text(bxr - 0.03, yS - 0.32, r'$S_%d$' % i, ha='right', va='center',
                fontsize=6.5, color=SEM)
        path(ax, [(bxr, yC + 0.14), (bxr, yR - 0.06)], color=CON, hw=0.035, hl=0.045)
        ax.text(bxr + 0.03, yC + 0.32, r'$\times g_%d$' % i, ha='left', va='center',
                fontsize=6.5, color=CON)
        hx = c + 0.33
        box(ax, hx, yR, 0.17, 0.18, 'h', ec=DARK, fc=DARK, fs=6.5, color='white', shadow=True)
        path(ax, [(bxr + 0.06, yR), (hx - 0.085, yR)], color='black', hw=0.03, hl=0.04)
        if i > 1:
            path(ax, [(hx + 0.085, yR), (hx + 0.18, yR)], color='black', hw=0.03, hl=0.04)
            ax.text(hx + 0.19, yR, r'$z_%d$' % i, ha='left', va='center', fontsize=7)

# ------------------------------------------------------------ output
sx = 6.62
hx1 = cols[-1] + 0.33
op(ax, sx, yR, r'$\Sigma$', ec='black', r=0.08, fs=6.5)
path(ax, [(hx1 + 0.085, yR), (sx - 0.08, yR)], color='black')
ax.text((hx1 + sx) / 2 + 0.03, yR + 0.06, r'$z_1$', ha='center', va='bottom', fontsize=7)
ax.text(sx, yR + 0.36, r'$z_2,\ z_3$', ha='center', va='center', fontsize=7)
path(ax, [(sx, yR + 0.28), (sx, yR + 0.08)], color='black', hw=0.035, hl=0.045)
box(ax, sx, 1.45, 0.28, 0.20, r'$\sigma$', ec=DARK, fc=DARK, fs=8, color='white', shadow=True)
path(ax, [(sx, yR - 0.08), (sx, 1.55)], color='black', hw=0.035, hl=0.045)
path(ax, [(sx, 1.35), (sx, 1.25)], color='black', hw=0.035, hl=0.045)
cuboid(sx - 0.15, 0.97, 0.26, 0.26, 0.03, '#E3E3E3')
ax.text(sx - 0.02, 1.10, 'p', ha='center', va='center', fontsize=7.5, style='italic', zorder=5)
ax.text(sx, 0.93, '1×352×352\nprobability', ha='center', va='top', fontsize=6,
        linespacing=1.1)
ax.text(sx, 3.00, '3.483 M params\n1.686 GMACs', ha='center', va='center', fontsize=6,
        color=NEU, linespacing=1.1)


# ------------------------------------------------------------ detail insets
def inset(xa, xb, title, col, col2=None):
    w = xb - xa
    ax.add_patch(FancyBboxPatch((xa, 0.0), w, 0.58, boxstyle='round,pad=0,rounding_size=0.04',
                 ec='none', fc='#FBFBFC', zorder=0.4))
    if col2 is None:
        ax.add_patch(FancyBboxPatch((xa, 0.445), w, 0.135,
                     boxstyle='round,pad=0,rounding_size=0.04', ec='none', fc=col, zorder=0.45))
        ax.add_patch(Rectangle((xa, 0.445), w, 0.06, ec='none', fc=col, zorder=0.46))
    else:                                           # shared by both streams: split bar
        for (x_, c_) in [(xa, col), (xa + w / 2, col2)]:
            ax.add_patch(FancyBboxPatch((x_, 0.445), w / 2, 0.135,
                         boxstyle='round,pad=0,rounding_size=0.04', ec='none', fc=c_, zorder=0.45))
        ax.add_patch(Rectangle((xa + 0.05, 0.445), w - 0.10, 0.135, ec='none', fc=col, zorder=0.46))
        ax.add_patch(Rectangle((xa + w / 2, 0.445), w / 2 - 0.05, 0.135, ec='none', fc=col2, zorder=0.46))
        ax.add_patch(Rectangle((xa, 0.445), w, 0.06, ec='none', fc=col, zorder=0.46))
        ax.add_patch(Rectangle((xa + w / 2, 0.445), w / 2, 0.06, ec='none', fc=col2, zorder=0.47))
    ax.add_patch(FancyBboxPatch((xa, 0.0), w, 0.58, boxstyle='round,pad=0,rounding_size=0.04',
                 ec='#B5B5B5', fc='none', lw=0.6, zorder=0.5))
    ax.text(xa + 0.06, 0.512, title, ha='left', va='center', fontsize=6.5, weight='bold',
            color='white', zorder=0.6)


def ibox(cx, cy, w, text, ec, fc):
    box(ax, cx, cy, w, 0.25, text, ec=ec, fc=fc, fs=6, lw=0.7, shadow=True)


# (i) encoder stage: teal family, graded by depth of processing
inset(0.05, 2.78, r'Encoder stage $E_i$', E_DARK)
ax.text(0.12, 0.21, 'in', ha='left', va='center', fontsize=6)
box(ax, 0.66, 0.21, 0.70, 0.25, 'overlapping\npatch embedding', ec=E_MID, fc=E_TINT,
    fs=6, lw=0.7, shadow=True)
ax.add_patch(FancyBboxPatch((1.06, 0.045), 1.40, 0.33, boxstyle='round,pad=0,rounding_size=0.03',
             ec=E_MID, fc='#F1F8F7', lw=0.6, ls=(0, (2.5, 1.5)), zorder=1))
box(ax, 1.46, 0.21, 0.70, 0.25, 'spatial-reduction\nattention', ec=E_DARK, fc=E_DARK,
    fs=6, lw=0.7, color='white', shadow=True)
box(ax, 2.15, 0.21, 0.50, 0.25, 'Mix-FFN', ec=E_MID, fc=E_MID, fs=6, lw=0.7,
    color='white', shadow=True)
path(ax, [(0.22, 0.21), (0.31, 0.21)], hw=0.03, hl=0.04)
path(ax, [(1.01, 0.21), (1.11, 0.21)], hw=0.03, hl=0.04)
path(ax, [(1.81, 0.21), (1.90, 0.21)], hw=0.03, hl=0.04)
ax.text(2.53, 0.21, r'$\times 2$', ha='left', va='center', fontsize=7, color=E_DARK, weight='bold')

# (ii) refinement, shared by both streams: two-tone slate
inset(2.86, 4.62, r'Depthwise-separable refinement $R_i$', R_DARK)
ax.text(2.93, 0.21, 'in', ha='left', va='center', fontsize=6)
box(ax, 3.47, 0.21, 0.76, 0.25, 'depthwise spatial\nmixing', ec=R_DARK, fc=R_DARK, fs=6,
    lw=0.7, color='white', shadow=True)
box(ax, 4.24, 0.21, 0.66, 0.25, 'pointwise\nchannel mixing', ec=R_MID, fc=R_MID, fs=6,
    lw=0.7, color='white', shadow=True)
path(ax, [(3.03, 0.21), (3.09, 0.21)], hw=0.03, hl=0.04)
path(ax, [(3.85, 0.21), (3.91, 0.21)], hw=0.03, hl=0.04)

# (iii) high-frequency residual (all arrowheads visible)
inset(4.68, 5.98, r'High-frequency residual', CON)
ax.text(4.73, 0.31, r'$F_i$', ha='left', va='center', fontsize=7)
dot(ax, 4.87, 0.31)
box(ax, 5.30, 0.12, 0.58, 0.17, r'local mean $A_3$', ec=CON, fc=CON_FILL, fs=6, lw=0.7,
    shadow=True)
op(ax, 5.70, 0.31, '−', ec=CON, r=0.055)
path(ax, [(4.85, 0.31), (5.645, 0.31)], hw=0.03, hl=0.04)
path(ax, [(4.87, 0.31), (4.87, 0.12), (5.01, 0.12)], hw=0.03, hl=0.04, z=3)
path(ax, [(5.59, 0.12), (5.70, 0.12), (5.70, 0.255)], hw=0.03, hl=0.04, z=3)
path(ax, [(5.755, 0.31), (5.83, 0.31)], color=CON, hw=0.03, hl=0.04)
ax.text(5.845, 0.31, r'$D_i$', ha='left', va='center', fontsize=7, color=CON)

# (iv) bounded gain
inset(6.04, 6.96, r'Bounded gain $g_i$', CON)
gax = fig.add_axes([6.29 / W, 0.14 / H, 0.60 / W, 0.26 / H])
gax.patch.set_alpha(0)
a = np.linspace(-3, 3, 200)
gax.plot(a, 0.25 * np.tanh(a), color=CON, lw=1.1)
for v in (0.25, -0.25):
    gax.axhline(v, color=NEU, lw=0.5, ls=(0, (2, 1.5)))
gax.plot([0], [0], 'o', ms=2.6, color=DARK)
gax.set_xticks([0]); gax.set_xticklabels([r'$a_i{=}0$'], fontsize=6)
gax.set_yticks([-0.25, 0.25]); gax.set_yticklabels([r'$-\frac{1}{4}$', r'$\frac{1}{4}$'], fontsize=6)
gax.tick_params(length=1.5, pad=1)
for s_ in ('top', 'right'):
    gax.spines[s_].set_visible(False)
gax.set_ylim(-0.33, 0.33)

save(fig, 'fig2.png')
