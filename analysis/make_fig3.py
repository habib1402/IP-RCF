"""Fig. 3 (alternative design): contour correction at one readout scale and its
training supervision. Writes figures/fig3_alt.png; make_fig3.py is left untouched.

Every formula is taken from the implementation:
  lib/iprcf.py           local_contour_residual, _contour_path, forward
                          (corrected feature -> bilinear U_x to input grid -> shared 1x1 head)
  TrainFRBPFFStudent.py  _boundary_target (max-pool dilation - erosion, square (2r+1)^2),
                          _one_boundary_loss (0.5*[BCE with pos_weight clip(N-/N+,1,20)
                          + soft Dice over non-empty bands])
  TrainIPRCFStudent.py   _contour_consistency (mean smooth-L1 between sigmoid of adjacent
                          boundary logits, already at input resolution); total loss
                          averaged over clean and artifact-simulated views
  TrainMPECStudent.py / Train.py   L_seg = sum_i L_str(z_i) + 0.5 L_str(sum_i z_i),
                          L_str = weighted BCE + weighted IoU, w = 1 + 5|A_31(y) - y|
Panel (b) thumbnails use one Kvasir training image and its annotation.
"""
import numpy as np
import cv2
from figstyle import *

TRAIN = DATA_ROOT + '/TrainDataset'
EXAMPLE = 'cju1b3zgj3d8e0801kpolea6c.png'
DARK = '#2F2F2F'
SLATE = '#3E4A56'
DASH = (0, (3.2, 2.0))

W, H = FULL_W, 3.45
CROP = 0.31
fig, ax = canvas(W, H - CROP)
ax.set_ylim(CROP, H)


def solid(cx, cy, w, h, text, col, fs=8):
    box(ax, cx, cy, w, h, text, ec=col, fc=col, fs=fs, color='white', r=0.02, shadow=True)


def dashed(cx, cy, w, h, text, fs=7.5):
    box(ax, cx, cy, w, h, text, ec=AUX, fc='white', ls=DASH, lw=0.8, fs=fs, color=AUX, r=0.02)


def kernel(cx, cy, cell, values, cols, fs=5.6, txtcol=None):
    """3x3 kernel drawn as a grid of cells centred at (cx, cy)."""
    for r in range(3):
        for c in range(3):
            x = cx - 1.5 * cell + c * cell
            y = cy + 0.5 * cell - r * cell
            ax.add_patch(Rectangle((x, y), cell, cell, fc=cols[r][c], ec='white', lw=0.6,
                                   zorder=3))
            ax.text(x + cell / 2, y + cell / 2, values[r][c], ha='center', va='center',
                    fontsize=fs, color=(txtcol or [['black'] * 3] * 3)[r][c], zorder=4)
    ax.add_patch(Rectangle((cx - 1.5 * cell, cy - 1.5 * cell), 3 * cell, 3 * cell,
                           fill=False, ec=SLATE, lw=0.6, zorder=4))


def shape(x, y, s, ha='center'):
    ax.text(x, y, s, ha=ha, va='top', fontsize=6, color='#6A6A6A')


# ================================================================ (a) inference
ax.text(0.05, 3.40, '(a)', ha='left', va='top', fontsize=8, weight='bold')
ym, yt = 2.66, 3.08

ax.text(0.22, ym, r'$F_i$', ha='center', va='center', fontsize=9)
shape(0.22, ym - 0.12, r'64$\times H_i\times W_i$')
dot(ax, 0.48, ym)
path(ax, [(0.34, ym), (1.335, ym)])

# A_3: 3x3 local mean (uniform kernel)
k3 = 0.105
light = '#E3E7EC'
kernel(0.88, 2.18, k3, [['1'] * 3] * 3, [[light] * 3] * 3, fs=6)
ax.text(0.88, 2.18 + 1.5 * k3 + 0.03, r'$A_3 = \frac{1}{9}\,\times$', ha='center', va='bottom', fontsize=7.5,
        color=SLATE)
path(ax, [(0.48, ym), (0.48, 2.18), (0.88 - 1.5 * k3 - 0.005, 2.18)], hw=0.035, hl=0.045)
path(ax, [(0.88 + 1.5 * k3, 2.18), (1.40, 2.18), (1.40, ym - 0.065)], hw=0.035, hl=0.045)
op(ax, 1.40, ym, '−', ec=CON)

# equivalent high-pass kernel of (I - A_3)
hp = [['−1'] * 3, ['−1', '8', '−1'], ['−1'] * 3]
hc = [['#DCE6F0'] * 3, ['#DCE6F0', CON, '#DCE6F0'], ['#DCE6F0'] * 3]
ht = [['black'] * 3, ['black', 'white', 'black'], ['black'] * 3]
kernel(1.86, 2.18, 0.105, hp, hc, fs=6, txtcol=ht)
ax.text(1.86, 2.18 + 0.19, r'$I - A_3 = \frac{1}{9}\,\times$', ha='center', va='bottom', fontsize=7.5, color=CON)
ax.text(1.86, 2.18 - 0.16 - 0.03, r'$D_i = (I - A_3)\,F_i$', ha='center', va='top',
        fontsize=7, color=CON)

ax_ = 2.36
path(ax, [(1.465, ym), (ax_ - 0.065, ym)], color=CON)
ax.text((1.465 + ax_) / 2, ym + 0.06, r'$D_i$', ha='center', va='bottom', fontsize=8,
        color=CON)
op(ax, ax_, ym, '+', ec=CON)
ax.text(ax_, yt, r'$U_i(C_{i+1})$', ha='center', va='center', fontsize=8, color=CON)
path(ax, [(ax_, yt - 0.09), (ax_, ym + 0.065)], color=CON)

rx = 2.92
solid(rx, ym, 0.46, 0.32, r'$R^c_i$', CON, fs=9)
path(ax, [(ax_ + 0.065, ym), (rx - 0.23, ym)], color=CON)
ax.text(rx, ym - 0.24, r'$W^{\mathrm{dw}}_{3\times3} \rightarrow W^{\mathrm{pw}}_{1\times1}$',
        ha='center', va='top', fontsize=6.5, color=CON)

cx = 3.42
path(ax, [(rx + 0.23, ym), (cx, ym)], color=CON, head=False)
dot(ax, cx, ym, color=CON)
ax.text(cx - 0.05, ym + 0.06, r'$C_i$', ha='right', va='bottom', fontsize=8, color=CON)
path(ax, [(cx, ym), (cx, yt - 0.10)], color=CON)
ax.text(cx, yt, r'$U_{i-1}$', ha='center', va='center', fontsize=7.5, color=CON)

gx = 3.98
op(ax, gx, ym, '×', ec=CON)
path(ax, [(cx, ym), (gx - 0.065, ym)], color=CON)
ax.text(gx, yt, r'$g_i = \frac{1}{4}\tanh a_i$', ha='center', va='center', fontsize=8,
        color=CON)
path(ax, [(gx, yt - 0.11), (gx, ym + 0.065)], color=CON)
ax.text(gx, ym - 0.12, r'$|g_i| < \frac{1}{4}$', ha='center', va='top', fontsize=6.5,
        color=CON)

sx = 4.60
op(ax, sx, ym, '+', ec='black')
path(ax, [(gx + 0.065, ym), (sx - 0.065, ym)], color=CON)
ax.text((gx + sx) / 2, ym + 0.06, r'$g_i C_i$', ha='center', va='bottom', fontsize=8,
        color=CON)
solid(sx, yt, 0.42, 0.24, r'$S_i$', SEM, fs=8.5)
path(ax, [(sx, yt - 0.12), (sx, ym + 0.065)], color=SEM)

ux = 5.12
box(ax, ux, ym, 0.30, 0.26, r'$U_x$', ec=SLATE, fc='#ECEEF1', fs=8, r=0.02, shadow=True)
path(ax, [(sx + 0.065, ym), (ux - 0.15, ym)], color='black')
ax.text((sx + ux - 0.15) / 2, ym + 0.06, r'$\tilde{S}_i$', ha='center', va='bottom',
        fontsize=8)
hx = 5.56
solid(hx, ym, 0.26, 0.26, 'h', DARK, fs=8)
path(ax, [(ux + 0.15, ym), (hx - 0.13, ym)], color='black', hw=0.035, hl=0.045)
shape(ux, ym - 0.16, '352×352')
path(ax, [(hx + 0.13, ym), (5.88, ym)], color='black')
ax.text(5.91, ym, r'$z_i$', ha='left', va='center', fontsize=9)
shape(5.96, ym - 0.13, r'1$\times$352$\times$352')

# analytical notes
ax.plot([6.26, 6.26], [2.05, 3.12], color='#C8C8C8', lw=0.6)
nx = 6.33
ax.text(nx, 2.86, r'$a_i = 0 \rightarrow \tilde{S}_i = S_i$', ha='left', va='center',
        fontsize=7.5)
ax.text(nx, 2.46, r'$z_i = h(U_x S_i)$', ha='left', va='center', fontsize=7.5)
ax.text(nx + 0.08, 2.30, r'$+\, g_i\, w^{\mathrm{T}} U_x C_i$', ha='left', va='center',
        fontsize=7.5, color=CON)
ax.text(nx, 2.12, r'$h(v) = w^{\mathrm{T}} v + b$', ha='left', va='center', fontsize=6.5,
        color='#6A6A6A')

# ================================================================ (b) training only
ax.plot([0.05, 6.95], [1.86, 1.86], color='#D5D5D5', lw=0.6)
ax.text(0.05, 1.80, '(b)', ha='left', va='top', fontsize=8, weight='bold')
yb = 1.30

path(ax, [(cx, ym - 0.03), (cx, yb + 0.13)], color=AUX, ls=DASH)
dashed(cx, yb, 0.30, 0.26, r'$\varphi$', fs=9)
ax.text(cx, yb - 0.19, r'$b_i = U_x\,\varphi(C_i)$', ha='center', va='top', fontsize=7,
        color=AUX)

lx = 4.42
dashed(lx, yb, 0.56, 0.28, r'$L_{\mathrm{bnd}}$', fs=8.5)
path(ax, [(cx + 0.15, yb), (lx - 0.28, yb)], color=AUX, ls=DASH)
ax.text((cx + 0.15 + lx - 0.28) / 2, yb + 0.05, r'$b_i$', ha='center', va='bottom',
        fontsize=8, color=AUX)
ax.text(lx, yb - 0.19,
        r'$\frac{1}{2}\left[\mathrm{BCE}_{w^{+}}(b_i, B) + \mathrm{Dice}(\sigma(b_i), B)\right]$',
        ha='center', va='top', fontsize=7, color=AUX)
ax.text(lx, yb - 0.38, r'$w^{+} = \mathrm{clip}(N^{-}/N^{+},\, 1,\, 20)$', ha='center',
        va='top', fontsize=6.5, color='#6A6A6A')

kx = 1.70
dashed(kx, yb, 0.56, 0.28, r'$L_{\mathrm{cons}}$', fs=8.5)
path(ax, [(cx - 0.15, yb), (kx + 0.28, yb)], color=AUX, ls=DASH)
ax.text(0.40, yb, r'$b_{i+1}$', ha='center', va='center', fontsize=8.5, color=AUX)
path(ax, [(0.55, yb), (kx - 0.28, yb)], color=AUX, ls=DASH)
ax.text(kx, yb - 0.19, r'smooth-$L_1\left(\sigma(b_i),\, \sigma(b_{i+1})\right)$',
        ha='center', va='top', fontsize=7, color=AUX)

# real training example and its morphological band target (square SE, as in code)
im = cv2.cvtColor(cv2.imread('{}/images/{}'.format(TRAIN, EXAMPLE)), cv2.COLOR_BGR2RGB)
gt = cv2.imread('{}/masks/{}'.format(TRAIN, EXAMPLE), 0) > 127
im = cv2.resize(im, (352, 352), interpolation=cv2.INTER_AREA)
gt = cv2.resize(gt.astype(np.uint8), (352, 352), interpolation=cv2.INTER_NEAREST)
se = np.ones((5, 5), np.uint8)                                   # (2r+1)^2, r = 2
band = cv2.dilate(gt, se) - cv2.erode(gt, se)
over = im.copy()
cs, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(over, cs, -1, (255, 255, 255), 3, lineType=cv2.LINE_AA)
aux_rgb = np.array([int(AUX[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
tgt = np.full((352, 352, 3), 255, np.uint8)
tgt[gt > 0] = (226, 226, 226)
tgt[band > 0] = aux_rgb
sz = 0.60
bxc, ixc = 5.48, 6.50
for img, c_ in ((tgt, bxc), (over, ixc)):
    ax.imshow(img, extent=(c_ - sz / 2, c_ + sz / 2, yb - sz / 2 + 0.05, yb + sz / 2 + 0.05),
              interpolation='nearest' if img is tgt else 'lanczos', zorder=2)
    ax.add_patch(Rectangle((c_ - sz / 2, yb - sz / 2 + 0.05), sz, sz, fill=False,
                           ec='#3A3A3A', lw=0.5, zorder=3))
ax.text(bxc, yb - sz / 2 + 0.01, r'$B(y)$', ha='center', va='top', fontsize=8, color=AUX)
ax.text(ixc, yb - sz / 2 + 0.01, r'$x,\ y$', ha='center', va='top', fontsize=8)
path(ax, [(ixc - sz / 2 - 0.02, yb + 0.05), (bxc + sz / 2 + 0.03, yb + 0.05)], color=AUX,
     ls=DASH, hw=0.035, hl=0.045)
ax.text((ixc + bxc) / 2, yb - 0.03, r'$r = 2$', ha='center', va='top', fontsize=7, color=AUX)
ax.text((ixc + bxc) / 2, yb + 0.11, r'$\delta_r - \varepsilon_r$', ha='center',
        va='bottom', fontsize=7.5, color=AUX)
path(ax, [(bxc - sz / 2 - 0.02, yb), (lx + 0.28, yb)], color=AUX, ls=DASH)

# ================================================================ objective
ax.text(3.50, 0.66,
        r'$L = \frac{1}{2}\Sigma_{v\,=\,\mathrm{clean,\,artifact}}\left[L^{v}_{\mathrm{seg}}'
        r' + \lambda_b\, \Sigma_i\, L^{v}_{\mathrm{bnd}} + \lambda_c\, L^{v}_{\mathrm{cons}}\right],'
        r'\quad \lambda_b = 0.25,\ \lambda_c = 0.10$',
        ha='center', va='center', fontsize=7.5)
ax.text(3.50, 0.45,
        r'$L_{\mathrm{seg}} = \Sigma_i\, L_{\mathrm{str}}(z_i, y) + \frac{1}{2} L_{\mathrm{str}}'
        r'(\Sigma_i z_i,\, y),\quad L_{\mathrm{str}} = \mathrm{wBCE} + \mathrm{wIoU},'
        r'\quad w = 1 + 5\,|A_{31}(y) - y|$',
        ha='center', va='center', fontsize=7.5)

save(fig, 'fig3_alt.png')
