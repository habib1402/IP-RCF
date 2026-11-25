"""Fig. 4: within-model contour-gain intervention on the frozen IP-RCF checkpoint (epoch 95).

Sources (all on the same 798 development images, threshold 0.5):
  analyses/iprcf_gain_intervention_seed42/learned  CPU re-run with fitted gains
                                                   (reproduces the frozen GPU macro Dice to 1e-6)
  analyses/iprcf_gain_intervention_seed42/zero     same weights, a_i = 0 so g_i = 0
  results/tmi_mpec_data_control_c64_seed42/predictions_raw_sigmoid   matched Plain-PFF
(a) Per-dataset Dice of the three predictors; macro with 95% bootstrap intervals of the paired
    differences (images resampled within datasets).
(b) Example: ETIS image with the largest Dice gain from the fitted gains (learned - zero).
"""
import os, csv, json
import numpy as np
import cv2
from matplotlib.colors import TwoSlopeNorm
from figstyle import *

ST = STUDY_ROOT
GI = ST + '/analyses/iprcf_gain_intervention_seed42'
SRC = {'learned': GI + '/learned', 'zero': GI + '/zero',
       'plain': ST + '/results/tmi_mpec_data_control_c64_seed42/predictions_raw_sigmoid'}
TEST = DATA_ROOT + '/TestDataset'
DS = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
SHORT = {'CVC-300': 'CVC-300', 'CVC-ClinicDB': 'ClinicDB', 'Kvasir': 'Kvasir',
         'CVC-ColonDB': 'ColonDB', 'ETIS-LaribPolypDB': 'ETIS'}


def load(src):
    path = src + '/per_image_metrics.csv'
    return {(r['dataset'], r['name']): float(r['dice']) for r in csv.DictReader(open(path))}


D = {k: load(v) for k, v in SRC.items()}
keys = sorted(D['plain'])
assert set(keys) == set(D['learned']) == set(D['zero'])
per = {k: {ds: np.mean([D[k][x] for x in keys if x[0] == ds]) for ds in DS} for k in D}
macro = {k: np.mean([per[k][ds] for ds in DS]) for k in D}

rng = np.random.default_rng(3)
def macro_ci(a, b):
    out = []
    arr = {ds: np.array([D[a][x] - D[b][x] for x in keys if x[0] == ds]) for ds in DS}
    for _ in range(2000):
        out.append(np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in arr.values()]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
ci_corr = macro_ci('learned', 'zero')        # effect of the fitted correction within the model
ci_sem = macro_ci('zero', 'plain')           # semantic stream of IP-RCF vs matched Plain-PFF
ci_tot = macro_ci('learned', 'plain')

# example: ETIS image with the largest gain from the fitted gains
etis = [x for x in keys if x[0] == 'ETIS-LaribPolypDB']
ex = max(etis, key=lambda x: (D['learned'][x] - D['zero'][x], x[1]))

W, H = COL_W, 3.45
fig = plt.figure(figsize=(W, H))
axa = fig.add_axes([0.70 / W, 1.66 / H, 2.65 / W, 1.58 / H])
rows = DS + ['macro']
y = np.arange(len(rows))[::-1]
for yy, r in zip(y, rows):
    v = {k: (macro[k] if r == 'macro' else per[k][r]) for k in D}
    axa.plot([v['zero'], v['learned']], [yy, yy], color=CON, lw=0.8, zorder=1)
    axa.plot(v['zero'], yy, 'D', mfc='white', mec=CON, mew=0.9, ms=4, zorder=3)
    axa.plot(v['learned'], yy, 'D', color=CON, ms=4, zorder=3)
    axa.plot(v['plain'], yy, 'o', color=SEM, ms=4, zorder=3)
axa.set_yticks(y)
axa.set_yticklabels([SHORT.get(r, 'Macro') for r in rows])
axa.set_xlabel('Dice', labelpad=1)
for s_ in ('top', 'right'):
    axa.spines[s_].set_visible(False)
from matplotlib.lines import Line2D
hs = [Line2D([], [], marker='o', color=SEM, ls='none', ms=4),
      Line2D([], [], marker='D', mfc='white', mec=CON, ls='none', ms=4),
      Line2D([], [], marker='D', color=CON, ls='none', ms=4)]
axa.legend(hs, ['Plain-PFF', r'IP-RCF, $g_i = 0$', 'IP-RCF, fitted $g_i$'], loc='lower right',
           frameon=False, fontsize=6.2, handletextpad=0.2, borderaxespad=0.1)
fig.text(0.04 / W, 3.40 / H, '(a)', fontsize=8, weight='bold', va='top')

# (b) example panels
rgb = cv2.cvtColor(cv2.imread('{}/{}/images/{}'.format(TEST, *ex)), cv2.COLOR_BGR2RGB)
g = cv2.imread('{}/{}/masks/{}'.format(TEST, *ex), 0) > 127
pl = cv2.imread('{}/{}/{}'.format(SRC['learned'], *ex), 0).astype(float) / 255
pz = cv2.imread('{}/{}/{}'.format(SRC['zero'], *ex), 0).astype(float) / 255
ys, xs = np.where(g | (pl >= 0.5) | (pz >= 0.5))
side = int(min(max(ys.ptp(), xs.ptp()) * 1.35 + 20, min(g.shape)))
y0 = int(np.clip((ys.min() + ys.max()) / 2 - side / 2, 0, g.shape[0] - side))
x0 = int(np.clip((xs.min() + xs.max()) / 2 - side / 2, 0, g.shape[1] - side))
cr = lambda a: a[y0:y0 + side, x0:x0 + side]
over = cr(rgb).copy()
cs, _ = cv2.findContours(cr(g).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
lw = max(1, int(round(side / 180)))
cv2.drawContours(over, cs, -1, (0, 0, 0), lw + 2, lineType=cv2.LINE_AA)
cv2.drawContours(over, cs, -1, (255, 255, 255), lw, lineType=cv2.LINE_AA)
cell, gap, x0f = 0.78, 0.03, 0.14
heads = ['input, $y$', r'$p$, fitted $g_i$', r'$p$, $g_i = 0$', r'$\Delta p$']
dp = cr(pl - pz)
m = max(np.abs(dp).max(), 1e-3)
norm = TwoSlopeNorm(0, -m, m)
for c, (img, hd) in enumerate(zip([over, cr(pl), cr(pz), dp], heads)):
    ax = fig.add_axes([(x0f + c * (cell + gap)) / W, 0.36 / H, cell / W, cell / H])
    if c == 0:
        ax.imshow(img, interpolation='lanczos')
    elif c < 3:
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.contour(cr(g).astype(float), levels=[0.5], colors=[CON], linewidths=0.5)
    else:
        im_ = ax.imshow(img, cmap='RdBu_r', norm=norm, interpolation='nearest')
        ax.contour(cr(g).astype(float), levels=[0.5], colors='black', linewidths=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_linewidth(0.4); s_.set_color('#808080')
    ax.text(0.5, 1.03, hd, transform=ax.transAxes, ha='center', va='bottom', fontsize=6.5)
    if c in (1, 2):
        dv = D['learned' if c == 1 else 'zero'][ex]
        ax.text(0.04, 0.04, '{:.3f}'.format(dv), transform=ax.transAxes, fontsize=6,
                color='white', va='bottom', bbox=dict(fc='black', ec='none', alpha=0.6, pad=1))
cax = fig.add_axes([(x0f + 3 * (cell + gap)) / W, 0.20 / H, cell / W, 0.05 / H])
cb = fig.colorbar(im_, cax=cax, orientation='horizontal', ticks=[-m, 0, m])
cb.ax.set_xticklabels(['−{:.2f}'.format(m), '0', '{:.2f}'.format(m)], fontsize=5.8)
cb.outline.set_linewidth(0.4)
fig.text(0.04 / W, 1.33 / H, '(b)', fontsize=8, weight='bold', va='top')
fig.text(x0f / W, 0.20 / H, 'ETIS {}'.format(ex[1].replace('.png', '')), fontsize=6.2,
         color=NEU, va='center')

save(fig, 'fig4.png')
res = dict(per_dataset=per, macro=macro, ci_correction=ci_corr, ci_semantic_vs_plain=ci_sem,
           ci_total=ci_tot, example=list(ex),
           example_dice={k: D[k][ex] for k in D}, delta_p_scale=float(m),
           improved_by_correction={ds: int(sum(D['learned'][x] > D['zero'][x] + 1e-9
                                               for x in keys if x[0] == ds)) for ds in DS},
           degraded_by_correction={ds: int(sum(D['learned'][x] < D['zero'][x] - 1e-9
                                               for x in keys if x[0] == ds)) for ds in DS})
os.makedirs('outputs', exist_ok=True)
json.dump(res, open('outputs/fig4_intervention.json', 'w'), indent=1)
print(json.dumps({k: res[k] for k in ('macro', 'ci_correction', 'ci_semantic_vs_plain', 'ci_total',
                                      'example', 'example_dice', 'improved_by_correction',
                                      'degraded_by_correction')}, indent=1))
