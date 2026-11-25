"""Fig. 1 (evidence-based): frozen IP-RCF vs matched Plain-PFF on the 798 development images.

(a) Per-dataset paired Dice change (IP-RCF - Plain-PFF), 95% paired bootstrap CI over images,
    with counts of images improved / degraded.
(b) Paired change in predicted foreground area, normalised by annotated area.
(c) Example rows selected by a fixed rule from the paired per-image Dice change:
    ETIS largest loss, ETIS median change, ETIS largest gain, CVC-300 largest loss
    (images with an annotated polyp; ties broken by name). Columns: crop with annotation,
    Plain-PFF error map, IP-RCF error map, signed probability change with annotation.

Sources (raw sigmoid at native size, threshold 0.5):
  IP-RCF   results/tmi_iprcf_multiscale_c64_seed42/predictions  (ckpt SHA-256 2722e758...)
  Plain    results/tmi_mpec_data_control_c64_seed42/predictions_raw_sigmoid (SHA-256 86ae0c86...)
"""
import os, json
import numpy as np
import cv2
from matplotlib.colors import TwoSlopeNorm
from figstyle import *

STUDY = STUDY_ROOT + '/results'
TEST = DATA_ROOT + '/TestDataset'
IP = STUDY + '/tmi_iprcf_multiscale_c64_seed42/predictions'
PL = STUDY + '/tmi_mpec_data_control_c64_seed42/predictions_raw_sigmoid'
DS = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
SHORT = {'CVC-300': 'CVC-300', 'CVC-ClinicDB': 'ClinicDB', 'Kvasir': 'Kvasir',
         'CVC-ColonDB': 'ColonDB', 'ETIS-LaribPolypDB': 'ETIS'}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')


def gray(p):
    a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    return a if a.ndim == 2 else a[..., 0]


def dice(p, g):
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else 2.0 * (p & g).sum() / s


def boot(x, reps=2000, seed=0, f=np.mean):
    rng = np.random.default_rng(seed)
    n = len(x)
    m = np.array([f(x[rng.integers(0, n, n)]) for _ in range(reps)])
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


# ------------------------------------------------------------------ paired statistics
rows = []
for ds in DS:
    for n in sorted(os.listdir('{}/{}/masks'.format(TEST, ds))):
        g = gray('{}/{}/masks/{}'.format(TEST, ds, n)) > 127
        pi = gray('{}/{}/{}'.format(IP, ds, n)) >= 128
        pp = gray('{}/{}/{}'.format(PL, ds, n)) >= 128
        ga = max(int(g.sum()), 1)
        rows.append(dict(ds=ds, name=n, gt=int(g.sum()), d_ip=dice(pi, g), d_pl=dice(pp, g),
                         area=(int(pi.sum()) - int(pp.sum())) / ga))
stats = {}
for ds in DS + ['macro']:
    if ds == 'macro':
        dd = np.array([stats[k]['mean'] for k in DS])
        stats['macro'] = dict(mean=float(dd.mean()))
        continue
    R = [r for r in rows if r['ds'] == ds]
    d = np.array([r['d_ip'] - r['d_pl'] for r in R])
    a = np.array([r['area'] for r in R])
    lo, hi = boot(d)
    alo, ahi = boot(a, seed=1, f=np.median)
    stats[ds] = dict(n=len(R), mean=float(d.mean()), lo=float(lo), hi=float(hi),
                     better=int((d > 1e-9).sum()), worse=int((d < -1e-9).sum()),
                     area_mean=float(a.mean()), area_median=float(np.median(a)),
                     area_lo=float(alo), area_hi=float(ahi),
                     ip=float(np.mean([r['d_ip'] for r in R])),
                     pl=float(np.mean([r['d_pl'] for r in R])))
# macro CI: resample images within each dataset
rng = np.random.default_rng(7)
mac = []
for _ in range(2000):
    v = []
    for ds in DS:
        d = np.array([r['d_ip'] - r['d_pl'] for r in rows if r['ds'] == ds])
        v.append(d[rng.integers(0, len(d), len(d))].mean())
    mac.append(np.mean(v))
stats['macro'].update(lo=float(np.percentile(mac, 2.5)), hi=float(np.percentile(mac, 97.5)))


def pick(ds, mode):
    R = sorted([r for r in rows if r['ds'] == ds and r['gt'] > 0], key=lambda r: r['name'])
    d = [r['d_ip'] - r['d_pl'] for r in R]
    if mode == 'loss':
        return R[int(np.argmin(d))]
    if mode == 'gain':
        return R[int(np.argmax(d))]
    med = np.median(d)
    return R[int(np.argmin([abs(x - med) for x in d]))]


examples = [('ETIS\nlargest loss', pick('ETIS-LaribPolypDB', 'loss')),
            ('ETIS\nmedian change', pick('ETIS-LaribPolypDB', 'median')),
            ('ETIS\nlargest gain', pick('ETIS-LaribPolypDB', 'gain')),
            ('CVC-300\nlargest loss', pick('CVC-300', 'loss'))]

# ------------------------------------------------------------------ figure
W = FULL_W
cell, gap = 0.88, 0.03
nrow = len(examples)
top, bottom = 0.42, 0.40
Hh = top + bottom + nrow * cell + (nrow - 1) * gap
fig = plt.figure(figsize=(W, Hh))

# (a) paired Dice change
la_x, la_w = 0.62, 1.90
order = DS + ['macro']
labels = [SHORT[d] for d in DS] + ['Macro']
a_h = 1.70
a_y = Hh - top - a_h
axa = fig.add_axes([la_x / W, a_y / Hh, la_w / W, a_h / Hh])
y = np.arange(len(order))[::-1]
for yy, k in zip(y, order):
    s = stats[k]
    col = CON if k != 'macro' else 'black'
    axa.errorbar(s['mean'], yy, xerr=[[s['mean'] - s['lo']], [s['hi'] - s['mean']]], fmt='o',
                 color=col, ms=3.2, capsize=1.8, elinewidth=0.8, lw=0.8)
    if k != 'macro':
        axa.text(0.024, yy, '{}/{}'.format(s['better'], s['worse']), ha='left',
                 va='center', fontsize=6.5, color=NEU)
axa.axvline(0, color=NEU, lw=0.6)
axa.set_yticks(y)
axa.set_yticklabels(labels)
axa.set_xlim(-0.06, 0.046)
axa.set_xticks([-0.06, -0.04, -0.02, 0, 0.02])
axa.set_xticklabels(['−0.06', '−0.04', '−0.02', '0', '0.02'])
axa.set_xlabel(r'$\Delta$Dice (IP-RCF $-$ Plain-PFF)', labelpad=1)
axa.text(0.024, y[0] + 0.55, 'better/worse', ha='left', va='bottom', fontsize=6, color=NEU)
for s_ in ('top', 'right'):
    axa.spines[s_].set_visible(False)
fig.text(0.04 / W, (a_y + a_h + 0.05) / Hh, '(a)', fontsize=8, weight='bold', va='bottom')

# (b) predicted-area change
b_h = 1.36
b_y = bottom
axb = fig.add_axes([la_x / W, b_y / Hh, la_w / W, b_h / Hh])
yb = np.arange(len(DS))[::-1]
for yy, k in zip(yb, DS):
    s = stats[k]
    axb.errorbar(s['area_median'], yy, xerr=[[s['area_median'] - s['area_lo']],
                 [s['area_hi'] - s['area_median']]], fmt='s', color=CON, ms=3.0, capsize=1.8,
                 elinewidth=0.8)
axb.axvline(0, color=NEU, lw=0.6)
axb.set_yticks(yb)
axb.set_yticklabels([SHORT[d] for d in DS])
axb.set_xlabel(r'median $(|P_{\mathrm{IP}}| - |P_{\mathrm{Plain}}|)\,/\,|G|$', labelpad=1)
for s_ in ('top', 'right'):
    axb.spines[s_].set_visible(False)
fig.text(0.04 / W, (b_y + b_h + 0.05) / Hh, '(b)', fontsize=8, weight='bold', va='bottom')

# (c) paired examples
cx0 = 3.30
heads = ['Input + annotation', 'Plain-PFF', 'IP-RCF',
         r'$p_{\mathrm{IP}} - p_{\mathrm{Plain}}$']
fig.text((cx0 - 0.70) / W, (Hh - top + 0.20) / Hh, '(c)', fontsize=8, weight='bold', va='top')
norm = TwoSlopeNorm(0, -1, 1)
ex_prov = []
for r_, (lab, r) in enumerate(examples):
    ds, n = r['ds'], r['name']
    rgb = cv2.cvtColor(cv2.imread('{}/{}/images/{}'.format(TEST, ds, n)), cv2.COLOR_BGR2RGB)
    g = gray('{}/{}/masks/{}'.format(TEST, ds, n)) > 127
    qi = gray('{}/{}/{}'.format(IP, ds, n)).astype(float) / 255
    qp = gray('{}/{}/{}'.format(PL, ds, n)).astype(float) / 255
    pi, pp = qi >= 0.5, qp >= 0.5
    union = g | pi | pp
    ys, xs = np.where(union)
    cyy, cxx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    side = int(min(max(ys.ptp(), xs.ptp()) * 1.35 + 20, min(g.shape)))
    y0 = int(np.clip(cyy - side / 2, 0, g.shape[0] - side))
    x0 = int(np.clip(cxx - side / 2, 0, g.shape[1] - side))
    cr = lambda a: a[y0:y0 + side, x0:x0 + side]
    lw = max(1, int(round(side / 180)))
    cs, _ = cv2.findContours(cr(g).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    over = cr(rgb).copy()
    cv2.drawContours(over, cs, -1, (0, 0, 0), lw + 2, lineType=cv2.LINE_AA)
    cv2.drawContours(over, cs, -1, (255, 255, 255), lw, lineType=cv2.LINE_AA)

    def emap(p):
        o = np.zeros(cr(g).shape + (3,), np.uint8)
        G, P = cr(g), cr(p)
        o[P & G] = 255
        o[P & ~G] = [int(FP[i:i + 2], 16) for i in (1, 3, 5)]
        o[~P & G] = [int(FN[i:i + 2], 16) for i in (1, 3, 5)]
        return o

    yb_ = Hh - top - (r_ + 1) * cell - r_ * gap
    axs = [fig.add_axes([(cx0 + c * (cell + gap)) / W, yb_ / Hh, cell / W, cell / Hh])
           for c in range(4)]
    for ax_, img in zip(axs[:3], [over, emap(pp), emap(pi)]):
        ax_.imshow(img, interpolation='lanczos' if img is over else 'nearest')
    axs[1].text(0.03, 0.04, '{:.3f}'.format(r['d_pl']), transform=axs[1].transAxes, fontsize=6.5,
                color='white', va='bottom', bbox=dict(fc='black', ec='none', alpha=0.6, pad=1))
    axs[2].text(0.03, 0.04, '{:.3f}'.format(r['d_ip']), transform=axs[2].transAxes, fontsize=6.5,
                color='white', va='bottom', bbox=dict(fc='black', ec='none', alpha=0.6, pad=1))
    dp = cr(qi - qp)
    im_ = axs[3].imshow(dp, cmap='RdBu_r', norm=norm, interpolation='nearest')
    axs[3].contour(cr(g).astype(float), levels=[0.5], colors='black', linewidths=0.5)
    for ax_ in axs:
        ax_.set_xticks([]); ax_.set_yticks([])
        for s_ in ax_.spines.values():
            s_.set_linewidth(0.4); s_.set_color('#808080')
    fig.text((cx0 - 0.05) / W, (yb_ + cell / 2) / Hh, '{}\n{}'.format(lab, n.replace('.png', '')),
             ha='right', va='center', fontsize=6.5, linespacing=1.15)
    if r_ == 0:
        for c, hd in enumerate(heads):
            fig.text((cx0 + c * (cell + gap) + cell / 2) / W, (yb_ + cell + 0.05) / Hh, hd,
                     ha='center', va='bottom', fontsize=7, weight='bold')
    ex_prov.append(dict(row=lab.replace('\n', ' '), dataset=ds, image=n,
                        dice_plain=round(r['d_pl'], 4), dice_iprcf=round(r['d_ip'], 4),
                        crop_yxs=[y0, x0, side]))

# colour bar and error legend
cax = fig.add_axes([(cx0 + 3 * (cell + gap)) / W, 0.20 / Hh, cell / W, 0.06 / Hh])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r'), cax=cax,
                  orientation='horizontal', ticks=[-1, 0, 1])
cb.ax.set_xticklabels(['−1', '0', '1'], fontsize=6)
cb.outline.set_linewidth(0.4)
x = cx0
for c_, t in [('#FFFFFF', 'TP'), (FP, 'FP'), (FN, 'FN')]:
    fig.patches.append(Rectangle((x / W, 0.185 / Hh), 0.09 / W, 0.09 / Hh,
                                 transform=fig.transFigure, fc=c_, ec='black', lw=0.4))
    fig.text((x + 0.13) / W, 0.23 / Hh, t, ha='left', va='center', fontsize=6.5)
    x += 0.45

save(fig, 'fig1.png')
os.makedirs(OUT, exist_ok=True)
json.dump(dict(stats=stats, examples=ex_prov), open(os.path.join(OUT, 'fig1_evidence.json'), 'w'),
          indent=1)
for k in order:
    s = stats[k]
    print(k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in s.items()})
for e in ex_prov:
    print(e)
