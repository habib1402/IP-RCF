"""Fig. 7: failure taxonomy and decomposition of the Dice change, frozen IP-RCF vs matched Plain-PFF,
same 798 development images, threshold 0.5 on the saved raw-sigmoid maps.

Categories per annotated image (P: prediction, G: annotation):
  missed lesion      : |P and G| = 0
  over-extension     : detected and |P| > 1.25 |G|
  under-segmentation : detected and |P| < 0.80 |G|
  spurious component : an 8-connected predicted component of >= 100 native px not touching G
(b) Mean Dice change per dataset split into images newly missed by IP-RCF, images whose
    lesion IP-RCF recovered, and all remaining images (contributions sum to the total).
"""
import os, json, collections
import numpy as np
import cv2
from figstyle import *

STUDY = STUDY_ROOT + '/results'
TEST = DATA_ROOT + '/TestDataset'
M = {'Plain-PFF': STUDY + '/tmi_mpec_data_control_c64_seed42/predictions_raw_sigmoid',
     'IP-RCF': STUDY + '/tmi_iprcf_multiscale_c64_seed42/predictions'}
DS = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
SHORT = {'CVC-300': 'CVC-300', 'CVC-ClinicDB': 'ClinicDB', 'Kvasir': 'Kvasir',
         'CVC-ColonDB': 'ColonDB', 'ETIS-LaribPolypDB': 'ETIS'}
CATS = ['missed lesion', 'under-segmentation', 'over-extension', 'spurious component']


def gray(p):
    a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    return a if a.ndim == 2 else a[..., 0]


def dice(p, g):
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else 2.0 * (p & g).sum() / s


count = collections.Counter()
parts = {ds: collections.Counter() for ds in DS}
nimg = collections.Counter()
for ds in DS:
    for n in sorted(os.listdir('{}/{}/masks'.format(TEST, ds))):
        g = gray('{}/{}/masks/{}'.format(TEST, ds, n)) > 127
        nimg[ds] += 1
        pr, mi = {}, {}
        for m, root in M.items():
            p = gray('{}/{}/{}'.format(root, ds, n)) >= 128
            pr[m] = p
            if not g.any():
                continue
            tp = (p & g).sum()
            ratio = p.sum() / g.sum()
            mi[m] = tp == 0
            if tp == 0:
                count[(m, 'missed lesion')] += 1
            elif ratio > 1.25:
                count[(m, 'over-extension')] += 1
            elif ratio < 0.80:
                count[(m, 'under-segmentation')] += 1
            k, lab, st, _ = cv2.connectedComponentsWithStats(p.astype(np.uint8), connectivity=8)
            if any(st[i, 4] >= 100 and not g[lab == i].any() for i in range(1, k)):
                count[(m, 'spurious component')] += 1
        d = dice(pr['IP-RCF'], g) - dice(pr['Plain-PFF'], g)
        if mi and mi['IP-RCF'] and not mi['Plain-PFF']:
            parts[ds]['new miss'] += d
        elif mi and mi['Plain-PFF'] and not mi['IP-RCF']:
            parts[ds]['recovered'] += d
        else:
            parts[ds]['other'] += d
dec = {ds: {k: parts[ds][k] / nimg[ds] for k in ('new miss', 'recovered', 'other')} for ds in DS}

W, H = COL_W, 3.55
fig = plt.figure(figsize=(W, H))

# (a) failure counts
axa = fig.add_axes([1.12 / W, 2.05 / H, 2.25 / W, 1.30 / H])
y = np.arange(len(CATS))[::-1]
hgt = 0.36
for j, (m, col) in enumerate([('Plain-PFF', SEM), ('IP-RCF', CON)]):
    v = [count[(m, c)] for c in CATS]
    axa.barh(y + (0.19 if j == 0 else -0.19), v, height=hgt, color=col, label=m, zorder=2)
    for yy, vv in zip(y, v):
        axa.text(vv + 2, yy + (0.19 if j == 0 else -0.19), str(vv), va='center', ha='left',
                 fontsize=6.2, color=col)
axa.set_yticks(y)
axa.set_yticklabels(CATS)
axa.set_xlabel('images (of 798)', labelpad=1)
axa.set_xlim(0, 135)
axa.legend(loc='lower right', frameon=False, fontsize=6.5, handlelength=1.0,
           bbox_to_anchor=(1.02, -0.02))
for s_ in ('top', 'right'):
    axa.spines[s_].set_visible(False)
fig.text(0.04 / W, 3.48 / H, '(a)', fontsize=8, weight='bold', va='top')

# (b) decomposition of the Dice change
axb = fig.add_axes([1.12 / W, 0.62 / H, 2.25 / W, 1.05 / H])
yb = np.arange(len(DS))[::-1]
cols = {'new miss': CON, 'recovered': SEM, 'other': '#A7A7A7'}
for yy, ds in zip(yb, DS):
    neg, pos = 0.0, 0.0
    for k in ('new miss', 'other', 'recovered'):
        v = dec[ds][k]
        if v == 0:
            continue
        left = neg + v if v < 0 else pos
        axb.barh(yy, abs(v), left=left, height=0.6, color=cols[k], zorder=2)
        if v < 0:
            neg += v
        else:
            pos += v
    tot = sum(dec[ds].values())
    axb.plot(tot, yy, 'd', color='black', ms=3.2, zorder=3)
axb.axvline(0, color=NEU, lw=0.6)
axb.set_yticks(yb)
axb.set_yticklabels([SHORT[d] for d in DS])
axb.set_xlim(-0.022, 0.008)
axb.set_xticks([-0.02, -0.01, 0, 0.005])
axb.set_xticklabels(['−0.02', '−0.01', '0', '0.005'])
axb.set_xlabel(r'contribution to $\Delta$Dice (IP-RCF $-$ Plain-PFF)', labelpad=1)
for s_ in ('top', 'right'):
    axb.spines[s_].set_visible(False)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
hs = [Patch(color=CON), Patch(color=SEM), Patch(color='#A7A7A7'),
      Line2D([], [], marker='d', color='black', ls='none', ms=3.2)]
fig.legend(hs, ['new misses', 'recovered lesions', 'other images', 'total'], loc='lower center',
           ncol=4, frameon=False, fontsize=6.2, handlelength=0.9, handletextpad=0.3,
           columnspacing=0.8, bbox_to_anchor=(0.5, 0.0))
fig.text(0.04 / W, 1.82 / H, '(b)', fontsize=8, weight='bold', va='top')

save(fig, 'fig7.png')
os.makedirs('outputs', exist_ok=True)
json.dump(dict(counts={'{}|{}'.format(*k): v for k, v in count.items()}, decomposition=dec,
               n=dict(nimg)), open('outputs/fig7_failure.json', 'w'), indent=1)
for c in CATS:
    print(c, count[('Plain-PFF', c)], count[('IP-RCF', c)])
for ds in DS:
    print(ds, {k: round(v, 4) for k, v in dec[ds].items()})
