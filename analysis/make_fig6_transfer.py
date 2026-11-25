"""Fig. 6: internal validation vs frozen five-benchmark Dice for every variant trained with the
identical positive/normal recipe (same positive split SHA 089132dd..., same HyperKvasir normal
manifest SHA 4b11116a..., PVTv2-B0, 64-channel decoder, seed 42).

x: internal positive soft Dice at the selected epoch (runs/<run>/epoch_metrics.csv,
   epoch with the best balanced harmonic score)
y: frozen macro Dice on the 798 development images (results/<run>/.../summary_results.json)
"""
import csv, glob, json, os
import numpy as np
from figstyle import *

STUDY = STUDY_ROOT
RUNS = [('tmi_mpec_data_control_c64_seed42', 'Plain-PFF control', 'control'),
        ('tmi_mpec_deep_presence_c64_seed42', 'Deep presence head', 'presence'),
        ('tmi_mpec_full_c64_seed42', 'Multi-scale presence', 'presence'),
        ('tmi_frbpff_dual_band_c64_seed42', 'Dual-band decoder', 'boundary'),
        ('tmi_iprcf_multiscale_c64_seed42', 'IP-RCF (multiscale)', 'boundary')]
STYLE = {'control': ('black', 'o'), 'presence': (NEU, 's'), 'boundary': (CON, 'D')}

pts = []
for run, label, kind in RUNS:
    em = list(csv.DictReader(open('{}/runs/{}/epoch_metrics.csv'.format(STUDY, run))))
    b = max(em, key=lambda x: float(x['balanced_harmonic_score']))
    f = glob.glob('{}/results/{}/**/summary_results.json'.format(STUDY, run), recursive=True)[0]
    d = json.load(open(f))
    pts.append(dict(run=run, label=label, kind=kind, epoch=int(b['epoch']),
                    internal=float(b['positive_raw_soft_dice']),
                    bench=float(d['macro_dataset_mean']['dice']),
                    sha=d.get('checkpoint_sha256', '')[:8]))

W, H = COL_W, 2.75
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0.62 / W, 0.48 / H, 2.75 / W, 2.10 / H])
c = pts[0]
ax.axhline(c['bench'], color='#9A9A9A', lw=0.6, ls=(0, (3, 2)), zorder=0)
ax.axvline(c['internal'], color='#9A9A9A', lw=0.6, ls=(0, (3, 2)), zorder=0)
ax.fill_between([c['internal'], 0.928], c['bench'], 0.844, color='#E9EEF4', zorder=-1, lw=0)
ax.text(0.9265, 0.8435, 'better on both', ha='right', va='top', fontsize=6, color=SEM,
        style='italic')
offs = {'Plain-PFF control': (-0.0002, 0.0009, 'left'),
        'Deep presence head': (0.0003, -0.0007, 'left'),
        'Multi-scale presence': (0.0, -0.0009, 'center'),
        'Dual-band decoder': (-0.0003, 0.0005, 'right'),
        'IP-RCF (multiscale)': (0.0, -0.0009, 'center')}
for p in pts:
    col, mk = STYLE[p['kind']]
    ax.plot(p['internal'], p['bench'], mk, color=col, ms=5, mec='white', mew=0.5, zorder=3)
    dx, dy, ha = offs[p['label']]
    ax.text(p['internal'] + dx, p['bench'] + dy, p['label'], ha=ha, va='center', fontsize=6.5,
            color=col)
    if p['kind'] != 'control':
        ax.annotate('', xy=(p['internal'], p['bench']), xytext=(c['internal'], c['bench']),
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=0.6, alpha=0.55,
                                    shrinkA=3, shrinkB=4, mutation_scale=6), zorder=2)
ax.set_xlim(0.912, 0.928)
ax.set_ylim(0.828, 0.844)
ax.set_xlabel('internal soft Dice (selected epoch)')
ax.set_ylabel('frozen five-benchmark macro Dice')
ax.set_xticks([0.912, 0.916, 0.920, 0.924, 0.928])
ax.set_yticks([0.828, 0.832, 0.836, 0.840, 0.844])
for s_ in ('top', 'right'):
    ax.spines[s_].set_visible(False)
rho = np.corrcoef(np.argsort(np.argsort([p['internal'] for p in pts])),
                  np.argsort(np.argsort([p['bench'] for p in pts])))[0, 1]
ax.text(0.9275, 0.8288, r'Spearman $\rho = %.2f$, $n = 5$' % rho, ha='right', va='bottom',
        fontsize=6.5, color=NEU)
from matplotlib.lines import Line2D
hs = [Line2D([], [], marker=STYLE[k][1], color=STYLE[k][0], ls='none', ms=4.5, mec='white', mew=0.5)
      for k in ('control', 'presence', 'boundary')]
fig.legend(hs, ['control', 'presence head', 'boundary-oriented decoder'], loc='lower center',
           ncol=3, frameon=False, fontsize=6.5, handletextpad=0.3, columnspacing=1.0,
           bbox_to_anchor=(0.5, 0.0))
save(fig, 'fig6.png')
os.makedirs('outputs', exist_ok=True)
json.dump(dict(points=pts, spearman=rho), open('outputs/fig6_transfer.json', 'w'), indent=1)
for p in pts:
    print(p)
print('spearman', rho)
