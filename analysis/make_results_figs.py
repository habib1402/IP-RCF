"""Result figures 1, 4, 5, 6, 7 for the IP-RCF TMI manuscript.

Without frozen IP-RCF outputs, the script writes figN_draft.png and fills only the
panels whose evidence is verified:
  * development images, masks and the authors' released comparator maps
    (EnFormer, CATNet-P, SEPNet; 798/798 aligned by name and size);
  * matched Plain-PFF control predictions (runs/tmi_mpec_data_control_c64_seed42,
    checkpoint SHA-256 86ae0c86...; reproduces Table I);
  * prior artifact-recipe Plain-PFF external predictions
    (runs/tmi_b0_qc_artifact_supervision_c64_seed42, SHA-256 3bce9f46...).

Once the IP-RCF checkpoint is frozen and evaluated, pass
  --iprcf_dev   <dir>/<dataset>/<name>.png   raw-sigmoid maps at native size (0-255)
  --iprcf_ext   <dir>  external predictions laid out as EvaluateExternalDatasets.py
                        writes them (PolypGen/positive/C1__<id>.png, ...)
  --iprcf_tag   short checkpoint identifier printed in the panel headers
to replace the pending columns and write figN.png instead of the drafts.
"""
import argparse, csv, json, os
import numpy as np
import cv2
from figstyle import *
from matplotlib.colors import TwoSlopeNorm

STUDY = STUDY_ROOT
TEST = DATA_ROOT + '/TestDataset'
MAPS = COMPARATOR_ROOT + ''
COMPARATORS = [('EnFormer', MAPS + '/EnFormer/result_map/res/enformer'),
               ('CATNet-P', MAPS + '/CATNet_P/CATN_P'),
               ('SEPNet', MAPS + '/SEPNet')]
PLAIN_DEV = STUDY + '/results/tmi_mpec_data_control_c64_seed42/predictions_raw_sigmoid'
PRIOR_EXT = STUDY + '/external_results/tmi_plainpff_external_seed42'
DATASETS = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
SHORT = {'CVC-300': 'CVC-300', 'CVC-ClinicDB': 'ClinicDB', 'Kvasir': 'Kvasir',
         'CVC-ColonDB': 'ColonDB', 'ETIS-LaribPolypDB': 'ETIS'}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

# Fig. 1 candidate rows: rank-1 images of the model-independent screen in
# select_challenge_candidates.py (GT and image statistics only).
CHALLENGE_ROWS = [('Small\npolyp', 'ETIS-LaribPolypDB', '70.png'),
                  ('Low-contrast\nboundary', 'CVC-ColonDB', '120.png'),
                  ('Glare /\ntexture', 'ETIS-LaribPolypDB', '105.png')]


def local(p):
    return p


def read_gray(p):
    a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(p)
    return a if a.ndim == 2 else a[..., 0]


def read_rgb(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def binarize(m):
    return m > 127


def dice(p, g):
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else 2.0 * (p & g).sum() / s


def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)


def error_map(pred, gt):
    out = np.zeros(gt.shape + (3,), np.uint8)
    out[pred & gt] = 255
    out[pred & ~gt] = hex_rgb(FP)
    out[~pred & gt] = hex_rgb(FN)
    return out


def contour_overlay(rgb, gt, width):
    out = rgb.copy()
    cs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, (0, 0, 0), width + 2, lineType=cv2.LINE_AA)
    cv2.drawContours(out, cs, -1, (255, 255, 255), width, lineType=cv2.LINE_AA)
    return out


def square_crop(gt, margin_frac=0.9, min_side=120):
    ys, xs = np.where(gt)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    side = max((ys.max() - ys.min()), (xs.max() - xs.min())) * (1 + margin_frac)
    side = int(min(max(side, min_side), min(gt.shape)))
    y0 = int(np.clip(cy - side / 2, 0, gt.shape[0] - side))
    x0 = int(np.clip(cx - side / 2, 0, gt.shape[1] - side))
    return y0, x0, side, side


def aspect_crop(gt, aspect=4 / 3):
    h, w = gt.shape
    ch, cw = (h, int(round(h * aspect))) if h * aspect <= w else (int(round(w / aspect)), w)
    ys, xs = np.where(gt)
    cy, cx = ys.mean(), xs.mean()
    y0 = int(np.clip(cy - ch / 2, 0, h - ch))
    x0 = int(np.clip(cx - cw / 2, 0, w - cw))
    return y0, x0, ch, cw


def show(ax, img):
    ax.imshow(img, interpolation='lanczos' if img.ndim == 3 else 'nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color('#808080')


def corner(ax, text, dark=True):
    ax.text(0.03, 0.04, text, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=6, color='white' if dark else 'black',
            bbox=dict(boxstyle='round,pad=0.12', fc='black' if dark else 'white',
                      ec='none', alpha=0.6))


def error_legend(fig, y_in, x_in=0.62, extra=''):
    """Swatch legend at inch position (x_in, y_in)."""
    W, H = fig.get_figwidth(), fig.get_figheight()
    x = x_in
    for c, t in [('#FFFFFF', 'TP'), (FP, 'FP (predicted, not annotated)'),
                 (FN, 'FN (annotated, missed)')]:
        fig.patches.append(Rectangle((x / W, (y_in - 0.045) / H), 0.09 / W, 0.09 / H,
                                     transform=fig.transFigure, fc=c, ec='black', lw=0.4))
        t_ = fig.text((x + 0.13) / W, y_in / H, t, ha='left', va='center', fontsize=6.5)
        x += 0.13 + 0.042 * len(t) + 0.2
    if extra:
        fig.text(x / W, y_in / H, extra, ha='left', va='center', fontsize=6.5, color=NEU)


# ============================================================== Figure 1
def figure1(args, prov):
    ncol = 6
    W = FULL_W
    left, right, gap = 0.72, 0.04, 0.035
    cw = (W - left - right - gap * (ncol - 1)) / ncol
    top, bottom = 0.42, 0.30
    Hh = top + bottom + 3 * cw + 2 * gap
    fig = plt.figure(figsize=(W, Hh))
    headers = ['Input frame', 'Crop + annotation', 'EnFormer', 'CATNet-P',
               'SEPNet', 'IP-RCF' + (' ({})'.format(args.iprcf_tag) if args.iprcf_dev else '')]
    rows_prov = []
    for r, (label, ds, name) in enumerate(CHALLENGE_ROWS):
        rgb = read_rgb('{}/{}/images/{}'.format(TEST, ds, name))
        gt = binarize(read_gray('{}/{}/masks/{}'.format(TEST, ds, name)))
        y0, x0, hh, ww = square_crop(gt)
        crop = lambda a: a[y0:y0 + hh, x0:x0 + ww]
        lw = max(1, int(round(hh / 200)))
        yb = Hh - top - (r + 1) * cw - r * gap
        axs = [fig.add_axes([(left + c * (cw + gap)) / W, yb / Hh, cw / W, cw / Hh])
               for c in range(ncol)]
        # full frame, letterboxed, with crop rectangle
        frame = rgb.copy()
        t = max(3, int(round(max(rgb.shape) / 220)))
        cv2.rectangle(frame, (x0, y0), (x0 + ww, y0 + hh), (255, 255, 255), t + 2)
        cv2.rectangle(frame, (x0, y0), (x0 + ww, y0 + hh), (0, 0, 0), max(1, t - 2))
        s = max(frame.shape[:2])
        pad = np.full((s, s, 3), 255, np.uint8)
        oy, ox = (s - frame.shape[0]) // 2, (s - frame.shape[1]) // 2
        pad[oy:oy + frame.shape[0], ox:ox + frame.shape[1]] = frame
        show(axs[0], pad)
        for sp in axs[0].spines.values():
            sp.set_visible(False)
        show(axs[1], contour_overlay(crop(rgb), crop(gt), lw))
        entry = dict(row=label.replace('\n', ' '), dataset=ds, image=name,
                     crop_yxhw=[y0, x0, hh, ww], gt_area_fraction=float(gt.mean()))
        for c, (m, root) in enumerate(COMPARATORS):
            pred = binarize(read_gray('{}/{}/{}'.format(root, ds, name)))
            assert pred.shape == gt.shape
            d = dice(pred, gt)
            show(axs[2 + c], error_map(crop(pred), crop(gt)))
            corner(axs[2 + c], 'Dice {:.3f}'.format(d))
            entry[m + '_dice'] = round(float(d), 4)
        if args.iprcf_dev:
            pred = binarize(read_gray('{}/{}/{}'.format(args.iprcf_dev, ds, name)))
            d = dice(pred, gt)
            show(axs[5], error_map(crop(pred), crop(gt)))
            corner(axs[5], 'Dice {:.3f}'.format(d))
            entry['IP-RCF_dice'] = round(float(d), 4)
        else:
            pending_panel(axs[5], 'frozen IP-RCF\nprediction pending')
        fig.text(0.06 / W, (yb + cw / 2) / Hh,
                 label + '\n{} {}'.format(SHORT[ds], name.replace('.png', '')),
                 ha='left', va='center', fontsize=6.5, linespacing=1.2)
        if r == 0:
            for c, hd in enumerate(headers):
                fig.text((left + c * (cw + gap) + cw / 2) / W,
                         (yb + cw + 0.06) / Hh, hd, ha='center', va='bottom',
                         fontsize=7, weight='bold')
        rows_prov.append(entry)
    error_legend(fig, 0.13, x_in=0.72,
                 extra='Comparators: authors’ released maps, threshold 0.5. Dice on the full frame.')
    draft = not args.iprcf_dev
    if draft:
        draft_banner(fig, 'DRAFT — rows from a model-independent candidate screen; '
                     'IP-RCF column awaits frozen-checkpoint predictions')
    prov['fig1'] = dict(rows=rows_prov, comparators=[m for m, _ in COMPARATORS],
                        draft=draft)
    save(fig, 'fig1_draft.png' if draft else 'fig1.png')


# ============================================================== Figure 4
def figure4(args, prov):
    W, Hh = COL_W, 4.30
    fig = plt.figure(figsize=(W, Hh))
    titles = ['(a) Input + annotated contour', '(b) p, fitted gains',
              '(c) p, gains set to 0', r'(d) $\Delta p$ = (b) $-$ (c)']
    pw, gx, x0 = 1.45, 0.16, 0.22
    for k in range(4):
        r, c = divmod(k, 2)
        yb = Hh - 0.44 - (r + 1) * pw - r * 0.30
        ax = fig.add_axes([(x0 + c * (pw + gx)) / W, yb / Hh, pw / W, pw / Hh])
        pending_panel(ax, ['same frozen checkpoint\nand image ID',
                           'pending', 'pending', 'pending'][k])
        ax.text(0.0, 1.03, titles[k], transform=ax.transAxes, ha='left',
                va='bottom', fontsize=6.5, weight='bold')
    cax = fig.add_axes([0.45 / W, 0.52 / Hh, 2.6 / W, 0.07 / Hh])
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=TwoSlopeNorm(0, -1, 1))
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[-1, 0, 1])
    cb.ax.set_xticklabels([r'$-m$', '0', r'$+m$'])
    cb.outline.set_linewidth(0.4)
    cb.set_label(r'$\Delta p$, diverging scale centered at 0; $m = \max|\Delta p|$',
                 fontsize=6.5, labelpad=2)
    draft_banner(fig, 'DRAFT — awaits frozen checkpoint')
    prov['fig4'] = dict(draft=True)
    save(fig, 'fig4_draft.png')


# ============================================================== Figure 5
IN_EDGES = [0, 2, 4, 8, 16, np.inf]
BIN_LABELS = (['>16', '8–16', '4–8', '2–4', '0–2'] +
              ['0–2', '2–4', '4–8', '8–16', '>16'])


def distance_bins(gt):
    """Signed-distance bin index per pixel: 0..4 inside (far->near), 5..9 outside."""
    g8 = gt.astype(np.uint8)
    din = cv2.distanceTransform(g8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dout = cv2.distanceTransform(1 - g8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    idx = np.full(gt.shape, -1, np.int8)
    for k in range(5):
        lo, hi = IN_EDGES[k], IN_EDGES[k + 1]
        idx[gt & (din > lo) & (din <= hi)] = 4 - k
        idx[~gt & (dout > lo) & (dout <= hi)] = 5 + k
    return idx


def per_image_bin_errors(pred_root):
    rows = []
    for ds in DATASETS:
        for name in sorted(os.listdir('{}/{}/masks'.format(TEST, ds))):
            gt = binarize(read_gray('{}/{}/masks/{}'.format(TEST, ds, name)))
            pred = read_gray('{}/{}/{}'.format(pred_root, ds, name)) >= 128
            assert pred.shape == gt.shape
            idx = distance_bins(gt)
            err = pred != gt
            e = np.full(10, np.nan)
            for b in range(10):
                m = idx == b
                if m.any():
                    e[b] = err[m].mean()
            rows.append((ds, name, e))
    return rows


def boot_ci(mat, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = mat.shape[0]
    means = np.empty((reps, mat.shape[1]))
    for k in range(reps):
        means[k] = np.nanmean(mat[rng.integers(0, n, n)], axis=0)
    return np.nanpercentile(means, 2.5, 0), np.nanpercentile(means, 97.5, 0)


def figure5(args, prov):
    os.makedirs(OUT, exist_ok=True)
    plain = per_image_bin_errors(PLAIN_DEV)
    P = np.stack([e for _, _, e in plain])
    names = [(d, n) for d, n, _ in plain]
    Ipr = None
    if args.iprcf_dev:
        ip = per_image_bin_errors(args.iprcf_dev)
        assert [(d, n) for d, n, _ in ip] == names
        Ipr = np.stack([e for _, _, e in ip])
    with open(os.path.join(OUT, 'fig5_per_image_bin_error.csv'), 'w') as f:
        w = csv.writer(f)
        w.writerow(['model', 'dataset', 'image'] + ['bin{}'.format(b) for b in range(10)])
        for (d, n), e in zip(names, P):
            w.writerow(['Plain-PFF', d, n] + list(e))
        if Ipr is not None:
            for (d, n), e in zip(names, Ipr):
                w.writerow(['IP-RCF', d, n] + list(e))
    mean_p = np.nanmean(P, 0)
    lo_p, hi_p = boot_ci(P)
    count = np.sum(~np.isnan(P), 0)

    W, Hh = COL_W, 3.4
    fig = plt.figure(figsize=(W, Hh))
    a1 = fig.add_axes([0.62 / W, 1.70 / Hh, 2.78 / W, 1.38 / Hh])
    a2 = fig.add_axes([0.62 / W, 0.62 / Hh, 2.78 / W, 0.80 / Hh], sharex=a1)
    x = np.arange(10)
    off = 0.13 if Ipr is not None else 0.0
    a1.errorbar(x - off, mean_p, yerr=[mean_p - lo_p, hi_p - mean_p], fmt='o-',
                color=SEM, ms=2.8, lw=0.9, capsize=1.6, elinewidth=0.7,
                label='Plain-PFF (matched control)')
    if Ipr is not None:
        mean_i = np.nanmean(Ipr, 0); lo_i, hi_i = boot_ci(Ipr)
        a1.errorbar(x + off, mean_i, yerr=[mean_i - lo_i, hi_i - mean_i], fmt='s-',
                    color=CON, mec='#7A2E1C', ms=2.8, lw=0.9, capsize=1.6,
                    elinewidth=0.7, label='IP-RCF ({})'.format(args.iprcf_tag))
        D = Ipr - P
        md = np.nanmean(D, 0); lo_d, hi_d = boot_ci(D)
        a2.errorbar(x, md, yerr=[md - lo_d, hi_d - md], fmt='o', color='black',
                    ms=2.8, capsize=1.6, elinewidth=0.7)
        prov['fig5_iprcf'] = dict(mean=md.tolist(), lo=lo_d.tolist(), hi=hi_d.tolist())
    else:
        a2.text(0.5, 0.5, 'paired difference pending frozen IP-RCF predictions\n'
                '(same 798 images, same bins, paired image bootstrap)',
                transform=a2.transAxes, ha='center', va='center', fontsize=6.5,
                color='#555555')
        a2.set_yticks([])
        a1.plot([], [], 's-', color=CON, ms=2.8, lw=0.9,
                label='IP-RCF (pending frozen checkpoint)')
    a1.set_yscale('log')
    a1.set_ylabel('pixel error rate')
    a1.axvline(4.5, color=NEU, lw=0.6, ls=(0, (2, 2)))
    ytop = a1.get_ylim()[1]
    a1.text(4.4, 1.02, 'inside (FN)', transform=a1.get_xaxis_transform(),
            ha='right', va='bottom', fontsize=6.5, color=NEU)
    a1.text(4.6, 1.02, 'outside (FP)', transform=a1.get_xaxis_transform(),
            ha='left', va='bottom', fontsize=6.5, color=NEU)
    a1.legend(loc='lower left', frameon=False, handlelength=1.6,
              borderaxespad=0.2, fontsize=6)
    a2.axhline(0, color=NEU, lw=0.6)
    a2.axvline(4.5, color=NEU, lw=0.6, ls=(0, (2, 2)))
    a2.set_ylabel('IP-RCF −\nPlain-PFF')
    a2.set_xticks(x)
    a2.set_xticklabels(BIN_LABELS, rotation=0, fontsize=5.8)
    plt.setp(a1.get_xticklabels(), visible=False)
    a2.set_xlabel('distance to the annotated contour (native pixels)')
    for a in (a1, a2):
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    fig.text(0.04 / W, (1.70 + 1.38) / Hh, '(a)', fontsize=8, weight='bold', va='top')
    fig.text(0.04 / W, 1.42 / Hh, '(b)', fontsize=8, weight='bold', va='top')
    fig.text(0.5, 0.03 / Hh,
             'Image-mean error per bin, 798 development images (five benchmarks);\n'
             'bars: 95% bootstrap CI over images (2,000 resamples).',
             ha='center', va='bottom', fontsize=5.8, color=NEU, linespacing=1.15)
    draft = Ipr is None
    if draft:
        draft_banner(fig, 'DRAFT — Plain-PFF measured; IP-RCF pending')
    prov['fig5'] = dict(draft=draft, bins=BIN_LABELS, images_per_bin=count.tolist(),
                        plain_mean=mean_p.tolist(), plain_lo=lo_p.tolist(),
                        plain_hi=hi_p.tolist(),
                        plain_source=PLAIN_DEV)
    save(fig, 'fig5_draft.png' if draft else 'fig5.png')


# ============================================================== Figure 6
EXT_GROUPS = [('PolypGen', 'C5'), ('PolypGen', 'C2'), ('PolypGen', 'C3'),
              ('FAR-POLYP-SEG', 'positive')]


def external_rows():
    """Per group, the positive frame closest to the group's median prior Dice."""
    rows = [x for x in csv.DictReader(open(PRIOR_EXT + '/per_image_metrics.csv'))
            if x['kind'] == 'positive']
    out = []
    for ds, sg in EXT_GROUPS:
        L = [x for x in rows if x['dataset'] == ds and x['subgroup'] == sg]
        med = float(np.median([float(x['hard_dice']) for x in L]))
        best = min(L, key=lambda x: (abs(float(x['hard_dice']) - med), x['sample_id']))
        out.append((ds, sg, best, med, len(L)))
    return out


def figure6(args, prov):
    ncol = 4
    W = COL_W
    left, gap = 0.50, 0.03
    cw = (W - left - 0.03 - gap * (ncol - 1)) / ncol
    ch = cw * 3 / 4
    rows = external_rows()
    top, bottom = 0.46, 0.42
    Hh = top + bottom + len(rows) * ch + (len(rows) - 1) * gap
    fig = plt.figure(figsize=(W, Hh))
    headers = ['Input', 'Annotation', 'Prior\nPlain-PFF†',
               'IP-RCF' + (' ({})'.format(args.iprcf_tag) if args.iprcf_ext else '')]
    entries = []
    for r, (ds, sg, x, med, n) in enumerate(rows):
        rgb = read_rgb(local(x['image_path']))
        gt = binarize(read_gray(local(x['mask_path'])))
        pr_path = PRIOR_EXT + '/predictions/' + x['prediction_path'].split('/predictions/')[1]
        prior = read_gray(pr_path) >= 128
        assert prior.shape == gt.shape == rgb.shape[:2]
        y0, x0, hh, ww = aspect_crop(gt)
        crop = lambda a: a[y0:y0 + hh, x0:x0 + ww]
        yb = Hh - top - (r + 1) * ch - r * gap
        axs = [fig.add_axes([(left + c * (cw + gap)) / W, yb / Hh, cw / W, ch / Hh])
               for c in range(ncol)]
        lw = max(2, int(round(hh / 170)))
        show(axs[0], crop(rgb))
        show(axs[1], contour_overlay(crop(rgb), crop(gt), lw))
        dprior = dice(prior, gt)
        show(axs[2], error_map(crop(prior), crop(gt)))
        corner(axs[2], '{:.3f}'.format(dprior))
        e = dict(dataset=ds, group=sg, sample_id=x['sample_id'],
                 patient_id=x['patient_id'], prior_plainpff_dice=round(float(dprior), 4),
                 group_median_prior_dice=round(med, 4), group_n=n,
                 crop_yxhw=[y0, x0, hh, ww])
        if args.iprcf_ext:
            ip = read_gray(os.path.join(args.iprcf_ext, pr_path.split('/predictions/')[1])) >= 128
            d = dice(ip, gt)
            show(axs[3], error_map(crop(ip), crop(gt)))
            corner(axs[3], '{:.3f}'.format(d))
            e['iprcf_dice'] = round(float(d), 4)
        else:
            pending_panel(axs[3], 'pending', fs=6)
        lab = ('PolypGen\n{}'.format(sg) if ds == 'PolypGen'
               else 'FAR\n{}'.format(x['patient_id'].replace('Patient', 'pt. ')))
        fig.text(0.04 / W, (yb + ch / 2) / Hh, lab, ha='left', va='center',
                 fontsize=6.5, linespacing=1.15)
        if r == 0:
            for c, hd in enumerate(headers):
                fig.text((left + c * (cw + gap) + cw / 2) / W, (yb + ch + 0.04) / Hh,
                         hd, ha='center', va='bottom', fontsize=6.5, weight='bold',
                         linespacing=1.1)
        entries.append(e)
    fig.text(0.04 / W, 0.25 / Hh,
             '† earlier artifact-recipe checkpoint; not a matched ablation. '
             'Corner: Dice on the full frame.', ha='left', va='center', fontsize=5.8,
             color=NEU)
    ys = 0.09 / Hh
    x = 0.04 / W
    for c, t in [('#FFFFFF', 'TP'), (FP, 'FP'), (FN, 'FN')]:
        fig.patches.append(Rectangle((x, ys - 0.045 / Hh), 0.09 / W, 0.09 / Hh,
                                     transform=fig.transFigure, fc=c, ec='black', lw=0.4))
        fig.text(x + 0.13 / W, ys, t, ha='left', va='center', fontsize=6.5)
        x += 0.45 / W
    fig.text(x, ys, 'Frame per group: closest to the group median prior Dice.',
             ha='left', va='center', fontsize=5.8, color=NEU)
    draft = not args.iprcf_ext
    if draft:
        draft_banner(fig, 'DRAFT — IP-RCF external predictions pending')
    prov['fig6'] = dict(draft=draft, rows=entries)
    save(fig, 'fig6_draft.png' if draft else 'fig6.png')


# ============================================================== Figure 7
def figure7(args, prov):
    W = COL_W
    ncol, left, gap = 4, 0.72, 0.03
    cw = (W - left - 0.03 - gap * (ncol - 1)) / ncol
    ch = cw * 3 / 4
    top, bottom = 0.46, 0.55
    Hh = top + bottom + 3 * ch + 2 * gap
    fig = plt.figure(figsize=(W, Hh))
    headers = ['Input', 'Annotation', 'IP-RCF\nmask', 'Error\noverlay']
    labels = ['Missed\npolyp', 'Boundary\nerror', 'Healthy-frame\nfalse alarm\n(FAR)']
    for r in range(3):
        yb = Hh - top - (r + 1) * ch - r * gap
        for c in range(ncol):
            ax = fig.add_axes([(left + c * (cw + gap)) / W, yb / Hh, cw / W, ch / Hh])
            pending_panel(ax, 'empty mask' if (r == 2 and c == 1) else 'pending', fs=6)
            if r == 0:
                fig.text((left + c * (cw + gap) + cw / 2) / W, (yb + ch + 0.04) / Hh,
                         headers[c], ha='center', va='bottom', fontsize=6.5,
                         weight='bold', linespacing=1.1)
        fig.text(0.04 / W, (yb + ch / 2) / Hh, labels[r], ha='left', va='center',
                 fontsize=6.5, linespacing=1.15)
    fig.text(0.04 / W, 0.30 / Hh,
             'Missed polyp: no predicted component overlaps the annotation.\n'
             'False alarm: threshold 0.5 and an 8-connected component\n'
             'of at least 100 native pixels on a healthy FAR frame.',
             ha='left', va='center', fontsize=5.8, color=NEU, linespacing=1.2)
    draft_banner(fig, 'DRAFT — awaits frozen IP-RCF evaluations')
    prov['fig7'] = dict(draft=True)
    save(fig, 'fig7_draft.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--iprcf_dev', default=None)
    ap.add_argument('--iprcf_ext', default=None)
    ap.add_argument('--iprcf_tag', default='frozen')
    ap.add_argument('--only', nargs='*', default=['1', '4', '5', '6', '7'])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pf = os.path.join(OUT, 'figure_provenance.json')
    prov = json.load(open(pf)) if os.path.exists(pf) else {}
    for k in args.only:
        {'1': figure1, '4': figure4, '5': figure5, '6': figure6, '7': figure7}[k](args, prov)
    json.dump(prov, open(pf, 'w'), indent=1)
    print('provenance ->', pf)
