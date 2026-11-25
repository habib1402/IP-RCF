"""Within-model contour-gain intervention for the frozen IP-RCF checkpoint (CPU only).

Two passes over the same 798 development images with the frozen epoch-95 weights:
  learned : unchanged model (must reproduce the frozen GPU evaluation)
  zero    : gain logits a_i set to 0, so g_i = 0.25*tanh(0) = 0 and S~_i = S_i
Everything else (loader, preprocessing, logit sum, resize, sigmoid, threshold 0.5,
metrics) is the frozen evaluation code from TestIPRCF.py / TestFRBPFF.py.
Run with CUDA_VISIBLE_DEVICES="" so no GPU is touched.
"""
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

STUDY = Path(os.environ.get('IPRCF_ROOT', str(Path(__file__).resolve().parent.parent)))
sys.path.insert(0, str(STUDY))
os.chdir(STUDY)

import numpy as np
import torch

from TestIPRCF import load_model, verify_inputs, INPUT_SIZE, THRESHOLD, CHECKPOINT_SHA
from TestFRBPFF import DATA_ROOT, DATASETS, METRICS, evaluate_dataset

OUT = STUDY / 'analyses' / 'iprcf_gain_intervention_seed42'


def main():
    assert not torch.cuda.is_available(), 'run with CUDA_VISIBLE_DEVICES=""'
    torch.set_num_threads(12)
    verify_inputs()
    device = torch.device('cpu')
    config = SimpleNamespace(data_path=DATA_ROOT, input_size=INPUT_SIZE, threshold=THRESHOLD)
    OUT.mkdir(parents=True, exist_ok=False)
    summary = {'checkpoint_sha256': CHECKPOINT_SHA, 'device': 'cpu', 'modes': {}}
    for mode in ('learned', 'zero'):
        model = load_model().to(device)
        decoder = model.base.progressive_decoder
        gains_before = decoder.bounded_gains.detach().tolist()
        if mode == 'zero':
            with torch.no_grad():
                decoder.gain_logits.zero_()
        rows, per = [], {}
        for ds in DATASETS:
            r, per[ds] = evaluate_dataset(config, model, device, ds, OUT / mode)
            rows.extend(r)
        with (OUT / mode / 'per_image_metrics.csv').open('w', newline='') as h:
            w = csv.DictWriter(h, fieldnames=['dataset', 'name', *METRICS])
            w.writeheader()
            w.writerows(rows)
        summary['modes'][mode] = {
            'gains': decoder.bounded_gains.detach().tolist(),
            'gains_learned': gains_before,
            'datasets': per,
            'macro_dice': float(np.mean([per[d]['dice'] for d in DATASETS])),
            'macro_iou': float(np.mean([per[d]['iou'] for d in DATASETS])),
        }
        print(mode, 'macro Dice {:.6f}'.format(summary['modes'][mode]['macro_dice']), flush=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
