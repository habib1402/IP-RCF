"""Evaluate the frozen seed-42 multiscale IP-RCF checkpoint on five datasets."""

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from lib.iprcf import build_iprcf_model
from TestFRBPFF import (
    DATA_ROOT, DATASETS, EXPECTED_COUNTS, METRICS, PROJECT_DIR,
    audit_dataset, evaluate_dataset, load_state_dict, sha256_file,
)


VARIANT = 'multiscale_residual'
RUN = 'tmi_iprcf_multiscale_c64_seed42'
CHECKPOINT = PROJECT_DIR / 'runs' / RUN / 'PolypPVT_best.pth'
CHECKPOINT_SHA = '2722e758e08bbb325b8b491c089444c6a3fe1d11d3b79dcef505cf0c6577ded0'
OUTPUT = PROJECT_DIR / 'results' / RUN / 'predictions'
CONTROL = (
    PROJECT_DIR / 'results' / 'tmi_mpec_data_control_c64_seed42' /
    'predictions_raw_sigmoid'
)
CONTROL_SUMMARY_SHA = 'e6a19dcc94508e3937c49e60e4c5e0389fc78c89beaa3810abda5f52aafb080a'
CONTROL_CSV_SHA = '3a2eca2c4158248e88183e3cc8e00b6572608328ef7b64b71298ee4725c3ce6a'
INPUT_SIZE = 352
THRESHOLD = 0.5


def verify_hash(path, expected):
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError('SHA-256 mismatch for {}: {}'.format(path, observed))


def verify_inputs():
    verify_hash(CHECKPOINT, CHECKPOINT_SHA)
    summary_path = CONTROL / 'summary_results.json'
    metrics_path = CONTROL / 'per_image_metrics.csv'
    verify_hash(summary_path, CONTROL_SUMMARY_SHA)
    verify_hash(metrics_path, CONTROL_CSV_SHA)
    control = json.loads(summary_path.read_text())

    expected_ids = {dataset: set() for dataset in DATASETS}
    with metrics_path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            expected_ids[row['dataset']].add(Path(row['name']).stem)
    for dataset in DATASETS:
        loader = audit_dataset(dataset, DATA_ROOT, INPUT_SIZE)
        observed_ids = {Path(path).stem for path in loader.images}
        if observed_ids != expected_ids[dataset]:
            raise ValueError('{} sample IDs differ from the control'.format(dataset))
    return control


def load_model():
    model = build_iprcf_model(
        VARIANT, channel=64, backbone='pvtv2_b0', load_backbone_weights=False,
    )
    model.load_state_dict(load_state_dict(CHECKPOINT), strict=True)
    return model.eval()


def write_results(folder, rows, results, control):
    with (folder / 'per_image_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['dataset', 'name', *METRICS])
        writer.writeheader()
        writer.writerows(rows)

    macro = {
        metric: float(np.mean([results[dataset][metric] for dataset in DATASETS]))
        for metric in METRICS
    }
    control_macro = control['macro_dataset_mean']
    delta_dice = {
        dataset: results[dataset]['dice'] - control['datasets'][dataset]['dice']
        for dataset in DATASETS
    }
    macro_gain = macro['dice'] - control_macro['dice']
    matches_or_improves = sum(gain >= 0.0 for gain in delta_dice.values())
    passes = macro_gain > 0.0 and matches_or_improves >= 3
    summary = {
        'study': 'Frozen seed-42 IP-RCF five-dataset evaluation',
        'variant': VARIANT,
        'training_seed': 42,
        'selected_epoch': 95,
        'checkpoint': str(CHECKPOINT),
        'checkpoint_sha256': CHECKPOINT_SHA,
        'backbone': 'pvtv2_b0',
        'channel': 64,
        'input_size': INPUT_SIZE,
        'threshold': THRESHOLD,
        'prediction_protocol': (
            'sum the three logits, bilinearly resize to native mask shape, '
            'apply raw sigmoid, then threshold; metrics before PNG quantization'
        ),
        'datasets': results,
        'macro_dataset_mean': macro,
        'plain_pff_control': {
            'path': str(CONTROL),
            'checkpoint_sha256': control['checkpoint_sha256'],
            'summary_sha256': CONTROL_SUMMARY_SHA,
            'per_image_metrics_sha256': CONTROL_CSV_SHA,
            'macro_dataset_mean': control_macro,
        },
        'comparison': {
            'dice_delta_by_dataset': delta_dice,
            'macro_dice_gain': macro_gain,
            'datasets_matched_or_improved': matches_or_improves,
            'passes_frozen_benchmark_rule': passes,
        },
        'provenance': {
            'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'visible_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpu': torch.cuda.get_device_name(0),
            'data_root': str(DATA_ROOT),
            'source_sha256': {
                name: sha256_file(PROJECT_DIR / name)
                for name in ('TestIPRCF.py', 'TestFRBPFF.py', 'lib/iprcf.py',
                             'lib/pvt.py', 'utils/dataloader.py')
            },
        },
    }
    (folder / 'summary_results.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False) + '\n'
    )
    lines = [
        'Frozen IP-RCF multiscale residual | seed 42 | epoch 95',
        'SHA-256: ' + CHECKPOINT_SHA,
        '',
        '{:<22} {:>5} {:>10} {:>10} {:>10} {:>10}'.format(
            'Dataset', 'N', 'IP-RCF D', 'IP-RCF IoU', 'Plain D', 'Delta D',
        ),
    ]
    for dataset, result in results.items():
        lines.append(
            '{:<22} {:>5d} {:>10.6f} {:>10.6f} {:>10.6f} {:>+10.6f}'.format(
                dataset, result['samples'], result['dice'], result['iou'],
                control['datasets'][dataset]['dice'], delta_dice[dataset],
            )
        )
    lines += [
        '',
        'Macro Dice: {:.6f} vs {:.6f} ({:+.6f})'.format(
            macro['dice'], control_macro['dice'], macro_gain,
        ),
        'Macro IoU:  {:.6f} vs {:.6f}'.format(macro['iou'], control_macro['iou']),
        'Datasets matched or improved: {}/5'.format(matches_or_improves),
        'Frozen benchmark rule: {}'.format('PASS' if passes else 'FAIL'),
    ]
    report = '\n'.join(lines) + '\n'
    (folder / 'summary_results.txt').write_text(report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    control = verify_inputs()
    model = load_model()
    if args.verify_only:
        with torch.inference_mode():
            outputs = model(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE))
            response = model.response_logits(outputs)
            if len(outputs) != 3 or response.shape != (1, 1, INPUT_SIZE, INPUT_SIZE):
                raise ValueError('Unexpected IP-RCF inference output shape')
            if not torch.isfinite(response).all().item():
                raise ValueError('Non-finite IP-RCF smoke-test response')
        print('Preflight passed: epoch 95, strict CPU load, 798 matched image IDs.')
        return

    if OUTPUT.exists():
        raise FileExistsError('Output already exists: {}'.format(OUTPUT))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the full benchmark evaluation')
    device = torch.device('cuda:0')
    model.to(device)
    config = SimpleNamespace(data_path=DATA_ROOT, input_size=INPUT_SIZE, threshold=THRESHOLD)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.iprcf_eval_', dir=OUTPUT.parent) as temp:
        stage = Path(temp) / 'predictions'
        rows, results = [], {}
        for dataset in DATASETS:
            dataset_rows, results[dataset] = evaluate_dataset(config, model, device, dataset, stage)
            rows.extend(dataset_rows)
        if len(rows) != sum(EXPECTED_COUNTS.values()):
            raise RuntimeError('Unexpected total image count')
        write_results(stage, rows, results, control)
        stage.rename(OUTPUT)
    print('Results saved to {}'.format(OUTPUT), flush=True)


if __name__ == '__main__':
    main()
