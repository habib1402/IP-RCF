"""Frozen five-dataset evaluation for the selected dual-band FRB-PFF model.

The architecture and checkpoint were selected using the internal validation
split before inspecting these development benchmarks.  This evaluator uses
native-size raw sigmoid probabilities, a fixed 0.5 threshold, and no
per-image normalization or post-processing.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.frbpff import build_frbpff_model
from utils.dataloader import test_dataset


PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get('POLYP_DATA_ROOT', str(PROJECT_DIR / 'data'))) / 'TestDataset'
FROZEN_VARIANT = 'dual_band'
FROZEN_CHECKPOINT_SHA256 = (
    '5901ca917f6c1e11dc5a81614369d67638c78704894dc9bb7bb836b1742bca27'
)
EXPECTED_COUNTS = {
    'CVC-300': 60,
    'CVC-ClinicDB': 62,
    'Kvasir': 100,
    'CVC-ColonDB': 380,
    'ETIS-LaribPolypDB': 196,
}
DATASETS = tuple(EXPECTED_COUNTS)
METRICS = (
    'dice',
    'soft_dice',
    'iou',
    'accuracy',
    'sensitivity',
    'specificity',
    'mean_probability',
    'max_probability',
)


def sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(block_size), b''):
            digest.update(block)
    return digest.hexdigest()


def load_state_dict(path):
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise TypeError('Checkpoint does not contain a non-empty state dict')
    if all(key.startswith('module.') for key in checkpoint):
        checkpoint = {
            key[len('module.'):]: value for key, value in checkpoint.items()
        }
    return checkpoint


def verify_checkpoint(config):
    checkpoint = Path(config.pth_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    observed = sha256_file(checkpoint)
    if observed != config.expected_checkpoint_sha256:
        raise RuntimeError(
            'Frozen checkpoint SHA-256 mismatch:\nexpected={}\nobserved={}'.format(
                config.expected_checkpoint_sha256, observed
            )
        )
    return observed


def build_and_load(config, device):
    model = build_frbpff_model(
        FROZEN_VARIANT,
        channel=config.channel,
        backbone=config.backbone,
        load_backbone_weights=False,
    )
    model.load_state_dict(load_state_dict(config.pth_path), strict=True)
    model.to(device).eval()
    return model


def audit_dataset(dataset_name, data_root, input_size):
    root = data_root / dataset_name
    loader = test_dataset(
        str(root / 'images'), str(root / 'masks'), input_size
    )
    image_count = len(loader.images)
    mask_count = len(loader.gts)
    expected = EXPECTED_COUNTS[dataset_name]
    if image_count != expected or mask_count != expected:
        raise RuntimeError(
            '{} count mismatch: expected {}, found {} images and {} masks'.format(
                dataset_name, expected, image_count, mask_count
            )
        )
    image_stems = [Path(path).stem for path in loader.images]
    mask_stems = [Path(path).stem for path in loader.gts]
    if image_stems != mask_stems:
        mismatches = [
            (image, mask)
            for image, mask in zip(image_stems, mask_stems)
            if image != mask
        ]
        raise RuntimeError(
            '{} image/mask ordering mismatch; examples={}'.format(
                dataset_name, mismatches[:5]
            )
        )
    if len(set(image_stems)) != expected:
        raise RuntimeError('{} contains duplicate stems'.format(dataset_name))
    return loader


def sample_metrics(probability, ground_truth, threshold):
    eps = 1e-8
    probability = probability.astype(np.float64, copy=False)
    target = (ground_truth > 0.5).astype(np.float64)
    prediction = (probability >= threshold).astype(np.float64)
    intersection = float((prediction * target).sum())
    prediction_sum = float(prediction.sum())
    target_sum = float(target.sum())
    union = prediction_sum + target_sum - intersection
    true_negative = float(((1.0 - prediction) * (1.0 - target)).sum())
    negative_sum = float((1.0 - target).sum())
    soft_intersection = float((probability * target).sum())
    return {
        'dice': (2.0 * intersection + eps) /
                (prediction_sum + target_sum + eps),
        'soft_dice': (2.0 * soft_intersection + eps) /
                     (float(probability.sum()) + target_sum + eps),
        'iou': (intersection + eps) / (union + eps),
        'accuracy': float((prediction == target).mean()),
        'sensitivity': (intersection + eps) / (target_sum + eps),
        'specificity': (true_negative + eps) / (negative_sum + eps),
        'mean_probability': float(probability.mean()),
        'max_probability': float(probability.max()),
    }


def evaluate_dataset(config, model, device, dataset_name, output_root):
    loader = audit_dataset(
        dataset_name, Path(config.data_path), config.input_size
    )
    prediction_dir = output_root / dataset_name
    prediction_dir.mkdir(parents=True, exist_ok=False)
    records = []
    with torch.inference_mode():
        for _index in range(len(loader)):
            image, ground_truth, name = loader.load_data()
            target = np.asarray(ground_truth, dtype=np.float32)
            if target.max() > 0.0:
                target /= target.max()
            image = image.to(device, non_blocking=True)
            outputs = model(image)
            response = model.response_logits(outputs)
            response = F.interpolate(
                response,
                size=target.shape,
                mode='bilinear',
                align_corners=False,
            )
            probability = (
                torch.sigmoid(response.float())
                .squeeze(0).squeeze(0).cpu().numpy()
            )
            metrics = sample_metrics(probability, target, config.threshold)
            records.append({
                'dataset': dataset_name,
                'name': name,
                **metrics,
            })
            encoded = np.rint(
                np.clip(probability, 0.0, 1.0) * 255.0
            ).astype(np.uint8)
            if not cv2.imwrite(str(prediction_dir / name), encoded):
                raise RuntimeError(
                    'Failed to write {}'.format(prediction_dir / name)
                )
    summary = {
        'samples': len(records),
        **{
            metric: float(np.mean([row[metric] for row in records]))
            for metric in METRICS
        },
    }
    print(
        '{}: n={} Dice={:.6f} softDice={:.6f} IoU={:.6f} '
        'Sens={:.6f} Spec={:.6f}'.format(
            dataset_name, summary['samples'], summary['dice'],
            summary['soft_dice'], summary['iou'], summary['sensitivity'],
            summary['specificity']
        ),
        flush=True,
    )
    return records, summary


def write_outputs(config, model, checkpoint_sha256, output_root, rows, results):
    fields = ['dataset', 'name', *METRICS]
    with (output_root / 'per_image_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    macro = {
        metric: float(np.mean([
            results[dataset][metric] for dataset in DATASETS
        ]))
        for metric in METRICS
    }
    pooled = {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in METRICS
    }
    summary = {
        'study': 'Frozen dual-band FRB-PFF five-dataset evaluation',
        'variant': FROZEN_VARIANT,
        'training_seed': 42,
        'architecture_selection_changed_by_this_evaluation': False,
        'selection_note': (
            'Dual-band was frozen as the internal-validation winner before '
            'these five development benchmarks were evaluated.'
        ),
        'checkpoint': str(Path(config.pth_path).resolve()),
        'checkpoint_sha256': checkpoint_sha256,
        'backbone': config.backbone,
        'channel': config.channel,
        'input_size': config.input_size,
        'threshold': config.threshold,
        'prediction_protocol': (
            'sum three segmentation logits, interpolate to native mask size, '
            'then raw sigmoid; no per-image min-max normalization or '
            'post-processing'
        ),
        'expected_dataset_counts': EXPECTED_COUNTS,
        'datasets': results,
        'macro_dataset_mean': macro,
        'pooled_image_mean': pooled,
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
        'provenance': {
            'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'command': [sys.executable] + sys.argv,
            'working_directory': str(Path.cwd()),
            'hostname': socket.gethostname(),
            'python': platform.python_version(),
            'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'visible_physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpu': torch.cuda.get_device_name(0),
            'code_sha256': {
                'TestFRBPFF.py': sha256_file(Path(__file__).resolve()),
                'lib/frbpff.py': sha256_file(PROJECT_DIR / 'lib' / 'frbpff.py'),
                'lib/pvt.py': sha256_file(PROJECT_DIR / 'lib' / 'pvt.py'),
            },
        },
    }
    with (output_root / 'summary_results.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
    lines = [
        'FROZEN DUAL-BAND FRB-PFF FIVE-DATASET EVALUATION',
        '=' * 96,
        'Variant: {}'.format(FROZEN_VARIANT),
        'Checkpoint SHA-256: {}'.format(checkpoint_sha256),
        'Protocol: raw sigmoid, threshold {:.2f}, no min-max or post-processing'.format(
            config.threshold
        ),
        '',
        '{:<22} {:>5} {:>10} {:>10} {:>10} {:>10} {:>10}'.format(
            'Dataset', 'N', 'Dice', 'softDice', 'IoU', 'Sens.', 'Spec.'
        ),
        '-' * 96,
    ]
    for dataset in DATASETS:
        record = results[dataset]
        lines.append(
            '{:<22} {:>5d} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f}'.format(
                dataset, record['samples'], record['dice'],
                record['soft_dice'], record['iou'], record['sensitivity'],
                record['specificity']
            )
        )
    lines.extend((
        '-' * 96,
        '{:<22} {:>5} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f}'.format(
            'Macro dataset mean', '', macro['dice'], macro['soft_dice'],
            macro['iou'], macro['sensitivity'], macro['specificity']
        ),
        '{:<22} {:>5d} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f}'.format(
            'Pooled image mean', len(rows), pooled['dice'],
            pooled['soft_dice'], pooled['iou'], pooled['sensitivity'],
            pooled['specificity']
        ),
    ))
    report = '\n'.join(lines) + '\n'
    (output_root / 'summary_results.txt').write_text(report)
    print(report, flush=True)


def verify(config):
    checkpoint_sha256 = verify_checkpoint(config)
    model = build_and_load(config, torch.device('cpu'))
    counts = {
        dataset: len(audit_dataset(
            dataset, Path(config.data_path), config.input_size
        ))
        for dataset in DATASETS
    }
    print(
        'FRB-PFF frozen-test verification passed: variant={}, parameters={}, '
        'checkpoint_sha256={}, dataset_counts={}'.format(
            FROZEN_VARIANT,
            sum(parameter.numel() for parameter in model.parameters()),
            checkpoint_sha256,
            counts,
        ),
        flush=True,
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--data_path', default=str(DATA_ROOT))
    parser.add_argument('--backbone', default='pvtv2_b0', choices=('pvtv2_b0',))
    parser.add_argument('--channel', type=int, default=64, choices=(64,))
    parser.add_argument('--input_size', type=int, default=352, choices=(352,))
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument(
        '--expected_checkpoint_sha256',
        default=FROZEN_CHECKPOINT_SHA256,
    )
    parser.add_argument('--verify_only', action='store_true')
    return parser.parse_args()


def main():
    config = parse_arguments()
    if config.threshold != 0.5:
        raise ValueError('Controlled evaluation locks threshold to 0.5')
    checkpoint_sha256 = verify_checkpoint(config)
    if config.verify_only:
        verify(config)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for full FRB-PFF evaluation')
    output_root = Path(config.output_dir)
    if output_root.exists():
        raise FileExistsError(
            'Refusing to overwrite existing output: {}'.format(output_root)
        )
    for dataset in DATASETS:
        audit_dataset(dataset, Path(config.data_path), config.input_size)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda:0')
    model = build_and_load(config, device)
    output_root.mkdir(parents=True)
    rows = []
    results = {}
    for dataset in DATASETS:
        dataset_rows, summary = evaluate_dataset(
            config, model, device, dataset, output_root
        )
        rows.extend(dataset_rows)
        results[dataset] = summary
    write_outputs(
        config, model, checkpoint_sha256, output_root, rows, results
    )


if __name__ == '__main__':
    main()
