"""Controlled IP-RCF screen for lightweight polyp segmentation.

All three arms use the locked MPEC positive/normal data protocol and seed 42.
External datasets and the five development benchmarks are never loaded.
"""

import argparse
import json
import os
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from Train import DATASET_ROOT, PROJECT_DIR, StructureLoss
from Train import set_reproducibility, sha256_file
from TrainFRBPFFStudent import FRBPFFTrainer
from TrainMPECStudent import NormalPresenceDataset, file_hash
from lib.iprcf import IPRCF_VARIANTS, build_iprcf_model
from utils.utils import AvgMeter, clip_gradient


class IPRCFTrainer(FRBPFFTrainer):
    """Audited normal-aware trainer for the IP-RCF causal screen."""

    def _initialize_model(self):
        self.model = build_iprcf_model(
            variant=self.config.variant,
            channel=self.config.channel,
            backbone=self.config.backbone,
            backbone_weights=self.config.backbone_weights,
            load_backbone_weights=True,
        ).cuda()
        total = sum(parameter.numel() for parameter in self.model.parameters())
        print(
            'Model: variant={}, backbone={}, parameters={:,}'.format(
                self.config.variant, self.config.backbone, total
            ),
            flush=True,
        )

    def _write_manifest(self):
        total_parameters = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        backbone_parameters = sum(
            parameter.numel() for parameter in self.model.backbone.parameters()
        )
        manifest = {
            'study': 'TMI identity-preserving residual contour fusion screen',
            'variant': self.config.variant,
            'frozen_before_training': True,
            'model': {
                'backbone': self.config.backbone,
                'decoder': self.model.decoder_variant,
                'total_parameters': total_parameters,
                'backbone_parameters': backbone_parameters,
                'adapter_and_decoder_parameters': (
                    total_parameters - backbone_parameters
                ),
                'semantic_path': 'complete unmodified Plain-PFF topology',
                'contour_source': 'parameter-free 3x3 local-mean residual',
                'contour_injection_initialization': 'exact zero',
                'contour_injection_bound': '[-0.25, 0.25]',
                'training_only_boundary_heads': True,
            },
            'data': self.data_counts,
            'positive_split_manifest': str(self.positive_split_path),
            'positive_split_sha256': file_hash(self.positive_split_path),
            'normal_manifest': str(self.normal_manifest_path),
            'normal_manifest_sha256': file_hash(self.normal_manifest_path),
            'normal_split_unit': 'image; patient identifiers unavailable',
            'forbidden_model_selection_data': [
                'FAR-POLYP-SEG', 'PolypGen',
                'CVC-300', 'CVC-ClinicDB', 'Kvasir',
                'CVC-ColonDB', 'ETIS-LaribPolypDB',
            ],
            'training': {
                'epochs': self.config.epoch,
                'seed': self.config.training_seed,
                'input_size': self.config.trainsize,
                'batch_size': self.config.batchsize,
                'artifact_strength': self.config.artifact_strength,
                'response_loss_weight': self.config.response_loss_weight,
                'boundary_loss_weight': self.config.boundary_loss_weight,
                'contour_consistency_weight': (
                    self.config.contour_consistency_weight
                    if self.config.variant == 'multiscale_consistency'
                    else 0.0
                ),
                'boundary_target_radius_pixels_at_352': (
                    self.config.boundary_radius
                ),
                'multi_scale_rates': [0.75, 1.0, 1.25],
            },
            'checkpoint_selection': {
                'positive_metric': 'raw-sigmoid soft Dice',
                'normal_metric': (
                    'specificity of threshold-0.5 predictions with an '
                    '8-connected component >=100 pixels at 352x352'
                ),
                'score': 'harmonic mean of positive Dice and normal specificity',
                'boundary_f1_used_for_selection': False,
                'external_or_five_test_sets_used': False,
            },
            'research_design': {
                'finest_residual': (
                    'single finest-scale residual contour correction'
                ),
                'multiscale_residual': (
                    'progressive multiscale contour path with three residual '
                    'corrections'
                ),
                'multiscale_consistency': (
                    'multiscale residual architecture plus training-only '
                    'cross-scale contour consistency'
                ),
                'response_to_rejected_dual_band': (
                    'retains the full semantic feature stream rather than '
                    'replacing it with a low-frequency stream'
                ),
            },
            'provenance': {
                'command': [sys.executable] + sys.argv,
                'working_directory': str(Path.cwd()),
                'code_sha256': {
                    'TrainIPRCFStudent.py': sha256_file(Path(__file__).resolve()),
                    'lib/iprcf.py': sha256_file(
                        PROJECT_DIR / 'lib' / 'iprcf.py'
                    ),
                    'TrainFRBPFFStudent.py': sha256_file(
                        PROJECT_DIR / 'TrainFRBPFFStudent.py'
                    ),
                    'TrainMPECStudent.py': sha256_file(
                        PROJECT_DIR / 'TrainMPECStudent.py'
                    ),
                    'lib/pvt.py': sha256_file(PROJECT_DIR / 'lib' / 'pvt.py'),
                },
                'hostname': socket.gethostname(),
                'python': platform.python_version(),
                'torch': torch.__version__,
                'cuda_runtime': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(0),
                'visible_physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
            },
        }
        output = Path(self.config.train_save)
        output.mkdir(parents=True, exist_ok=True)
        with (output / 'experiment_config.json').open('w') as handle:
            json.dump(manifest, handle, indent=2)

    @staticmethod
    def _contour_consistency(boundary_outputs):
        if len(boundary_outputs) < 2:
            return boundary_outputs[0].sum() * 0.0
        probabilities = [
            torch.sigmoid(output.float()) for output in boundary_outputs
        ]
        return torch.stack([
            F.smooth_l1_loss(first, second)
            for first, second in zip(probabilities[:-1], probabilities[1:])
        ]).mean()

    def _branch_loss(self, images, masks):
        details = self._forward(images)
        segmentation = self._segmentation_loss(
            details['outputs'], details['response'], masks
        )
        boundary, _target = self._boundary_loss(
            details['boundary_outputs'], masks
        )
        consistency = (
            self._contour_consistency(details['boundary_outputs'])
            if self.config.variant == 'multiscale_consistency'
            else boundary.detach() * 0.0
        )
        return segmentation, boundary, consistency, details

    def train_epoch(self, epoch):
        self.model.train()
        self.artifact_simulator.train()
        total_meter = AvgMeter()
        segmentation_meter = AvgMeter()
        boundary_meter = AvgMeter()
        consistency_meter = AvgMeter()
        normal_fraction_sum = 0.0
        gain_sum = np.zeros(3, dtype=np.float64)
        batches = 0

        for step, pack in enumerate(self.train_loader, start=1):
            images_cpu, masks_cpu, presence_cpu, _names, _sources = pack
            for rate in (0.75, 1.0, 1.25):
                self.optimizer.zero_grad(set_to_none=True)
                images = images_cpu.cuda(non_blocking=True)
                masks = masks_cpu.cuda(non_blocking=True)
                presence_targets = presence_cpu.cuda(non_blocking=True)
                if rate != 1.0:
                    size = int(round(self.config.trainsize * rate / 32) * 32)
                    images = F.interpolate(
                        images, size=(size, size), mode='bilinear',
                        align_corners=True,
                    )
                    masks = F.interpolate(
                        masks, size=(size, size), mode='bilinear',
                        align_corners=True,
                    )
                with torch.no_grad():
                    degraded_images, _statistics = self.artifact_simulator(images)
                with torch.cuda.amp.autocast(enabled=self.config.amp):
                    clean_seg, clean_boundary, clean_consistency, _clean = (
                        self._branch_loss(images, masks)
                    )
                    degraded_seg, degraded_boundary, degraded_consistency, details = (
                        self._branch_loss(degraded_images, masks)
                    )
                    segmentation_loss = 0.5 * (clean_seg + degraded_seg)
                    boundary_loss = 0.5 * (
                        clean_boundary + degraded_boundary
                    )
                    consistency_loss = 0.5 * (
                        clean_consistency + degraded_consistency
                    )
                    total_loss = (
                        segmentation_loss
                        + self.config.boundary_loss_weight * boundary_loss
                        + self.config.contour_consistency_weight
                        * consistency_loss
                    )
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_gradient(self.optimizer, self.config.clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if rate == 1.0:
                    batch_size = images.size(0)
                    total_meter.update(total_loss.detach(), batch_size)
                    segmentation_meter.update(
                        segmentation_loss.detach(), batch_size
                    )
                    boundary_meter.update(boundary_loss.detach(), batch_size)
                    consistency_meter.update(
                        consistency_loss.detach(), batch_size
                    )
                    normal_fraction_sum += float(
                        (presence_targets < 0.5).float().mean().item()
                    )
                    gain_sum += (
                        details['injection_gains'].detach().float().cpu().numpy()
                    )
                    batches += 1

            if step % 20 == 0 or step == len(self.train_loader):
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                    'Loss={:.4f}, Seg={:.4f}, Boundary={:.4f}, '
                    'Consistency={:.4f}'.format(
                        datetime.now(), epoch, self.config.epoch, step,
                        len(self.train_loader), total_meter.show(),
                        segmentation_meter.show(), boundary_meter.show(),
                        consistency_meter.show(),
                    ),
                    flush=True,
                )

        denominator = max(batches, 1)
        result = {
            'total_loss': float(total_meter.show()),
            'segmentation_loss': float(segmentation_meter.show()),
            'boundary_loss': float(boundary_meter.show()),
            'contour_consistency_loss': float(consistency_meter.show()),
            'sampled_normal_fraction': normal_fraction_sum / denominator,
        }
        for index in range(3):
            result['injection_gain_{}'.format(index)] = float(
                gain_sum[index] / denominator
            )
            result['gate_mean_{}'.format(index)] = 0.0
        return result


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Controlled identity-preserving residual contour screen'
    )
    parser.add_argument('--variant', choices=IPRCF_VARIANTS, required=True)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone_lr_mult', type=float, default=0.1)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--trainsize', type=int, default=352)
    parser.add_argument('--channel', type=int, default=64)
    parser.add_argument('--backbone', choices=('pvtv2_b0',), default='pvtv2_b0')
    parser.add_argument('--backbone_weights', default=None)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int, default=50)
    parser.add_argument('--artifact_strength', type=float, default=0.5)
    parser.add_argument('--response_loss_weight', type=float, default=0.5)
    parser.add_argument('--boundary_loss_weight', type=float, default=0.25)
    parser.add_argument('--contour_consistency_weight', type=float, default=0.10)
    parser.add_argument('--boundary_radius', type=int, default=2)
    parser.add_argument('--boundary_tolerance', type=int, default=2)
    parser.add_argument('--training_seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--train_path', default=str(DATASET_ROOT / 'TrainDataset'))
    parser.add_argument('--train_save', required=True)
    parser.add_argument(
        '--positive_split_manifest',
        default=str(PROJECT_DIR / 'splits' / 'positive_split_seed42.json'),
    )
    parser.add_argument('--normal_manifest', required=True)
    parser.add_argument('--verify_only', action='store_true')
    return parser.parse_args()


def validate_arguments(config):
    locked = {
        'epoch': 100,
        'lr': 1e-4,
        'backbone_lr_mult': 0.1,
        'batchsize': 8,
        'trainsize': 352,
        'channel': 64,
        'artifact_strength': 0.5,
        'response_loss_weight': 0.5,
        'boundary_loss_weight': 0.25,
        'contour_consistency_weight': 0.10,
        'boundary_radius': 2,
        'boundary_tolerance': 2,
        'training_seed': 42,
    }
    for field, expected in locked.items():
        if getattr(config, field) != expected:
            raise ValueError(
                'Controlled IP-RCF screen locks {} to {}'.format(
                    field, expected
                )
            )


def verify_only(config):
    trainer = object.__new__(IPRCFTrainer)
    trainer.config = config
    trainer.structure_loss = StructureLoss()
    trainer._load_protocol()
    positive_train, positive_validation = trainer._positive_datasets()
    normal_train = NormalPresenceDataset(
        trainer.normal_manifest['training'], config.trainsize, True
    )
    normal_validation = NormalPresenceDataset(
        trainer.normal_manifest['validation'], config.trainsize, False
    )
    model = build_iprcf_model(
        config.variant,
        channel=config.channel,
        backbone=config.backbone,
        load_backbone_weights=False,
    ).train()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if total_parameters > 3_550_000:
        raise RuntimeError(
            'IP-RCF parameter budget exceeded: {:,}'.format(total_parameters)
        )
    image, _mask, _label, name, source = normal_train[0]
    images = F.interpolate(
        image.unsqueeze(0).repeat(2, 1, 1, 1),
        size=(128, 128),
        mode='bilinear',
        align_corners=False,
    )
    masks = torch.zeros(2, 1, 128, 128)
    masks[0, :, 32:96, 40:88] = 1.0
    details = model(images, return_details=True)
    segmentation = sum(
        trainer.structure_loss.compute(output, masks)
        for output in details['outputs']
    ) + config.response_loss_weight * trainer.structure_loss.compute(
        details['response'], masks
    )
    boundary, _target = trainer._boundary_loss(
        details['boundary_outputs'], masks
    )
    consistency = (
        trainer._contour_consistency(details['boundary_outputs'])
        if config.variant == 'multiscale_consistency'
        else boundary.detach() * 0.0
    )
    loss = (
        segmentation
        + config.boundary_loss_weight * boundary
        + config.contour_consistency_weight * consistency
    )
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite IP-RCF verification loss')
    missing_gradients = [
        parameter_name
        for parameter_name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing_gradients:
        raise RuntimeError(
            'Missing IP-RCF gradients: {}'.format(missing_gradients[:8])
        )
    print(
        'IP-RCF verification passed: variant={}, parameters={}, '
        'positive_train={}, positive_validation={}, normal_train={}, '
        'normal_validation={}, sample={}, source={}, outputs={}, '
        'boundaries={}, response={}, loss={:.6f}'.format(
            config.variant, total_parameters, len(positive_train),
            len(positive_validation), len(normal_train),
            len(normal_validation), name, source,
            [tuple(value.shape) for value in details['outputs']],
            [tuple(value.shape) for value in details['boundary_outputs']],
            tuple(details['response'].shape), float(loss.detach()),
        ),
        flush=True,
    )


def main():
    config = parse_arguments()
    validate_arguments(config)
    set_reproducibility(config.training_seed)
    if config.verify_only:
        verify_only(config)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for IP-RCF training')
    IPRCFTrainer(config).train()


if __name__ == '__main__':
    main()
