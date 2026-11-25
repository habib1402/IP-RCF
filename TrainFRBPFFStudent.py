"""Controlled FRB-PFF screen for lightweight polyp segmentation.

All variants use PVTv2-B0, the locked positive/normal split, paired clean and
artifact-degraded supervision, and the same optimization recipe as the MPEC
data control.  FAR-POLYP-SEG, PolypGen and the five conventional test sets are
not loaded for training or checkpoint selection.
"""

import argparse
import json
import os
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from Train import DATASET_ROOT, PROJECT_DIR, StructureLoss, seed_worker
from Train import set_reproducibility, sha256_file
from TrainMPECStudent import MPECTrainer, NormalPresenceDataset, file_hash
from lib.frbpff import FRBPFF_VARIANTS, build_frbpff_model
from utils.utils import AvgMeter, adjust_lr, clip_gradient


class FRBPFFTrainer(MPECTrainer):
    """Reuse the audited MPEC data protocol with an FRB-PFF model and losses."""

    def _initialize_model(self):
        self.model = build_frbpff_model(
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
        decoder_parameters = total_parameters - backbone_parameters
        manifest = {
            'study': 'TMI FRB-PFF controlled progressive-decoder screen',
            'variant': self.config.variant,
            'frozen_before_training': True,
            'model': {
                'backbone': self.config.backbone,
                'decoder': self.model.decoder_variant,
                'total_parameters': total_parameters,
                'backbone_parameters': backbone_parameters,
                'adapter_and_decoder_parameters': decoder_parameters,
                'frequency_split': 'parameter-free 3x3 local mean residual',
                'boundary_injection_initialization': 'zero',
                'boundary_injection_bound': '[-0.5, 0.5]',
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
                'boundary_metric': (
                    'segmentation boundary F1 at threshold 0.5 with '
                    '{}-pixel tolerance at 352x352; reporting only'.format(
                        self.config.boundary_tolerance
                    )
                ),
                'far_polypgen_or_five_tests_used': False,
            },
            'research_design': {
                'boundary_aux': (
                    'Plain-PFF segmentation graph plus auxiliary high-frequency '
                    'boundary supervision'
                ),
                'dual_band': (
                    'separate progressive low-frequency region and '
                    'high-frequency boundary streams with bounded injection'
                ),
                'frb_full': (
                    'dual-band streams plus coarse-prediction uncertainty and '
                    'boundary-probability conditioned injection'
                ),
                'difference_from_bcmgpf': (
                    'explicit supervised boundary representation rather than '
                    'routing among convolution geometries'
                ),
            },
            'provenance': {
                'command': [sys.executable] + sys.argv,
                'working_directory': str(Path.cwd()),
                'code_sha256': {
                    'TrainFRBPFFStudent.py': sha256_file(Path(__file__).resolve()),
                    'lib/frbpff.py': sha256_file(
                        PROJECT_DIR / 'lib' / 'frbpff.py'
                    ),
                    'TrainMPECStudent.py': sha256_file(
                        PROJECT_DIR / 'TrainMPECStudent.py'
                    ),
                    'lib/pvt.py': sha256_file(PROJECT_DIR / 'lib' / 'pvt.py'),
                    'lib/quality_consistency.py': sha256_file(
                        PROJECT_DIR / 'lib' / 'quality_consistency.py'
                    ),
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

    def _forward(self, images):
        return self.model(images, return_details=True)

    @staticmethod
    def _boundary_target(mask, radius):
        binary = (mask > 0.5).float()
        kernel = 2 * int(radius) + 1
        dilated = F.max_pool2d(binary, kernel, stride=1, padding=radius)
        eroded = -F.max_pool2d(-binary, kernel, stride=1, padding=radius)
        return (dilated - eroded).clamp(0.0, 1.0)

    @staticmethod
    def _one_boundary_loss(logit, target):
        positive = target.sum()
        negative = target.numel() - positive
        positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)
        bce = F.binary_cross_entropy_with_logits(
            logit.float(), target.float(), pos_weight=positive_weight
        )
        probability = torch.sigmoid(logit.float())
        numerator = 2.0 * (probability * target).flatten(1).sum(dim=1)
        denominator = (
            probability.flatten(1).sum(dim=1)
            + target.flatten(1).sum(dim=1)
        )
        valid = target.flatten(1).sum(dim=1) > 0
        if bool(valid.any()):
            dice = 1.0 - (
                (numerator[valid] + 1.0) / (denominator[valid] + 1.0)
            ).mean()
        else:
            dice = bce.detach() * 0.0
        return 0.5 * (bce + dice)

    def _boundary_loss(self, boundary_outputs, masks):
        scaled_radius = max(
            1,
            int(round(
                self.config.boundary_radius
                * masks.shape[-1]
                / self.config.trainsize
            )),
        )
        target = self._boundary_target(masks, scaled_radius)
        loss = sum(
            self._one_boundary_loss(output, target)
            for output in boundary_outputs
        )
        return loss, target

    def _branch_loss(self, images, masks):
        details = self._forward(images)
        segmentation = self._segmentation_loss(
            details['outputs'], details['response'], masks
        )
        boundary, _target = self._boundary_loss(
            details['boundary_outputs'], masks
        )
        return segmentation, boundary, details

    def train_epoch(self, epoch):
        self.model.train()
        self.artifact_simulator.train()
        total_meter = AvgMeter()
        segmentation_meter = AvgMeter()
        boundary_meter = AvgMeter()
        normal_fraction_sum = 0.0
        gain_sum = np.zeros(3, dtype=np.float64)
        gate_sum = np.zeros(3, dtype=np.float64)
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
                    clean_seg, clean_boundary, _clean_details = (
                        self._branch_loss(images, masks)
                    )
                    degraded_seg, degraded_boundary, degraded_details = (
                        self._branch_loss(degraded_images, masks)
                    )
                    segmentation_loss = 0.5 * (clean_seg + degraded_seg)
                    boundary_loss = 0.5 * (
                        clean_boundary + degraded_boundary
                    )
                    total_loss = segmentation_loss + (
                        self.config.boundary_loss_weight * boundary_loss
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
                    normal_fraction_sum += float(
                        (presence_targets < 0.5).float().mean().item()
                    )
                    gain_sum += degraded_details['injection_gains'].detach().float().cpu().numpy()
                    gate_sum += degraded_details['gate_means'].detach().float().cpu().numpy()
                    batches += 1

            if step % 20 == 0 or step == len(self.train_loader):
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                    'Loss={:.4f}, Seg={:.4f}, Boundary={:.4f}'.format(
                        datetime.now(), epoch, self.config.epoch, step,
                        len(self.train_loader), total_meter.show(),
                        segmentation_meter.show(), boundary_meter.show(),
                    ),
                    flush=True,
                )

        denominator = max(batches, 1)
        result = {
            'total_loss': float(total_meter.show()),
            'segmentation_loss': float(segmentation_meter.show()),
            'boundary_loss': float(boundary_meter.show()),
            'sampled_normal_fraction': normal_fraction_sum / denominator,
        }
        for index in range(3):
            result['injection_gain_{}'.format(index)] = float(
                gain_sum[index] / denominator
            )
            result['gate_mean_{}'.format(index)] = float(
                gate_sum[index] / denominator
            )
        return result

    @staticmethod
    def _boundary_band(binary):
        binary = np.asarray(binary, dtype=np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        eroded = cv2.erode(binary, kernel, iterations=1)
        return (dilated != eroded).astype(np.uint8)

    @classmethod
    def _boundary_f1(cls, prediction, target, tolerance):
        prediction_boundary = cls._boundary_band(prediction)
        target_boundary = cls._boundary_band(target)
        predicted_count = float(prediction_boundary.sum())
        target_count = float(target_boundary.sum())
        if predicted_count == 0.0 and target_count == 0.0:
            return 1.0
        if predicted_count == 0.0 or target_count == 0.0:
            return 0.0
        kernel_size = 2 * int(tolerance) + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        target_tolerance = cv2.dilate(target_boundary, kernel, iterations=1)
        prediction_tolerance = cv2.dilate(
            prediction_boundary, kernel, iterations=1
        )
        precision = float(
            (prediction_boundary * target_tolerance).sum()
        ) / predicted_count
        recall = float(
            (target_boundary * prediction_tolerance).sum()
        ) / target_count
        return 2.0 * precision * recall / max(precision + recall, 1e-8)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        positive_dice = []
        positive_detected = []
        boundary_f1 = []
        latest_gains = np.zeros(3, dtype=np.float64)
        latest_gates = np.zeros(3, dtype=np.float64)

        for images, masks, _labels, _names, _sources in self.positive_validation_loader:
            images = images.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)
            details = self._forward(images)
            probability = torch.sigmoid(details['response'].float())
            intersection = (probability * masks).flatten(1).sum(dim=1)
            denominator = (
                probability.flatten(1).sum(dim=1)
                + masks.flatten(1).sum(dim=1)
            )
            dice = (2.0 * intersection + 1e-8) / (denominator + 1e-8)
            positive_dice.extend(dice.cpu().tolist())
            predictions = probability.squeeze(1).cpu().numpy() >= 0.5
            targets = masks.squeeze(1).cpu().numpy() >= 0.5
            for prediction, target in zip(predictions, targets):
                positive_detected.append(self._has_component(prediction))
                boundary_f1.append(
                    self._boundary_f1(
                        prediction, target, self.config.boundary_tolerance
                    )
                )
            latest_gains = details['injection_gains'].detach().float().cpu().numpy()
            latest_gates = details['gate_means'].detach().float().cpu().numpy()

        normal_false_positive = []
        normal_probability = []
        for images, _masks, _labels, _names, _sources in self.normal_validation_loader:
            images = images.cuda(non_blocking=True)
            details = self._forward(images)
            probability = torch.sigmoid(details['response'].float())
            normal_probability.extend(
                probability.mean(dim=(1, 2, 3)).cpu().tolist()
            )
            for value in probability.squeeze(1).cpu().numpy():
                normal_false_positive.append(self._has_component(value))

        dice = float(np.mean(positive_dice))
        specificity = 1.0 - float(np.mean(normal_false_positive))
        balanced_score = (
            2.0 * dice * specificity / max(dice + specificity, 1e-8)
        )
        result = {
            'positive_raw_soft_dice': dice,
            'positive_boundary_f1_tolerance_{}'.format(
                self.config.boundary_tolerance
            ): float(np.mean(boundary_f1)),
            'positive_component_sensitivity': float(np.mean(positive_detected)),
            'normal_gated_false_positive_rate': float(
                np.mean(normal_false_positive)
            ),
            'normal_gated_specificity': specificity,
            'normal_mean_probability': float(np.mean(normal_probability)),
            'balanced_harmonic_score': balanced_score,
        }
        for index in range(3):
            result['injection_gain_{}'.format(index)] = float(latest_gains[index])
            result['gate_mean_{}'.format(index)] = float(latest_gates[index])
        return result

    def train(self):
        print('#' * 20 + ' Start FRB-PFF Training ' + '#' * 20, flush=True)
        for epoch in range(1, self.config.epoch + 1):
            adjust_lr(
                self.optimizer,
                self.config.lr,
                epoch,
                self.config.decay_rate,
                self.config.decay_epoch,
            )
            training_metrics = self.train_epoch(epoch)
            validation_metrics = self.validate()
            print('Validation: {}'.format(validation_metrics), flush=True)
            self.save_epoch(epoch, training_metrics, validation_metrics)
        print(
            'Training completed: best epoch={}, score={:.6f}'.format(
                self.best_epoch, self.best_score
            ),
            flush=True,
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Controlled normal-aware FRB-PFF training screen'
    )
    parser.add_argument('--variant', choices=FRBPFF_VARIANTS, required=True)
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
        'boundary_radius': 2,
        'boundary_tolerance': 2,
        'training_seed': 42,
    }
    for field, expected in locked.items():
        if getattr(config, field) != expected:
            raise ValueError(
                'Controlled FRB-PFF screen locks {} to {}'.format(
                    field, expected
                )
            )


def verify_only(config):
    trainer = object.__new__(FRBPFFTrainer)
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
    model = build_frbpff_model(
        config.variant,
        channel=config.channel,
        backbone=config.backbone,
        load_backbone_weights=False,
    ).train()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if total_parameters > 3_600_000:
        raise RuntimeError(
            'FRB-PFF parameter budget exceeded: {:,}'.format(total_parameters)
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
    loss = segmentation + config.boundary_loss_weight * boundary
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite FRB-PFF verification loss')
    missing_gradients = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    # Zero-initialized boundary injection deliberately blocks segmentation
    # gradients into part of the detail pathway, but boundary supervision must
    # still cover every trainable tensor.  Any missing gradient is a bug.
    if missing_gradients:
        raise RuntimeError(
            'Missing FRB-PFF gradients: {}'.format(missing_gradients[:8])
        )
    print(
        'FRB-PFF verification passed: variant={}, parameters={}, '
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
        raise RuntimeError('CUDA is required for FRB-PFF training')
    FRBPFFTrainer(config).train()


if __name__ == '__main__':
    main()
