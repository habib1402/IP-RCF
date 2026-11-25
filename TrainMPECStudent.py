"""Controlled normal-aware MPEC screen for lightweight polyp segmentation.

All variants use the same PVTv2-B0 + Plain-PFF segmentation trunk, the same
positive/normal data, and paired clean/degraded supervision.  The only model
difference is the presence-calibration mechanism.  FAR-POLYP-SEG and PolypGen
are never loaded by this script.
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import random
import socket
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from Train import (
    DATASET_ROOT,
    PROJECT_DIR,
    StructureLoss,
    seed_worker,
    set_reproducibility,
    sha256_file,
)
from lib.mpec import build_mpec_model, forward_mpec_model
from lib.quality_consistency import EndoscopyArtifactSimulator
from utils.dataloader import PolypDataset
from utils.utils import AvgMeter, adjust_lr, clip_gradient


VARIANTS = ('data_control', 'deep_presence', 'mpec')
EXPECTED_POSITIVE_SPLIT_SHA256 = (
    '089132dd865ba4f532d89fba4084c0caf4dbfc442d36b61ad72bec7b152261a8'
)
EXPECTED_NORMAL_COUNTS = {
    'total': 1409,
    'training': 1127,
    'validation': 282,
}
FORBIDDEN_TOKENS = ('far-polyp-seg', 'polypgen', 'external_polyp')
NORMALIZE = transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
)


def file_hash(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(block_size), b''):
            digest.update(block)
    return digest.hexdigest()


class PositivePresenceDataset(Dataset):
    """Attach a presence label and stable name to a PolypDataset subset."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        image, mask = self.dataset[source_index]
        name = Path(self.dataset.images[source_index]).name
        return image, mask, torch.tensor(1.0), name, 'positive'


class NormalPresenceDataset(Dataset):
    """Load audited HyperKvasir normal frames with all-zero masks."""

    def __init__(self, records, input_size, augmentation):
        self.records = list(records)
        self.input_size = int(input_size)
        self.augmentation = bool(augmentation)

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _augment(image):
        angle = float(torch.empty(1).uniform_(-90.0, 90.0).item())
        image = TF.rotate(
            image, angle, interpolation=InterpolationMode.NEAREST, fill=0
        )
        if torch.rand(()) < 0.5:
            image = TF.vflip(image)
        if torch.rand(()) < 0.5:
            image = TF.hflip(image)
        return image

    def __getitem__(self, index):
        record = self.records[index]
        path = Path(record['path'])
        with Image.open(str(path)) as handle:
            image = handle.convert('RGB')
        image = TF.resize(
            image,
            (self.input_size, self.input_size),
            interpolation=InterpolationMode.BILINEAR,
        )
        image = TF.to_tensor(image)
        if self.augmentation:
            image = self._augment(image)
        image = NORMALIZE(image)
        mask = torch.zeros(1, self.input_size, self.input_size)
        name = '{}__{}'.format(record['class'], path.name)
        return image, mask, torch.tensor(0.0), name, record['class']


class MPECTrainer:
    def __init__(self, config):
        self.config = config
        self.structure_loss = StructureLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
        self.artifact_simulator = EndoscopyArtifactSimulator(
            config.artifact_strength
        ).cuda()
        self.best_score = -float('inf')
        self.best_epoch = 0
        self.history = []
        self._load_protocol()
        self._initialize_model()
        self._initialize_optimizer()
        self._setup_data_loaders()
        self._write_manifest()

    def _load_protocol(self):
        self.positive_split_path = Path(
            self.config.positive_split_manifest
        ).expanduser().resolve()
        self.normal_manifest_path = Path(
            self.config.normal_manifest
        ).expanduser().resolve()
        for path in (self.positive_split_path, self.normal_manifest_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        if file_hash(self.positive_split_path) != EXPECTED_POSITIVE_SPLIT_SHA256:
            raise RuntimeError('Unexpected positive validation split manifest')
        with self.positive_split_path.open() as handle:
            self.positive_split = json.load(handle)
        with self.normal_manifest_path.open() as handle:
            self.normal_manifest = json.load(handle)
        source_paths = [str(self.normal_manifest.get('root', ''))]
        for split in ('training', 'validation'):
            source_paths.extend(
                str(row.get('path', ''))
                for row in self.normal_manifest.get(split, [])
            )
        serialized_paths = '\n'.join(source_paths).lower()
        if any(token in serialized_paths for token in FORBIDDEN_TOKENS):
            raise RuntimeError(
                'Normal manifest contains a forbidden external-test source'
            )
        if self.normal_manifest.get('external_test_sources_used') != []:
            raise RuntimeError('Normal manifest lists an external test source')
        if self.normal_manifest.get('patient_identifiers_available') is not False:
            raise RuntimeError('Unexpected HyperKvasir patient-ID declaration')
        if self.normal_manifest.get('observed_total') != EXPECTED_NORMAL_COUNTS['total']:
            raise RuntimeError('Unexpected total normal-frame count')
        if len(self.normal_manifest.get('training', [])) != EXPECTED_NORMAL_COUNTS['training']:
            raise RuntimeError('Unexpected normal training count')
        if len(self.normal_manifest.get('validation', [])) != EXPECTED_NORMAL_COUNTS['validation']:
            raise RuntimeError('Unexpected normal validation count')
        for split in ('training', 'validation'):
            missing = [
                row['path'] for row in self.normal_manifest[split]
                if not Path(row['path']).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    'Missing HyperKvasir normal files: {}'.format(missing[:3])
                )

    def _initialize_model(self):
        self.model = build_mpec_model(
            variant=self.config.variant,
            channel=self.config.channel,
            backbone=self.config.backbone,
            backbone_weights=self.config.backbone_weights,
            load_backbone_weights=True,
            initial_calibration_strength=(
                self.config.initial_calibration_strength
            ),
        ).cuda()
        total = sum(parameter.numel() for parameter in self.model.parameters())
        print(
            'Model: variant={}, backbone={}, parameters={:,}'.format(
                self.config.variant, self.config.backbone, total
            ),
            flush=True,
        )

    def _initialize_optimizer(self):
        backbone_parameters = list(self.model.backbone.parameters())
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        head_parameters = [
            parameter for parameter in self.model.parameters()
            if id(parameter) not in backbone_ids
        ]
        groups = [
            {
                'params': backbone_parameters,
                'lr': self.config.lr * self.config.backbone_lr_mult,
                'lr_scale': self.config.backbone_lr_mult,
            },
            {
                'params': head_parameters,
                'lr': self.config.lr,
                'lr_scale': 1.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            groups, lr=self.config.lr, weight_decay=1e-4
        )

    def _positive_datasets(self):
        image_root = Path(self.config.train_path) / 'images'
        mask_root = Path(self.config.train_path) / 'masks'
        augmented = PolypDataset(
            str(image_root), str(mask_root), self.config.trainsize,
            augmentations=True,
        )
        validation = PolypDataset(
            str(image_root), str(mask_root), self.config.trainsize,
            augmentations=False,
        )
        validation_names = set(self.positive_split['validation_files'])
        training_indices = [
            index for index, path in enumerate(augmented.images)
            if Path(path).name not in validation_names
        ]
        validation_indices = [
            index for index, path in enumerate(validation.images)
            if Path(path).name in validation_names
        ]
        if len(training_indices) != 1305 or len(validation_indices) != 145:
            raise RuntimeError(
                'Positive split count mismatch: train={}, validation={}'
                .format(len(training_indices), len(validation_indices))
            )
        return (
            PositivePresenceDataset(augmented, training_indices),
            PositivePresenceDataset(validation, validation_indices),
        )

    def _setup_data_loaders(self):
        positive_train, positive_validation = self._positive_datasets()
        normal_train = NormalPresenceDataset(
            self.normal_manifest['training'], self.config.trainsize, True
        )
        normal_validation = NormalPresenceDataset(
            self.normal_manifest['validation'], self.config.trainsize, False
        )
        combined = ConcatDataset((positive_train, normal_train))
        weights = torch.cat((
            torch.full((len(positive_train),), 0.5 / len(positive_train)),
            torch.full((len(normal_train),), 0.5 / len(normal_train)),
        ))
        generator = torch.Generator().manual_seed(self.config.training_seed)
        samples_per_epoch = 2 * max(len(positive_train), len(normal_train))
        sampler = WeightedRandomSampler(
            weights,
            num_samples=samples_per_epoch,
            replacement=True,
            generator=generator,
        )
        loader_options = {
            'num_workers': self.config.num_workers,
            'pin_memory': True,
            'persistent_workers': self.config.num_workers > 0,
            'worker_init_fn': seed_worker,
        }
        self.train_loader = DataLoader(
            combined,
            batch_size=self.config.batchsize,
            sampler=sampler,
            drop_last=True,
            **loader_options,
        )
        self.positive_validation_loader = DataLoader(
            positive_validation,
            batch_size=self.config.batchsize,
            shuffle=False,
            drop_last=False,
            **loader_options,
        )
        self.normal_validation_loader = DataLoader(
            normal_validation,
            batch_size=self.config.batchsize,
            shuffle=False,
            drop_last=False,
            **loader_options,
        )
        self.data_counts = {
            'positive_training': len(positive_train),
            'positive_validation': len(positive_validation),
            'normal_training': len(normal_train),
            'normal_validation': len(normal_validation),
            'balanced_samples_per_epoch': samples_per_epoch,
            'steps_per_epoch': len(self.train_loader),
        }
        print('Data: {}'.format(self.data_counts), flush=True)

    def _write_manifest(self):
        total_parameters = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        backbone_parameters = sum(
            parameter.numel() for parameter in self.model.backbone.parameters()
        )
        presence_parameters = sum(
            parameter.numel() for name, parameter in self.model.named_parameters()
            if 'presence_head' in name or 'calibration_strength' in name
        )
        manifest = {
            'study': 'TMI MPEC normal-aware controlled screen',
            'variant': self.config.variant,
            'frozen_before_training': True,
            'model': {
                'backbone': self.config.backbone,
                'decoder': 'pff_baseline',
                'presence_mechanism': self.config.variant,
                'total_parameters': total_parameters,
                'backbone_parameters': backbone_parameters,
                'presence_parameters': presence_parameters,
                'initial_calibration_strength': (
                    self.config.initial_calibration_strength
                ),
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
                'presence_loss_weight': self.config.presence_loss_weight,
                'response_loss_weight': self.config.response_loss_weight,
                'multi_scale_rates': [0.75, 1.0, 1.25],
            },
            'checkpoint_selection': {
                'positive_metric': 'raw-sigmoid soft Dice',
                'normal_metric': (
                    'specificity of threshold-0.5 predictions with an '
                    '8-connected component >=100 pixels at 352x352'
                ),
                'score': 'harmonic mean of positive Dice and normal specificity',
                'far_or_polypgen_used': False,
            },
            'provenance': {
                'command': [sys.executable] + sys.argv,
                'working_directory': str(Path.cwd()),
                'code_sha256': {
                    'TrainMPECStudent.py': sha256_file(Path(__file__).resolve()),
                    'lib/mpec.py': sha256_file(PROJECT_DIR / 'lib' / 'mpec.py'),
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
        return forward_mpec_model(
            self.model, self.config.variant, images
        )

    def _segmentation_loss(self, outputs, response, masks):
        output_loss = sum(
            self.structure_loss.compute(output, masks) for output in outputs
        )
        response_loss = self.structure_loss.compute(response, masks)
        return output_loss + self.config.response_loss_weight * response_loss

    def _branch_loss(self, images, masks, presence_targets):
        outputs, response, presence_logit = self._forward(images)
        segmentation = self._segmentation_loss(outputs, response, masks)
        if presence_logit is None:
            presence = segmentation.detach() * 0.0
        else:
            presence = F.binary_cross_entropy_with_logits(
                presence_logit.float(), presence_targets.float()
            )
        return segmentation, presence, response, presence_logit

    def train_epoch(self, epoch):
        self.model.train()
        self.artifact_simulator.train()
        total_meter = AvgMeter()
        segmentation_meter = AvgMeter()
        presence_meter = AvgMeter()
        normal_fraction_sum = 0.0
        batches = 0
        for step, pack in enumerate(self.train_loader, start=1):
            images_cpu, masks_cpu, presence_cpu, _names, _sources = pack
            for rate in (0.75, 1.0, 1.25):
                self.optimizer.zero_grad()
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
                    clean_seg, clean_presence, _response, _presence = (
                        self._branch_loss(images, masks, presence_targets)
                    )
                    degraded_seg, degraded_presence, _response, _presence = (
                        self._branch_loss(
                            degraded_images, masks, presence_targets
                        )
                    )
                    segmentation_loss = 0.5 * (clean_seg + degraded_seg)
                    presence_loss = 0.5 * (
                        clean_presence + degraded_presence
                    )
                    total_loss = segmentation_loss + (
                        self.config.presence_loss_weight * presence_loss
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
                    presence_meter.update(presence_loss.detach(), batch_size)
                    normal_fraction_sum += float(
                        (presence_targets < 0.5).float().mean().item()
                    )
                    batches += 1
            if step % 20 == 0 or step == len(self.train_loader):
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                    'Loss={:.4f}, Seg={:.4f}, Presence={:.4f}'.format(
                        datetime.now(), epoch, self.config.epoch, step,
                        len(self.train_loader), total_meter.show(),
                        segmentation_meter.show(), presence_meter.show(),
                    ),
                    flush=True,
                )
        return {
            'total_loss': float(total_meter.show()),
            'segmentation_loss': float(segmentation_meter.show()),
            'presence_loss': float(presence_meter.show()),
            'sampled_normal_fraction': normal_fraction_sum / max(batches, 1),
        }

    @staticmethod
    def _has_component(probability, threshold=0.5, minimum_area=100):
        binary = np.asarray(probability >= threshold, dtype=np.uint8)
        labels, _label_map, stats, _centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if labels <= 1:
            return False
        return bool(np.any(stats[1:, cv2.CC_STAT_AREA] >= minimum_area))

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        positive_dice = []
        positive_detected = []
        positive_presence = []
        for images, masks, _labels, _names, _sources in self.positive_validation_loader:
            images = images.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)
            _outputs, response, presence_logit = self._forward(images)
            probability = torch.sigmoid(response.float())
            intersection = (probability * masks).flatten(1).sum(dim=1)
            denominator = probability.flatten(1).sum(dim=1) + masks.flatten(1).sum(dim=1)
            dice = (2.0 * intersection + 1e-8) / (denominator + 1e-8)
            positive_dice.extend(dice.cpu().tolist())
            for value in probability.squeeze(1).cpu().numpy():
                positive_detected.append(self._has_component(value))
            if presence_logit is not None:
                positive_presence.extend(
                    torch.sigmoid(presence_logit.float()).cpu().tolist()
                )

        normal_false_positive = []
        normal_probability = []
        normal_presence = []
        for images, _masks, _labels, _names, _sources in self.normal_validation_loader:
            images = images.cuda(non_blocking=True)
            _outputs, response, presence_logit = self._forward(images)
            probability = torch.sigmoid(response.float())
            normal_probability.extend(
                probability.mean(dim=(1, 2, 3)).cpu().tolist()
            )
            for value in probability.squeeze(1).cpu().numpy():
                normal_false_positive.append(self._has_component(value))
            if presence_logit is not None:
                normal_presence.extend(
                    torch.sigmoid(presence_logit.float()).cpu().tolist()
                )

        dice = float(np.mean(positive_dice))
        specificity = 1.0 - float(np.mean(normal_false_positive))
        balanced_score = (
            2.0 * dice * specificity / max(dice + specificity, 1e-8)
        )
        result = {
            'positive_raw_soft_dice': dice,
            'positive_component_sensitivity': float(np.mean(positive_detected)),
            'normal_gated_false_positive_rate': float(
                np.mean(normal_false_positive)
            ),
            'normal_gated_specificity': specificity,
            'normal_mean_probability': float(np.mean(normal_probability)),
            'balanced_harmonic_score': balanced_score,
            'positive_presence_mean': (
                float(np.mean(positive_presence)) if positive_presence else None
            ),
            'normal_presence_mean': (
                float(np.mean(normal_presence)) if normal_presence else None
            ),
            'calibration_strength': (
                float(self.model.calibration_strength.detach().cpu().item())
                if hasattr(self.model, 'calibration_strength') else 0.0
            ),
        }
        return result

    def save_epoch(self, epoch, training_metrics, validation_metrics):
        output = Path(self.config.train_save)
        torch.save(self.model.state_dict(), output / 'PolypPVT_last.pth')
        score = validation_metrics['balanced_harmonic_score']
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            torch.save(self.model.state_dict(), output / 'PolypPVT_best.pth')
            print(
                'New best checkpoint: score={:.6f}, Dice={:.6f}, '
                'specificity={:.6f}'.format(
                    score,
                    validation_metrics['positive_raw_soft_dice'],
                    validation_metrics['normal_gated_specificity'],
                ),
                flush=True,
            )
        row = {'epoch': epoch}
        row.update(training_metrics)
        row.update(validation_metrics)
        row['best_balanced_score'] = self.best_score
        row['best_epoch'] = self.best_epoch
        self.history.append(row)
        with (output / 'epoch_metrics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.history[0]))
            writer.writeheader()
            writer.writerows(self.history)
        with (output / 'best_validation.json').open('w') as handle:
            json.dump({
                'best_balanced_harmonic_score': self.best_score,
                'best_epoch': self.best_epoch,
                'checkpoint': str(output / 'PolypPVT_best.pth'),
                'latest_validation': validation_metrics,
            }, handle, indent=2)

    def train(self):
        print('#' * 20 + ' Start MPEC Training ' + '#' * 20, flush=True)
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
            print(
                'Validation: {}'.format(validation_metrics), flush=True
            )
            self.save_epoch(epoch, training_metrics, validation_metrics)
        print(
            'Training completed: best epoch={}, score={:.6f}'.format(
                self.best_epoch, self.best_score
            ),
            flush=True,
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Controlled normal-aware MPEC training screen'
    )
    parser.add_argument('--variant', choices=VARIANTS, required=True)
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
    parser.add_argument('--presence_loss_weight', type=float, default=1.0)
    parser.add_argument('--response_loss_weight', type=float, default=0.5)
    parser.add_argument('--initial_calibration_strength', type=float, default=0.10)
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
        'presence_loss_weight': 1.0,
        'response_loss_weight': 0.5,
        'initial_calibration_strength': 0.10,
        'training_seed': 42,
    }
    for field, expected in locked.items():
        if getattr(config, field) != expected:
            raise ValueError(
                'Controlled MPEC screen locks {} to {}'.format(field, expected)
            )


def verify_only(config):
    trainer = object.__new__(MPECTrainer)
    trainer.config = config
    trainer._load_protocol()
    positive_train, positive_validation = trainer._positive_datasets()
    normal_train = NormalPresenceDataset(
        trainer.normal_manifest['training'], config.trainsize, True
    )
    normal_validation = NormalPresenceDataset(
        trainer.normal_manifest['validation'], config.trainsize, False
    )
    model = build_mpec_model(
        config.variant,
        channel=config.channel,
        backbone=config.backbone,
        load_backbone_weights=False,
        initial_calibration_strength=config.initial_calibration_strength,
    ).eval()
    image, mask, label, name, source = normal_train[0]
    with torch.no_grad():
        outputs, response, presence = forward_mpec_model(
            model, config.variant, image.unsqueeze(0)
        )
    print(
        'MPEC verification passed: variant={}, positive_train={}, '
        'positive_validation={}, normal_train={}, normal_validation={}, '
        'sample={}, source={}, image={}, mask={}, label={}, outputs={}, '
        'response={}, presence={}'.format(
            config.variant, len(positive_train), len(positive_validation),
            len(normal_train), len(normal_validation), name, source,
            tuple(image.shape), tuple(mask.shape), float(label),
            [tuple(value.shape) for value in outputs], tuple(response.shape),
            None if presence is None else tuple(presence.shape),
        )
    )


def main():
    config = parse_arguments()
    validate_arguments(config)
    set_reproducibility(config.training_seed)
    if config.verify_only:
        verify_only(config)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for MPEC training')
    MPECTrainer(config).train()


if __name__ == '__main__':
    main()
