"""
PolypPVT Training Script
========================

This script provides a comprehensive training pipeline for the PolypPVT model
for polyp segmentation, including:
- Custom structure loss function combining BCE and IoU
- Multi-scale training with different image sizes
- Leakage-free validation on a held-out TrainDataset split
- Model checkpointing and best model selection
- Training visualization and logging
"""
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import hashlib
import json
import logging
import platform
import random
import socket
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import math
from torch.utils.data import DataLoader, Subset

# Import custom modules
from lib.pvt import (
    DECODER_DESCRIPTIONS,
    PolypPVT,
    SUPPORTED_BACKBONES,
    SUPPORTED_DECODERS,
)
from utils.dataloader import PolypDataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter


PROJECT_DIR = Path(__file__).resolve().parent
SEGMENTATION_ROOT = PROJECT_DIR.parents[1]
# Release: dataset root is configurable; it must contain TrainDataset/ and TestDataset/.
DATASET_ROOT = Path(os.environ.get('POLYP_DATA_ROOT', str(PROJECT_DIR / 'data')))


def sha256_file(path):
    """Return a content hash for code provenance in experiment manifests."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def set_reproducibility(seed):
    """Seed Python, NumPy and PyTorch without forcing unsupported CUDA ops."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    """Give every loader worker a deterministic Python/NumPy seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class StructureLoss:
    """
    Structure Loss combining weighted BCE and weighted IoU.
    
    This loss function is specifically designed for segmentation tasks
    where boundary regions are given higher importance through weighting.
    """
    
    @staticmethod
    def compute(pred, mask):
        """
        Compute structure loss.
        
        Args:
            pred (torch.Tensor): Predicted logits [B, 1, H, W]
            mask (torch.Tensor): Ground truth mask [B, 1, H, W]
            
        Returns:
            torch.Tensor: Combined structure loss
        """
        # Create weight map emphasizing boundaries
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
        )
        
        # Weighted Binary Cross Entropy
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        # Weighted IoU Loss
        pred_sigmoid = torch.sigmoid(pred)
        inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
        union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()


class TrainingVisualizer:
    """
    Handles training visualization and plotting.
    """
    
    def __init__(self, datasets):
        """
        Initialize visualizer.
        
        Args:
            datasets (list): List of dataset names to track
        """
        self.datasets = datasets
        self.plot_data = {dataset: [] for dataset in datasets}
        self.colors = ['red', 'lawngreen', 'lime', 'gold', 'm', 'plum', 'blue']
        self.line_styles = ['-', '--']
    
    def update_scores(self, scores_dict):
        """
        Update plot data with new scores.
        
        Args:
            scores_dict (dict): Dictionary mapping dataset names to scores
        """
        for dataset, score in scores_dict.items():
            if dataset in self.plot_data:
                self.plot_data[dataset].append(score)
    
    def plot_training_curves(self, save_path='eval.png', baseline_scores=None):
        """
        Plot training curves for all datasets.
        
        Args:
            save_path (str): Path to save the plot
            baseline_scores (dict): Baseline scores to plot as horizontal lines
        """
        plt.figure(figsize=(12, 8))
        
        if baseline_scores is None:
            baseline_scores = {}
        
        for i, dataset in enumerate(self.datasets):
            if dataset in self.plot_data and self.plot_data[dataset]:
                # Plot training curve
                plt.plot(
                    self.plot_data[dataset], 
                    label=dataset,
                    color=self.colors[i % len(self.colors)],
                    linestyle=self.line_styles[(i + 1) % 2],
                    linewidth=2
                )
                
                # Plot baseline if available
                if dataset in baseline_scores:
                    plt.axhline(
                        y=baseline_scores[dataset],
                        color=self.colors[i % len(self.colors)],
                        linestyle=':',
                        alpha=0.7,
                        label=f'{dataset} baseline'
                    )
        
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Dice Score", fontsize=12)
        plt.title('Training Progress - Dice Scores by Dataset', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class PolypPVTTrainer:
    """
    Main trainer class for PolypPVT model.
    """
    
    def __init__(self, config):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object with training parameters
        """
        self.config = config
        self.best_score = 0.0
        self.structure_loss = StructureLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.amp)
        
        # Benchmark test sets are never consulted during training. Checkpoint
        # selection uses only a deterministic split of TrainDataset.
        self.visualizer = TrainingVisualizer(['validation'])
        
        # Setup logging
        self._setup_logging()
        
        # Initialize model and optimizer
        self._initialize_model()
        self._initialize_optimizer()
        
        # Setup data loaders
        self._setup_data_loaders()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        os.makedirs(self.config.train_save, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(self.config.train_save, 'train.log'),
            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
            level=logging.INFO,
            filemode='a',
            datefmt='%Y-%m-%d %I:%M:%S %p'
        )
    
    def _initialize_model(self):
        """Initialize the PolypPVT model."""
        self.model = PolypPVT(
            channel=self.config.channel,
            backbone=self.config.backbone,
            decoder_variant=self.config.decoder_variant,
            backbone_weights=self.config.backbone_weights,
            load_backbone_weights=True,
        )
        
        # Use DataParallel if multiple GPUs are available
        # if torch.cuda.device_count() > 1:
        #     self.model = torch.nn.DataParallel(self.model)
        
        # Move model to GPU       
        self.model = self.model.cuda()
        
        # Log model information
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(
            f"Model initialized - backbone={self.config.backbone}, "
            f"decoder={self.config.decoder_variant}, "
            f"Total params: {total_params:,}, Trainable: {trainable_params:,}"
        )
    
    def _initialize_optimizer(self):
        """Initialize optimizer based on configuration."""
        backbone_params = list(self.model.backbone.parameters())
        backbone_param_ids = {id(parameter) for parameter in backbone_params}
        decoder_params = [
            parameter for parameter in self.model.parameters()
            if id(parameter) not in backbone_param_ids
        ]
        params = [
            {
                'params': backbone_params,
                'lr': self.config.lr * self.config.backbone_lr_mult,
                'lr_scale': self.config.backbone_lr_mult,
            },
            {'params': decoder_params, 'lr': self.config.lr, 'lr_scale': 1.0},
        ]
        
        if self.config.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(params, self.config.lr, weight_decay=1e-4)
        else:
            self.optimizer = torch.optim.SGD(params, self.config.lr, weight_decay=1e-4, momentum=0.9)
        
        logging.info(f"Optimizer initialized: {self.optimizer}")
    
    def _setup_data_loaders(self):
        """Create deterministic, source-stratified train/validation loaders."""
        image_root = os.path.join(self.config.train_path, 'images/')
        gt_root = os.path.join(self.config.train_path, 'masks/')

        if not 0.0 < self.config.val_ratio < 0.5:
            raise ValueError('--val_ratio must be greater than 0 and less than 0.5')

        train_dataset = PolypDataset(
            image_root=image_root,
            gt_root=gt_root,
            trainsize=self.config.trainsize,
            augmentations=self.config.augmentation,
        )
        validation_dataset = PolypDataset(
            image_root=image_root,
            gt_root=gt_root,
            trainsize=self.config.trainsize,
            augmentations=False,
        )

        # Numeric filenames are from CVC-ClinicDB and cju* filenames are from
        # Kvasir. Split each source independently to preserve its proportion.
        groups = {'clinicdb': [], 'kvasir': []}
        for index, image_path in enumerate(train_dataset.images):
            stem = Path(image_path).stem
            group = 'clinicdb' if stem[0].isdigit() else 'kvasir'
            groups[group].append(index)

        rng = np.random.default_rng(self.config.split_seed)
        validation_indices = []
        split_counts = {}
        for group_name, indices in groups.items():
            indices = np.asarray(indices, dtype=np.int64)
            rng.shuffle(indices)
            validation_count = max(1, int(round(len(indices) * self.config.val_ratio)))
            validation_indices.extend(indices[:validation_count].tolist())
            split_counts[group_name] = {
                'total': int(len(indices)),
                'validation': int(validation_count),
                'training': int(len(indices) - validation_count),
            }

        validation_set = set(validation_indices)
        training_indices = [
            index for index in range(len(train_dataset)) if index not in validation_set
        ]
        validation_indices = sorted(validation_indices)

        loader_generator = torch.Generator()
        loader_generator.manual_seed(self.config.training_seed)
        loader_options = {
            'num_workers': self.config.num_workers,
            'pin_memory': True,
            'persistent_workers': self.config.num_workers > 0,
            'worker_init_fn': seed_worker,
        }
        self.train_loader = DataLoader(
            Subset(train_dataset, training_indices),
            batch_size=self.config.batchsize,
            shuffle=True,
            drop_last=True,
            generator=loader_generator,
            **loader_options,
        )
        self.validation_loader = DataLoader(
            Subset(validation_dataset, validation_indices),
            batch_size=self.config.batchsize,
            shuffle=False,
            drop_last=False,
            **loader_options,
        )

        os.makedirs(self.config.train_save, exist_ok=True)
        manifest = {
            'seed': self.config.split_seed,
            'validation_ratio': self.config.val_ratio,
            'training_samples': len(training_indices),
            'validation_samples': len(validation_indices),
            'source_counts': split_counts,
            'validation_files': [
                Path(validation_dataset.images[index]).name
                for index in validation_indices
            ],
        }
        with open(os.path.join(self.config.train_save, 'validation_split.json'), 'w') as handle:
            json.dump(manifest, handle, indent=2)

        backbone_params = sum(p.numel() for p in self.model.backbone.parameters())
        total_params = sum(p.numel() for p in self.model.parameters())
        if (
            self.config.backbone in {'pvtv2_b0', 'pvtv2_b1'}
            and self.config.decoder_variant == 'pff_baseline'
        ):
            experiment_purpose = 'controlled compact-student backbone screening'
        elif self.config.decoder_variant != 'mpfl_cfm_cam':
            experiment_purpose = 'controlled progressive-fusion decoder ablation'
        else:
            experiment_purpose = (
                'controlled backbone study and response-level KD teacher preparation'
            )

        experiment = {
            'purpose': experiment_purpose,
            'backbone': self.model.backbone_name,
            'backbone_pretrained_weights': self.model.backbone_weights_path,
            'backbone_native_channels': list(self.model.native_channels),
            'common_projected_channels': [self.config.channel] * 4,
            'decoder_variant': self.model.decoder_variant,
            'decoder': DECODER_DESCRIPTIONS[self.model.decoder_variant],
            'response_logit': 'raw element-wise sum of out1, out2 and out3',
            'parameter_counts': {
                'total': total_params,
                'backbone': backbone_params,
                'projection_and_decoder': total_params - backbone_params,
            },
            'training': {
                'epochs': self.config.epoch,
                'optimizer': self.config.optimizer,
                'learning_rate': self.config.lr,
                'backbone_lr_multiplier': self.config.backbone_lr_mult,
                'batch_size': self.config.batchsize,
                'image_size': self.config.trainsize,
                'multi_scale_rates': [0.75, 1.0, 1.25],
                'augmentation': self.config.augmentation,
                'amp': self.config.amp,
                'training_seed': self.config.training_seed,
                'split_seed': self.config.split_seed,
            },
            'selection_rule': 'best held-out validation soft Dice; test sets are not used during training',
            'benchmark_test_sets_consulted': False,
            'provenance': {
                'command': [sys.executable] + sys.argv,
                'working_directory': str(Path.cwd()),
                'code_sha256': {
                    'Train.py': sha256_file(Path(__file__).resolve()),
                    'lib/pvt.py': sha256_file(
                        Path(__file__).resolve().parent / 'lib' / 'pvt.py'
                    ),
                },
                'hostname': socket.gethostname(),
                'python': platform.python_version(),
                'torch': torch.__version__,
                'cuda_runtime': torch.version.cuda,
                'cudnn': torch.backends.cudnn.version(),
                'gpu': torch.cuda.get_device_name(0),
                'visible_physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
            },
        }
        with open(os.path.join(self.config.train_save, 'experiment_config.json'), 'w') as handle:
            json.dump(experiment, handle, indent=2)

        self.total_step = len(self.train_loader)
        print(
            f"Backbone: {self.model.backbone_name}; "
            f"pretrained weights: {self.model.backbone_weights_path}"
        )
        print(
            f"Decoder: {self.model.decoder_variant} - "
            f"{DECODER_DESCRIPTIONS[self.model.decoder_variant]}"
        )
        print(
            f"Fair split: {len(training_indices)} training / "
            f"{len(validation_indices)} validation samples "
            f"(seed={self.config.split_seed})"
        )
        logging.info(
            f"Fair split initialized - train={len(training_indices)}, "
            f"validation={len(validation_indices)}, seed={self.config.split_seed}, "
            f"steps={self.total_step}"
        )
    
    def train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch (int): Current epoch number
        """
        self.model.train()
        size_rates = [0.75, 1.0, 1.25]  # for multi-scale training
        loss_record = AvgMeter()
        
        for i, pack in enumerate(self.train_loader, start=1):
            for rate in size_rates:
                self.optimizer.zero_grad()
                
                # Prepare data
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                
                # Multi-scale training
                if rate != 1.0:
                    trainsize = int(round(self.config.trainsize * rate / 32) * 32)
                    images = F.interpolate(images, size=(trainsize, trainsize), 
                                         mode='bilinear', align_corners=True)
                    gts = F.interpolate(gts, size=(trainsize, trainsize), 
                                      mode='bilinear', align_corners=True)
                
                with torch.cuda.amp.autocast(enabled=self.config.amp):
                    p1, p2, p3 = self.model(images)
                    loss_p1 = self.structure_loss.compute(p1, gts)
                    loss_p2 = self.structure_loss.compute(p2, gts)
                    loss_p3 = self.structure_loss.compute(p3, gts)
                    total_loss = loss_p1 + loss_p2 + loss_p3

                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_gradient(self.optimizer, self.config.clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Record loss for the base scale only
                if rate == 1.0:
                    loss_record.update(total_loss.data, self.config.batchsize)
            
            # Log progress
            if i % 20 == 0 or i == self.total_step:
                print(f'{datetime.now()} Epoch [{epoch:03d}/{self.config.epoch:03d}], '
                      f'Step [{i:04d}/{self.total_step:04d}], Loss: {loss_record.show():.4f}')
    
    def validate(self):
        """Return mean soft Dice on the held-out TrainDataset split."""
        self.model.eval()
        dice_sum = 0.0
        sample_count = 0

        with torch.no_grad():
            for images, masks in self.validation_loader:
                images = images.cuda(non_blocking=True)
                masks = masks.cuda(non_blocking=True)
                outputs = self.model(images)
                prediction = torch.sigmoid(self.model.response_logits(outputs))
                prediction = F.interpolate(
                    prediction,
                    size=masks.shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )

                flat_prediction = prediction.flatten(1)
                pred_min = flat_prediction.min(dim=1, keepdim=True).values
                pred_max = flat_prediction.max(dim=1, keepdim=True).values
                flat_prediction = (flat_prediction - pred_min) / (
                    pred_max - pred_min + 1e-8
                )
                flat_masks = masks.flatten(1)
                intersection = (flat_prediction * flat_masks).sum(dim=1)
                dice = (2.0 * intersection + 1e-8) / (
                    flat_prediction.sum(dim=1) + flat_masks.sum(dim=1) + 1e-8
                )
                dice_sum += dice.sum().item()
                sample_count += images.size(0)

        return dice_sum / max(sample_count, 1)

    def evaluate_and_save(self, epoch):
        """Evaluate on validation only and maintain last/best checkpoints."""
        save_path = self.config.train_save
        os.makedirs(save_path, exist_ok=True)

        torch.save(
            self.model.state_dict(),
            os.path.join(save_path, 'PolypPVT_last.pth'),
        )

        validation_score = self.validate()
        self.visualizer.update_scores({'validation': validation_score})
        print(f'Validation Dice: {validation_score:.4f}')
        logging.info(f'Epoch {epoch}, validation Dice: {validation_score:.4f}')

        if validation_score > self.best_score:
            self.best_score = validation_score
            torch.save(
                self.model.state_dict(),
                os.path.join(save_path, 'PolypPVT_best.pth'),
            )
            print(f'New best model saved! Best score: {self.best_score:.4f}')
            logging.info(f'New best model saved! Best score: {self.best_score:.4f}')
    
    def train(self):
        """Main training loop."""
        print("#" * 20, "Start Training", "#" * 20)
        logging.info("Training started")
        
        for epoch in range(1, self.config.epoch + 1):
            # Adjust learning rate
            adjust_lr(
                self.optimizer,
                self.config.lr,
                epoch,
                self.config.decay_rate,
                self.config.decay_epoch,
            )
            
            # Train for one epoch
            self.train_epoch(epoch)
            
            # Evaluate and save
            if epoch % 1 == 0:  # Evaluate every epoch
                self.evaluate_and_save(epoch)
        
        # Generate final training plot
        self.visualizer.plot_training_curves(
            os.path.join(self.config.train_save, 'validation_curve.png')
        )
        logging.info("Training completed")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PolypPVT Training Script')
    
    # Training parameters
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--backbone_lr_mult', type=float, default=1.0,
                       help='Backbone learning-rate multiplier')
    parser.add_argument('--optimizer', type=str, default='AdamW', 
                       choices=['AdamW', 'SGD'], help='Optimizer type')
    parser.add_argument('--batchsize', type=int, default=8, help='Batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='Training image size')
    parser.add_argument('--channel', type=int, default=64,
                       help='Common projection and decoder width')
    parser.add_argument('--backbone', type=str, default='dinov3_convnext_base',
                       choices=SUPPORTED_BACKBONES,
                       help='Encoder used in the controlled backbone study')
    parser.add_argument('--decoder_variant', type=str, default='mpfl_cfm_cam',
                       choices=SUPPORTED_DECODERS,
                       help='Decoder used in the controlled architecture study')
    parser.add_argument('--backbone_weights', type=str, default=None,
                       help='Optional pretrained backbone checkpoint override')
    
    # Regularization
    parser.add_argument('--clip', type=float, default=0.5, help='Gradient clipping value')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='LR decay rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='LR decay epoch')
    
    # Data paths
    parser.add_argument('--train_path', type=str,
                       default=str(DATASET_ROOT / 'TrainDataset'),
                       help='Training dataset path')
    parser.add_argument('--train_save', type=str,
                       default=str(PROJECT_DIR / 'model_pth_runs' / 'PolypPVT'),
                       help='Model save path')
    parser.add_argument('--val_ratio', type=float, default=0.10,
                       help='Fraction of TrainDataset held out for validation')
    parser.add_argument('--split_seed', type=int, default=42,
                       help='Seed for deterministic train/validation splitting')
    parser.add_argument('--training_seed', type=int, default=42,
                       help='Seed for initialization, augmentation and data order')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Data loader worker processes')
    parser.add_argument('--amp', action='store_true',
                       help='Enable CUDA automatic mixed precision')
    
    # Training options
    parser.add_argument('--augmentation', action='store_true', default=True,
                       help='Enable data augmentation')
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    config = parse_arguments()
    set_reproducibility(config.training_seed)
    
    # Initialize trainer
    trainer = PolypPVTTrainer(config)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
