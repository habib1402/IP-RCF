"""
PolypPVT Dataloader Module
==========================

This module provides comprehensive data loading utilities for polyp segmentation tasks:
- Training dataset with configurable augmentations
- Test dataset for evaluation
- Robust file handling and validation
- Synchronized transformations for image-mask pairs
- Support for multiple image formats
"""

import os
import random
import logging
from typing import Tuple, Optional, List, Union
from pathlib import Path

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class PolypDataset(data.Dataset):
    """
    PyTorch Dataset for polyp segmentation training.
    
    This dataset class handles loading and preprocessing of polyp images and corresponding
    segmentation masks with support for data augmentation and validation.
    
    Args:
        image_root (str): Root directory containing training images
        gt_root (str): Root directory containing ground truth masks
        trainsize (int): Target size for training images (will be resized to trainsize x trainsize)
        augmentations (bool): Whether to apply data augmentations
        valid_img_extensions (tuple): Valid image file extensions
        valid_mask_extensions (tuple): Valid mask file extensions
    """
    
    def __init__(
        self, 
        image_root: str, 
        gt_root: str, 
        trainsize: int, 
        augmentations: bool = False,
        valid_img_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp'),
        valid_mask_extensions: tuple = ('.png', '.tif', '.tiff')
    ):
        self.image_root = Path(image_root)
        self.gt_root = Path(gt_root)
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.valid_img_extensions = valid_img_extensions
        self.valid_mask_extensions = valid_mask_extensions
        
        # Validate input directories
        self._validate_directories()
        
        # Load and validate file paths
        self._load_file_paths()
        
        # Setup transforms
        self._setup_transforms()
        
        logging.info(f"Dataset initialized: {len(self.images)} image-mask pairs")
        logging.info(f"Augmentations: {'Enabled' if augmentations else 'Disabled'}")

    def _validate_directories(self):
        """Validate that input directories exist and are accessible."""
        if not self.image_root.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_root}")
        if not self.gt_root.exists():
            raise FileNotFoundError(f"Ground truth directory not found: {self.gt_root}")
        
        if not self.image_root.is_dir():
            raise NotADirectoryError(f"Image path is not a directory: {self.image_root}")
        if not self.gt_root.is_dir():
            raise NotADirectoryError(f"Ground truth path is not a directory: {self.gt_root}")

    def _load_file_paths(self):
        """Load and validate image and mask file paths."""
        # Find all valid image files
        image_files = []
        for ext in self.valid_img_extensions:
            image_files.extend(list(self.image_root.glob(f"*{ext}")))
            image_files.extend(list(self.image_root.glob(f"*{ext.upper()}")))
        
        # Find all valid mask files
        mask_files = []
        for ext in self.valid_mask_extensions:
            mask_files.extend(list(self.gt_root.glob(f"*{ext}")))
            mask_files.extend(list(self.gt_root.glob(f"*{ext.upper()}")))
        
        if len(image_files) == 0:
            raise ValueError(f"No valid images found in {self.image_root}")
        if len(mask_files) == 0:
            raise ValueError(f"No valid masks found in {self.gt_root}")
        
        # Convert to strings and sort
        self.images = sorted([str(f) for f in image_files])
        self.gts = sorted([str(f) for f in mask_files])
        
        # Filter and validate pairs
        self._filter_and_validate_pairs()
        
        self.size = len(self.images)
        if self.size == 0:
            raise ValueError("No valid image-mask pairs found after filtering")

    def _filter_and_validate_pairs(self):
        """Filter files to ensure valid image-mask pairs exist."""
        valid_images = []
        valid_gts = []
        
        # Create lookup dictionary for masks by filename (without extension)
        mask_lookup = {}
        for gt_path in self.gts:
            gt_name = Path(gt_path).stem
            mask_lookup[gt_name] = gt_path
        
        for img_path in self.images:
            img_name = Path(img_path).stem
            
            # Look for corresponding mask
            if img_name in mask_lookup:
                gt_path = mask_lookup[img_name]
                
                # Validate that both files can be opened and have matching dimensions
                try:
                    img = Image.open(img_path)
                    gt = Image.open(gt_path)
                    
                    if img.size == gt.size:
                        valid_images.append(img_path)
                        valid_gts.append(gt_path)
                    else:
                        logging.warning(f"Size mismatch: {img_path} {img.size} vs {gt_path} {gt.size}")
                        
                except Exception as e:
                    logging.warning(f"Error loading pair {img_path}, {gt_path}: {e}")
                    continue
            else:
                logging.warning(f"No corresponding mask found for image: {img_path}")
        
        self.images = valid_images
        self.gts = valid_gts
        
        logging.info(f"Filtered dataset: {len(self.images)} valid pairs out of {len(self.images)} images")

    def _setup_transforms(self):
        """Setup image and mask transformations."""
        # Base transforms (always applied)
        base_img_transforms = [
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        
        base_gt_transforms = [
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ]
        
        if self.augmentations:
            # Augmentation transforms (applied before base transforms)
            aug_transforms = [
                transforms.RandomRotation(90, expand=False),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5)
            ]
            
            self.img_transform = transforms.Compose(aug_transforms + base_img_transforms)
            self.gt_transform = transforms.Compose(aug_transforms + base_gt_transforms)
            
            logging.info("Data augmentations enabled: RandomRotation(90°), RandomVerticalFlip, RandomHorizontalFlip")
        else:
            self.img_transform = transforms.Compose(base_img_transforms)
            self.gt_transform = transforms.Compose(base_gt_transforms)
            
            logging.info("No data augmentations applied")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample.
        
        Args:
            index (int): Sample index
            
        Returns:
            tuple: (image_tensor, mask_tensor)
        """
        if index >= len(self.images):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.images)}")
        
        # Load image and mask
        try:
            image = self._load_rgb_image(self.images[index])
            gt = self._load_grayscale_image(self.gts[index])
        except Exception as e:
            logging.error(f"Error loading sample {index}: {e}")
            # Return a random valid sample instead
            return self.__getitem__(random.randint(0, len(self.images) - 1))
        
        # Apply synchronized transformations
        if self.augmentations:
            image, gt = self._apply_synchronized_transforms(image, gt)
        else:
            image = self.img_transform(image)
            gt = self.gt_transform(gt)
        
        return image, gt

    def _apply_synchronized_transforms(self, image: Image.Image, gt: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply synchronized transformations to image and mask.
        
        Args:
            image (PIL.Image): Input image
            gt (PIL.Image): Ground truth mask
            
        Returns:
            tuple: Transformed (image_tensor, mask_tensor)
        """
        # Generate random seed for synchronized transformations
        seed = np.random.randint(2147483647)
        
        # Apply transforms to image
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.img_transform(image)
        
        # Apply same transforms to mask
        random.seed(seed)
        torch.manual_seed(seed)
        gt = self.gt_transform(gt)
        
        return image, gt

    def _load_rgb_image(self, path: str) -> Image.Image:
        """
        Load an RGB image from file.
        
        Args:
            path (str): Path to image file
            
        Returns:
            PIL.Image: RGB image
        """
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except Exception as e:
            raise IOError(f"Error loading image {path}: {e}")

    def _load_grayscale_image(self, path: str) -> Image.Image:
        """
        Load a grayscale image from file.
        
        Args:
            path (str): Path to mask file
            
        Returns:
            PIL.Image: Grayscale image
        """
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('L')  # Convert to grayscale
        except Exception as e:
            raise IOError(f"Error loading mask {path}: {e}")

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.size

    def get_sample_info(self, index: int) -> dict:
        """
        Get information about a specific sample.
        
        Args:
            index (int): Sample index
            
        Returns:
            dict: Sample information including paths and sizes
        """
        if index >= len(self.images):
            raise IndexError(f"Index {index} out of range")
        
        img_path = self.images[index]
        gt_path = self.gts[index]
        
        try:
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            
            return {
                'index': index,
                'image_path': img_path,
                'mask_path': gt_path,
                'image_size': img.size,
                'mask_size': gt.size,
                'image_mode': img.mode,
                'mask_mode': gt.mode
            }
        except Exception as e:
            return {'index': index, 'error': str(e)}


class TestDataset:
    """
    Dataset class for testing/evaluation.
    
    This class provides sequential access to test images and their corresponding
    ground truth masks for evaluation purposes.
    
    Args:
        image_root (str): Root directory containing test images
        gt_root (str): Root directory containing ground truth masks
        testsize (int): Target size for test images
        valid_img_extensions (tuple): Valid image file extensions
        valid_mask_extensions (tuple): Valid mask file extensions
    """
    
    def __init__(
        self, 
        image_root: str, 
        gt_root: str, 
        testsize: int,
        valid_img_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp'),
        valid_mask_extensions: tuple = ('.png', '.tif', '.tiff')
    ):
        self.image_root = Path(image_root)
        self.gt_root = Path(gt_root)
        self.testsize = testsize
        self.valid_img_extensions = valid_img_extensions
        self.valid_mask_extensions = valid_mask_extensions
        self.index = 0
        
        # Validate directories
        self._validate_directories()
        
        # Load file paths
        self._load_file_paths()
        
        # Setup transforms
        self._setup_transforms()
        
        logging.info(f"Test dataset initialized: {self.size} samples")

    def _validate_directories(self):
        """Validate that input directories exist."""
        if not self.image_root.exists():
            raise FileNotFoundError(f"Test image directory not found: {self.image_root}")
        if not self.gt_root.exists():
            raise FileNotFoundError(f"Test mask directory not found: {self.gt_root}")

    def _load_file_paths(self):
        """Load and sort file paths."""
        # Find all valid image files
        image_files = []
        for ext in self.valid_img_extensions:
            image_files.extend(list(self.image_root.glob(f"*{ext}")))
            image_files.extend(list(self.image_root.glob(f"*{ext.upper()}")))
        
        # Find all valid mask files
        mask_files = []
        for ext in self.valid_mask_extensions:
            mask_files.extend(list(self.gt_root.glob(f"*{ext}")))
            mask_files.extend(list(self.gt_root.glob(f"*{ext.upper()}")))
        
        # Convert to strings and sort
        self.images = sorted([str(f) for f in image_files])
        self.gts = sorted([str(f) for f in mask_files])
        
        self.size = len(self.images)
        
        if self.size == 0:
            raise ValueError(f"No test images found in {self.image_root}")
        
        logging.info(f"Found {len(self.images)} test images and {len(self.gts)} masks")

    def _setup_transforms(self):
        """Setup test transforms (no augmentation)."""
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.gt_transform = transforms.ToTensor()

    def load_data(self) -> Tuple[torch.Tensor, Image.Image, str]:
        """
        Load the next test sample.
        
        Returns:
            tuple: (image_tensor, ground_truth_pil, filename)
        """
        if self.index >= self.size:
            raise StopIteration("No more test samples available")
        
        try:
            # Load and transform image
            image = self._load_rgb_image(self.images[self.index])
            image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            
            # Load ground truth (keep as PIL for evaluation)
            gt = self._load_grayscale_image(self.gts[self.index])
            
            # Extract filename
            filename = Path(self.images[self.index]).name
            if filename.lower().endswith('.jpg'):
                filename = filename[:-4] + '.png'  # Convert jpg to png extension
                
            self.index += 1
            
            return image_tensor, gt, filename
            
        except Exception as e:
            logging.error(f"Error loading test sample {self.index}: {e}")
            self.index += 1
            return self.load_data()  # Try next sample

    def _load_rgb_image(self, path: str) -> Image.Image:
        """Load RGB image from file."""
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except Exception as e:
            raise IOError(f"Error loading image {path}: {e}")

    def _load_grayscale_image(self, path: str) -> Image.Image:
        """Load grayscale image from file."""
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('L')
        except Exception as e:
            raise IOError(f"Error loading mask {path}: {e}")

    def reset(self):
        """Reset the dataset index to start from beginning."""
        self.index = 0

    def __len__(self) -> int:
        """Return the size of the test dataset."""
        return self.size


def get_loader(
    image_root: str,
    gt_root: str,
    batchsize: int,
    trainsize: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    augmentation: bool = False,
    drop_last: bool = True
) -> data.DataLoader:
    """
    Create a training data loader.
    
    Args:
        image_root (str): Path to training images
        gt_root (str): Path to training masks
        batchsize (int): Batch size
        trainsize (int): Training image size
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of worker processes
        pin_memory (bool): Whether to pin memory
        augmentation (bool): Whether to apply data augmentation
        drop_last (bool): Whether to drop the last incomplete batch
        
    Returns:
        torch.utils.data.DataLoader: Training data loader
    """
    try:
        dataset = PolypDataset(
            image_root=image_root,
            gt_root=gt_root,
            trainsize=trainsize,
            augmentations=augmentation
        )
        
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=batchsize,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=num_workers > 0
        )
        
        logging.info(f"Training data loader created: {len(dataset)} samples, "
                    f"batch_size={batchsize}, {len(data_loader)} batches per epoch")
        
        return data_loader
        
    except Exception as e:
        logging.error(f"Error creating training data loader: {e}")
        raise


def get_test_loader(
    image_root: str,
    gt_root: str,
    testsize: int = 352
) -> TestDataset:
    """
    Create a test data loader.
    
    Args:
        image_root (str): Path to test images
        gt_root (str): Path to test masks
        testsize (int): Test image size
        
    Returns:
        TestDataset: Test dataset instance
    """
    try:
        test_dataset = TestDataset(
            image_root=image_root,
            gt_root=gt_root,
            testsize=testsize
        )
        
        logging.info(f"Test data loader created: {len(test_dataset)} samples")
        return test_dataset
        
    except Exception as e:
        logging.error(f"Error creating test data loader: {e}")
        raise


# Legacy compatibility - keep original function name
def test_dataset(image_root: str, gt_root: str, testsize: int) -> TestDataset:
    """
    Legacy function for backward compatibility.
    
    Args:
        image_root (str): Path to test images
        gt_root (str): Path to test masks
        testsize (int): Test image size
        
    Returns:
        TestDataset: Test dataset instance
    """
    return get_test_loader(image_root, gt_root, testsize)


def validate_dataset_structure(image_root: str, gt_root: str) -> dict:
    """
    Validate dataset structure and provide statistics.
    
    Args:
        image_root (str): Path to images
        gt_root (str): Path to masks
        
    Returns:
        dict: Dataset statistics and validation results
    """
    image_path = Path(image_root)
    gt_path = Path(gt_root)
    
    # Count files by extension
    stats = {
        'image_directory': str(image_path),
        'mask_directory': str(gt_path),
        'directories_exist': image_path.exists() and gt_path.exists(),
        'image_files': {},
        'mask_files': {},
        'total_images': 0,
        'total_masks': 0,
        'common_names': 0,
        'size_mismatches': 0,
        'errors': []
    }
    
    if not stats['directories_exist']:
        stats['errors'].append("One or both directories do not exist")
        return stats
    
    # Count image files
    valid_img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    for ext in valid_img_extensions:
        count = len(list(image_path.glob(f"*{ext}")) + list(image_path.glob(f"*{ext.upper()}")))
        if count > 0:
            stats['image_files'][ext] = count
            stats['total_images'] += count
    
    # Count mask files
    valid_mask_extensions = ('.png', '.tif', '.tiff')
    for ext in valid_mask_extensions:
        count = len(list(gt_path.glob(f"*{ext}")) + list(gt_path.glob(f"*{ext.upper()}")))
        if count > 0:
            stats['mask_files'][ext] = count
            stats['total_masks'] += count
    
    # Check for common names
    try:
        temp_dataset = PolypDataset(str(image_path), str(gt_path), 224, False)
        stats['common_names'] = len(temp_dataset)
    except Exception as e:
        stats['errors'].append(f"Error creating temporary dataset: {e}")
    
    return stats


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Example dataset validation
    image_root = "/path/to/images"
    gt_root = "/path/to/masks"
    
    try:
        stats = validate_dataset_structure(image_root, gt_root)
        print("Dataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error validating dataset: {e}")