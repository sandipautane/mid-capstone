import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import numpy as np
from PIL import Image
import os
import random

# ImageNet normalization stats (standard for ImageNet-based datasets)
#TODO: change this to  imagenet normalization stats 
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Convert to 0..255 for fill_value
fill_value_255 = tuple(int(m * 255) for m in MEAN)

def get_train_transforms(image_size=64):
    return A.Compose([
        A.Resize(image_size, image_size) if image_size != 64 else A.NoOp(),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.CoarseDropout(max_holes=1, max_height=int(image_size*0.25), max_width=int(image_size*0.25),
                        min_holes=1, min_height=int(image_size*0.25), min_width=int(image_size*0.25),
                        fill_value=fill_value_255, p=0.5),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def get_val_transforms(image_size=64):
    return A.Compose([
        A.Resize(image_size, image_size) if image_size != 64 else A.NoOp(),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


class ImageNetVal(Dataset):
    """
    Custom validation dataset for ImageNet that reads from val.txt
    Supports both ILSVRC structure and simple train/val structure
    """
    def __init__(self, val_dir, val_txt_path, transform=None):
        self.val_dir = val_dir
        self.transform = transform

        # Read val.txt to create mapping
        self.samples = []
        with open(val_txt_path, 'r') as f:
            for line in f:
                img_name, class_idx = line.strip().split()
                img_path = os.path.join(val_dir, img_name + '.JPEG')
                self.samples.append((img_path, int(class_idx) - 1))  # Convert to 0-indexed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return img, label


class AlbuImageNet1K(Dataset):
    """
    ImageNet dataset wrapper with Albumentations transforms
    Supports both ILSVRC structure (Data/CLS-LOC/train) and simple train/val structure
    """
    def __init__(self, root, train=True, transform=None):
        # Check if root has ILSVRC structure (Data/CLS-LOC/)
        ilsvrc_train_path = os.path.join(root, 'Data', 'CLS-LOC', 'train')
        ilsvrc_val_path = os.path.join(root, 'Data', 'CLS-LOC', 'val')

        if os.path.exists(ilsvrc_train_path) and train:
            # ILSVRC structure
            self.ds = datasets.ImageFolder(ilsvrc_train_path)
            self.is_ilsvrc = True
        elif os.path.exists(ilsvrc_val_path) and not train:
            # ILSVRC structure - but for val, we'll use ImageNetVal class instead
            self.ds = datasets.ImageFolder(ilsvrc_val_path)
            self.is_ilsvrc = True
        else:
            # Simple train/val structure
            split = 'train' if train else 'val'
            self.ds = datasets.ImageFolder(os.path.join(root, split))
            self.is_ilsvrc = False

        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]  # PIL Image
        img = np.array(img)        # -> HWC uint8 RGB
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, label


class SubsetImageNet1K(Dataset):
    """ImageNet subset dataset that samples a specified number of images per class"""
    def __init__(self, root, train=True, transform=None, subset_size=None, val_subset_size=None):
        # Check if root has ILSVRC structure (Data/CLS-LOC/)
        ilsvrc_train_path = os.path.join(root, 'Data', 'CLS-LOC', 'train')
        ilsvrc_val_path = os.path.join(root, 'Data', 'CLS-LOC', 'val')

        if os.path.exists(ilsvrc_train_path) and train:
            # ILSVRC structure
            self.full_dataset = datasets.ImageFolder(ilsvrc_train_path)
        elif os.path.exists(ilsvrc_val_path) and not train:
            # ILSVRC structure
            self.full_dataset = datasets.ImageFolder(ilsvrc_val_path)
        else:
            # Simple train/val structure
            split = 'train' if train else 'val'
            self.full_dataset = datasets.ImageFolder(os.path.join(root, split))

        self.transform = transform

        # Create subset indices
        if subset_size is not None and train:
            self.indices = self._create_subset_indices(subset_size)
        elif val_subset_size is not None and not train:
            self.indices = self._create_subset_indices(val_subset_size)
        else:
            self.indices = list(range(len(self.full_dataset)))
    
    def _create_subset_indices(self, subset_size):
        """Create balanced subset indices across all classes"""
        # Get class indices
        class_indices = {}
        for idx, (_, label) in enumerate(self.full_dataset.samples):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # Calculate samples per class
        num_classes = len(class_indices)
        samples_per_class = subset_size // num_classes
        remaining_samples = subset_size % num_classes
        
        subset_indices = []
        for i, (label, indices) in enumerate(class_indices.items()):
            # Add extra sample to first few classes if there are remaining samples
            class_size = samples_per_class + (1 if i < remaining_samples else 0)
            
            # Randomly sample indices for this class
            if len(indices) >= class_size:
                sampled_indices = random.sample(indices, class_size)
            else:
                # If class has fewer samples than needed, use all samples
                sampled_indices = indices
            
            subset_indices.extend(sampled_indices)
        
        # Shuffle the final indices
        random.shuffle(subset_indices)
        return subset_indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img, label = self.full_dataset[actual_idx]  # PIL Image
        img = np.array(img)        # -> HWC uint8 RGB
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, label






def get_dataloaders(data_path, batch_size=128, image_size=64, num_workers=2,
                   subset_size=None, val_subset_size=None, use_subset=False):
    """
    Create train and validation dataloaders for ImageNet 1K
    Supports both ILSVRC structure and simple train/val structure

    Args:
        data_path: Path to ImageNet dataset directory (ILSVRC root or parent dir with train/val)
        batch_size: Batch size for training
        image_size: Target image size (64, 128, or 224)
        num_workers: Number of worker processes
        subset_size: Size of training subset (if None, uses full dataset)
        val_subset_size: Size of validation subset (if None, uses full dataset)
        use_subset: Whether to use subset functionality
    """
    train_tfms = get_train_transforms(image_size)
    val_tfms = get_val_transforms(image_size)

    # Check if we have ILSVRC structure with val.txt
    val_txt_path = os.path.join(data_path, 'ImageSets', 'CLS-LOC', 'val.txt')
    val_dir = os.path.join(data_path, 'Data', 'CLS-LOC', 'val')
    use_val_txt = os.path.exists(val_txt_path) and os.path.exists(val_dir)

    if use_subset and subset_size is not None:
        print(f"Creating subset datasets:")
        print(f"  Training subset size: {subset_size}")
        print(f"  Validation subset size: {val_subset_size or 'full'}")

        train_ds = SubsetImageNet1K(root=data_path, train=True, transform=train_tfms,
                                   subset_size=subset_size)

        # For validation, use ImageNetVal if val.txt exists, otherwise use SubsetImageNet1K
        if use_val_txt and val_subset_size is None:
            # Use val.txt for full validation set
            val_ds = ImageNetVal(val_dir, val_txt_path, transform=val_tfms)
        else:
            # Use subset for validation
            val_ds = SubsetImageNet1K(root=data_path, train=False, transform=val_tfms,
                                     val_subset_size=val_subset_size)
    else:
        train_ds = AlbuImageNet1K(root=data_path, train=True, transform=train_tfms)

        # For validation, prefer ImageNetVal if val.txt exists
        if use_val_txt:
            val_ds = ImageNetVal(val_dir, val_txt_path, transform=val_tfms)
        else:
            val_ds = AlbuImageNet1K(root=data_path, train=False, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader


def get_subset_dataloaders(data_path, subset_size=10000, batch_size=128, image_size=64, num_workers=2):
    """
    Convenience function to create subset dataloaders
    
    Args:
        data_path: Path to ImageNet dataset directory
        subset_size: Size of training subset (validation will be 10% of this)
        batch_size: Batch size for training
        image_size: Target image size (64, 128, or 224)
        num_workers: Number of worker processes
    """
    return get_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        subset_size=subset_size,
        val_subset_size=subset_size // 10,
        use_subset=True
    )







################################################################
