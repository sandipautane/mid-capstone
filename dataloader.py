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
import xml.etree.ElementTree as ET

# ImageNet normalization stats (standard for ImageNet-based datasets)
#TODO: change this to  imagenet normalization stats 
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Convert to 0..255 for fill_value
fill_value_255 = tuple(int(m * 255) for m in MEAN)

def get_train_transforms(image_size=64):
    return A.Compose([
        A.Resize(image_size, image_size),  # Always resize to ensure consistent image sizes
        A.HorizontalFlip(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=15, p=0.5),
        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        # A.CoarseDropout(max_holes=1, max_height=int(image_size*0.25), max_width=int(image_size*0.25),
        #                 min_holes=1, min_height=int(image_size*0.25), min_width=int(image_size*0.25),
        #                 fill_value=fill_value_255, p=0.5),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def get_val_transforms(image_size=64):
    return A.Compose([
        A.Resize(image_size, image_size),  # Always resize to ensure consistent image sizes
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


class ImageNetVal(Dataset):
    """
    Custom validation dataset for ImageNet that reads from XML annotations
    Supports ILSVRC structure with Annotations/CLS-LOC/val
    """
    def __init__(self, val_dir, annotations_dir, transform=None, synset_to_idx=None):
        """
        Args:
            val_dir: Path to validation images (e.g., Data/CLS-LOC/val)
            annotations_dir: Path to validation annotations (e.g., Annotations/CLS-LOC/val)
            transform: Albumentations transform
            synset_to_idx: Optional mapping from synset to index (shared from training set)
        """
        self.val_dir = val_dir
        self.annotations_dir = annotations_dir
        self.transform = transform

        # Collect all XML files and extract class labels
        self.samples = []
        synset_names = set()

        # Process each XML file in the annotations directory
        for xml_file in sorted(os.listdir(annotations_dir)):
            if not xml_file.endswith('.xml'):
                continue

            xml_path = os.path.join(annotations_dir, xml_file)

            # Parse XML to get class label
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Get the class name from the first object
                obj = root.find('object')
                if obj is not None:
                    class_name = obj.find('name').text
                    synset_names.add(class_name)

                    # Get image filename
                    filename = root.find('filename').text
                    img_path = os.path.join(val_dir, filename + '.JPEG')

                    self.samples.append((img_path, class_name))
            except Exception as e:
                print(f"Warning: Could not parse {xml_path}: {e}")
                continue

        # Use provided synset_to_idx or create new one
        if synset_to_idx is not None:
            self.synset_to_idx = synset_to_idx
        else:
            # Create synset to index mapping (sorted for consistency)
            unique_synsets = sorted(synset_names)
            self.synset_to_idx = {synset: idx for idx, synset in enumerate(unique_synsets)}

        self.num_classes = len(self.synset_to_idx)

        print(f"ImageNetVal: Found {len(self.samples)} samples with {self.num_classes} classes")

    def __len__(self):
        return len(self.samples)

    def get_num_classes(self):
        """Return the number of classes in the dataset"""
        return self.num_classes

    def __getitem__(self, idx):
        img_path, synset = self.samples[idx]

        # Convert synset to class index
        if synset not in self.synset_to_idx:
            raise ValueError(f"Synset {synset} not found in synset_to_idx mapping")

        label = self.synset_to_idx[synset]

        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return img, label


class ImageNetTrain(Dataset):
    """
    Custom training dataset for ImageNet that reads from XML annotations
    Supports ILSVRC structure with Annotations/CLS-LOC/train
    """
    def __init__(self, train_dir, annotations_dir, transform=None):
        """
        Args:
            train_dir: Path to training images (e.g., Data/CLS-LOC/train)
            annotations_dir: Path to training annotations (e.g., Annotations/CLS-LOC/train)
            transform: Albumentations transform
        """
        self.train_dir = train_dir
        self.annotations_dir = annotations_dir
        self.transform = transform

        # Collect all XML files and extract class labels
        self.samples = []
        synset_names = set()

        # Walk through annotation directory structure
        for synset in sorted(os.listdir(annotations_dir)):
            synset_dir = os.path.join(annotations_dir, synset)
            if not os.path.isdir(synset_dir):
                continue

            synset_names.add(synset)

            # Process each XML file in this synset directory
            for xml_file in os.listdir(synset_dir):
                if not xml_file.endswith('.xml'):
                    continue

                xml_path = os.path.join(synset_dir, xml_file)

                # Parse XML to get class label
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()

                    # Get the class name from the first object
                    obj = root.find('object')
                    if obj is not None:
                        class_name = obj.find('name').text

                        # Get image filename
                        filename = root.find('filename').text
                        img_path = os.path.join(train_dir, synset, filename + '.JPEG')

                        self.samples.append((img_path, class_name))
                except Exception as e:
                    print(f"Warning: Could not parse {xml_path}: {e}")
                    continue

        # Create synset to index mapping (sorted for consistency)
        unique_synsets = sorted(synset_names)
        self.synset_to_idx = {synset: idx for idx, synset in enumerate(unique_synsets)}
        self.num_classes = len(self.synset_to_idx)

        print(f"ImageNetTrain: Found {len(self.samples)} samples with {self.num_classes} classes")
        print(f"Classes: {unique_synsets[:5]}... (showing first 5)")

    def __len__(self):
        return len(self.samples)

    def get_num_classes(self):
        """Return the number of classes in the dataset"""
        return self.num_classes

    def __getitem__(self, idx):
        img_path, synset = self.samples[idx]

        # Convert synset to class index
        label = self.synset_to_idx[synset]

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

    def get_num_classes(self):
        """Return the number of classes in the dataset"""
        return len(self.ds.classes)

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

        # Initialize label mapping as identity (no remapping by default)
        self.label_mapping = None

        # Create subset indices
        if subset_size is not None and train:
            self.indices = self._create_subset_indices(subset_size)
        elif val_subset_size is not None and not train:
            self.indices = self._create_subset_indices(val_subset_size)
        else:
            self.indices = list(range(len(self.full_dataset)))
    
    def _create_subset_indices(self, subset_size):
        """Create balanced subset indices across all classes and create label remapping"""
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
        used_classes = []
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
            used_classes.append(label)

        # Create label mapping from original labels to contiguous range [0, num_classes)
        # This is CRITICAL to avoid out-of-bounds errors when using subsets
        self.label_mapping = {original_label: new_label for new_label, original_label in enumerate(sorted(used_classes))}

        # Shuffle the final indices
        random.shuffle(subset_indices)
        return subset_indices
    
    def __len__(self):
        return len(self.indices)

    def get_num_classes(self):
        """Return the number of classes in the subset"""
        if self.label_mapping is not None:
            return len(self.label_mapping)
        else:
            # Return number of classes in full dataset
            return len(self.full_dataset.classes)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img, label = self.full_dataset[actual_idx]  # PIL Image

        # Apply label remapping if it exists
        if self.label_mapping is not None:
            label = self.label_mapping[label]

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

    # Check if we have ILSVRC structure with Annotations
    train_annotations_dir = os.path.join(data_path, 'Annotations', 'CLS-LOC', 'train')
    val_annotations_dir = os.path.join(data_path, 'Annotations', 'CLS-LOC', 'val')
    train_dir = os.path.join(data_path, 'Data', 'CLS-LOC', 'train')
    val_dir = os.path.join(data_path, 'Data', 'CLS-LOC', 'val')
    use_train_annotations = os.path.exists(train_annotations_dir) and os.path.exists(train_dir)
    use_val_annotations = os.path.exists(val_annotations_dir) and os.path.exists(val_dir)

    if use_subset and subset_size is not None:
        print(f"Creating subset datasets:")
        print(f"  Training subset size: {subset_size}")
        print(f"  Validation subset size: {val_subset_size or 'full'}")

        train_ds = SubsetImageNet1K(root=data_path, train=True, transform=train_tfms,
                                   subset_size=subset_size)

        # For validation, use ImageNetVal if annotations exist, otherwise use SubsetImageNet1K
        if use_val_annotations and val_subset_size is None:
            # Use annotations for full validation set - share synset mapping with train
            val_ds = ImageNetVal(val_dir, val_annotations_dir, transform=val_tfms,
                               synset_to_idx=train_ds.label_mapping)
        else:
            # Use subset for validation
            val_ds = SubsetImageNet1K(root=data_path, train=False, transform=val_tfms,
                                     val_subset_size=val_subset_size)
    else:
        # For training, prefer ImageNetTrain if annotations exist
        if use_train_annotations:
            print(f"Using XML annotations for training data")
            train_ds = ImageNetTrain(train_dir, train_annotations_dir, transform=train_tfms)
        else:
            train_ds = AlbuImageNet1K(root=data_path, train=True, transform=train_tfms)

        # For validation, prefer ImageNetVal if annotations exist
        if use_val_annotations:
            print(f"Using XML annotations for validation data")
            # Share synset mapping with training dataset
            synset_to_idx = getattr(train_ds, 'synset_to_idx', None)
            val_ds = ImageNetVal(val_dir, val_annotations_dir, transform=val_tfms,
                               synset_to_idx=synset_to_idx)
        else:
            val_ds = AlbuImageNet1K(root=data_path, train=False, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, train_ds


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
