import os
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet


def get_imagenet_loaders(data_dir='./data/imagenet', batch_size=256, num_workers=8, 
                        image_size=224, val_split=0.1):
    """
    Load ImageNet 1K dataset with data augmentation.
    
    Args:
        data_dir (str): Path to ImageNet dataset directory
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        image_size (int): Size to resize images to (default: 224 for ImageNet)
        val_split (float): Fraction of training data to use for validation
    
    Returns:
        tuple: (trainloader, valloader, testloader)
    """
    
    # Define transforms for training data (with augmentation)
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Define transforms for validation/test data (no augmentation)
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create ImageNet dataset
    try:
        # Load training dataset
        train_dataset = ImageNet(
            root=data_dir, 
            split='train', 
            transform=transform_train,
            download=False  # Set to True if you want to download automatically
        )
        
        # Load validation dataset
        val_dataset = ImageNet(
            root=data_dir, 
            split='val', 
            transform=transform_val,
            download=False  # Set to True if you want to download automatically
        )
        
        print(f"Loaded ImageNet dataset:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        
    except Exception as e:
        print(f"Error loading ImageNet dataset: {e}")
        print("Please ensure ImageNet dataset is downloaded and placed in the correct directory.")
        print(f"Expected structure: {data_dir}/train/ and {data_dir}/val/")
        raise
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_imagenet_subset_loaders(data_dir='./data/imagenet', batch_size=256, num_workers=8,
                               image_size=224, subset_size=10000):
    """
    Create a subset of ImageNet for faster training/testing.
    This is useful for development and testing purposes.
    
    Args:
        data_dir (str): Path to ImageNet dataset directory
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        image_size (int): Size to resize images to
        subset_size (int): Number of samples to use from training set
    
    Returns:
        tuple: (trainloader, valloader)
    """
    from torch.utils.data import Subset, random_split
    
    # Define transforms (same as above)
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Load full datasets
        train_dataset = ImageNet(
            root=data_dir, 
            split='train', 
            transform=transform_train,
            download=False
        )
        
        val_dataset = ImageNet(
            root=data_dir, 
            split='val', 
            transform=transform_val,
            download=False
        )
        
        # Create subset of training data
        if subset_size < len(train_dataset):
            indices = torch.randperm(len(train_dataset))[:subset_size]
            train_dataset = Subset(train_dataset, indices)
        
        # Use smaller validation set for subset
        val_subset_size = min(len(val_dataset), subset_size // 10)
        if val_subset_size < len(val_dataset):
            indices = torch.randperm(len(val_dataset))[:val_subset_size]
            val_dataset = Subset(val_dataset, indices)
        
        print(f"Created ImageNet subset:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        
    except Exception as e:
        print(f"Error loading ImageNet dataset: {e}")
        print("Please ensure ImageNet dataset is downloaded and placed in the correct directory.")
        raise
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_imagenet_classes():
    """
    Get the list of ImageNet class names.
    
    Returns:
        list: List of ImageNet class names
    """
    try:
        # Load a small subset to get class names
        dataset = ImageNet(root='./data/imagenet', split='val', download=False)
        return dataset.classes
    except:
        # Fallback: return a subset of common ImageNet classes
        return [f"class_{i}" for i in range(1000)]
