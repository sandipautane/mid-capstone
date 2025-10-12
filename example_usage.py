#!/usr/bin/env python3
"""
Example usage of the ImageNet dataloader and training script.

This script demonstrates how to use the ImageNet dataloader with the updated ResNet50 model.
"""

import torch
from dataloader import get_imagenet_loaders, get_imagenet_subset_loaders
from models.resnet import ResNet50

def main():
    print("ImageNet Dataloader Example")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Example 1: Using ImageNet subset (for testing/development)
    print("\n1. Loading ImageNet subset (10,000 samples)...")
    try:
        train_loader, val_loader = get_imagenet_subset_loaders(
            data_dir='./data/imagenet',
            batch_size=32,  # Smaller batch for demo
            num_workers=4,
            subset_size=1000  # Very small subset for demo
        )
        
        print(f"Train loader: {len(train_loader)} batches")
        print(f"Val loader: {len(val_loader)} batches")
        
        # Get a sample batch
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}: Images shape: {images.shape}, Labels shape: {labels.shape}")
            if batch_idx >= 2:  # Show only first 3 batches
                break
                
    except Exception as e:
        print(f"Error loading ImageNet subset: {e}")
        print("Make sure ImageNet dataset is available at ./data/imagenet/")
    
    # Example 2: Initialize ResNet50 model for ImageNet
    print("\n2. Initializing ResNet50 for ImageNet...")
    model = ResNet50(num_classes=1000, input_size=224)
    model = model.to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Model output shape: {output.shape}")
    
    print("\n3. Training script usage:")
    print("To train on ImageNet subset:")
    print("python train.py --subset --subset-size 10000 --epochs 10 --batch-size 64")
    print()
    print("To train on full ImageNet:")
    print("python train.py --epochs 90 --batch-size 256 --data-dir /path/to/imagenet")
    print()
    print("For more options, run: python train.py --help")

if __name__ == "__main__":
    main()
