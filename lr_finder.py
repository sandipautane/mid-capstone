# Install if needed
#!pip install torch-lr-finder

import argparse
import os
import torch.optim as optim
from torch_lr_finder import LRFinder
import torch.nn as nn
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt
from models.model import resnet50
from dataloader import get_dataloaders, get_train_transforms, get_val_transforms, ImageNetTrain

def main():
    parser = argparse.ArgumentParser(description='Learning Rate Finder for ResNet50 on ImageNet')
    parser.add_argument('--data-dir', type=str, default='./data/imagenet',
                        help='path to ImageNet dataset')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size for training')
    parser.add_argument('--image-size', type=int, default=224, choices=[64, 128, 224],
                        help='image size for training (64, 128, or 224)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of data loading workers')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='number of classes in dataset')
    parser.add_argument('--drop-path-rate', type=float, default=0.0,
                        help='drop path rate for model')
    parser.add_argument('--use-blurpool', action='store_true',
                        help='use blur pooling in model')
    parser.add_argument('--start-lr', type=float, default=1e-7,
                        help='starting learning rate for LR finder')
    parser.add_argument('--end-lr', type=float, default=0.1,
                        help='ending learning rate for LR finder')
    parser.add_argument('--num-iter', type=int, default=300,
                        help='number of iterations for LR finder')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay for optimizer')
    parser.add_argument('--subset', action='store_true',
                        help='use subset of ImageNet for faster LR finding')
    parser.add_argument('--subset-size', type=int, default=10000,
                        help='size of subset to use')
    parser.add_argument('--save-plot', type=str, default='lr_finder_plot.png',
                        help='path to save LR finder plot')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("LR Finder Configuration")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Drop path rate: {args.drop_path_rate}")
    print(f"Use blur pool: {args.use_blurpool}")
    print(f"LR range: {args.start_lr} to {args.end_lr}")
    print(f"Iterations: {args.num_iter}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Using subset: {args.subset}")
    if args.subset:
        print(f"Subset size: {args.subset_size}")
    print("=" * 60)

    # Determine actual number of classes from dataset if using annotations
    train_annotations_dir = os.path.join(args.data_dir, 'Annotations', 'CLS-LOC', 'train')
    train_dir = os.path.join(args.data_dir, 'Data', 'CLS-LOC', 'train')

    if os.path.exists(train_annotations_dir) and os.path.exists(train_dir):
        print("Detecting number of classes from XML annotations...")
        temp_ds = ImageNetTrain(train_dir, train_annotations_dir, transform=None)
        num_classes = temp_ds.get_num_classes()
        print(f"Detected {num_classes} classes from dataset")
    else:
        num_classes = args.num_classes
        print(f"Using specified number of classes: {num_classes}")

    # Create fresh model and optimizer
    print("\nInitializing model...")
    model = resnet50(
        num_classes=num_classes,
        drop_path_rate=args.drop_path_rate,
        use_blurpool=args.use_blurpool
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.start_lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader, _ = get_dataloaders(
        data_path=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        subset_size=args.subset_size if args.subset else None,
        val_subset_size=args.subset_size // 10 if args.subset else None,
        use_subset=args.subset
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Batches: {len(train_loader)}")

    # Run LR Finder
    print("\nRunning LR Finder...")
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(
        train_loader,
        end_lr=args.end_lr,
        num_iter=args.num_iter,
        step_mode='exp'
    )

    # Plot and save results
    print(f"\nGenerating and saving plot to {args.save_plot}...")
    _, ax = plt.subplots(figsize=(10, 6))
    lr_finder.plot(ax=ax)
    plt.savefig(args.save_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved successfully!")

    # Get suggested LR
    try:
        suggested_lr = lr_finder.get_steepest_gradient()
        print(f"\nSuggested LR (steepest gradient): {suggested_lr:.2e}")
    except:
        print("\nCould not automatically determine suggested LR. Please review the plot.")

    # Reset model
    lr_finder.reset()

    print("\nLR Finder complete!")
    print(f"Download the plot: {args.save_plot}")
    print("=" * 60)

if __name__ == "__main__":
    main()