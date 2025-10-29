"""
Additional Training Script for Fine-tuning from Checkpoint

This script allows flexible training from a checkpoint with custom:
- Learning rate and max learning rate
- Image size
- Number of epochs
- Always uses drop_path_rate=0.05 and blur_pool=True
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler

from models.model import resnet50
from dataloader import get_dataloaders, ImageNetTrain
from train_utils import train, test


def main():
    parser = argparse.ArgumentParser(description='Additional Training from Checkpoint with Custom Settings')

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to checkpoint to resume from')
    parser.add_argument('--data-dir', type=str, default='./data/imagenet',
                        help='path to ImageNet dataset')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, required=True,
                        help='number of additional epochs to train')
    parser.add_argument('--lr', type=float, required=True,
                        help='initial learning rate')
    parser.add_argument('--max-lr', type=float, required=True,
                        help='maximum learning rate for OneCycleLR')
    parser.add_argument('--image-size', type=int, required=True,
                        choices=[128, 224, 288, 320],
                        help='image size for training (128, 224, 288, or 320)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='batch size (if not specified, auto-set based on image size)')

    # Other settings
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of data loading workers')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='directory to save checkpoints')
    parser.add_argument('--subset', action='store_true',
                        help='use subset of ImageNet for testing')
    parser.add_argument('--subset-size', type=int, default=10000,
                        help='size of subset to use')

    args = parser.parse_args()

    # Fixed settings
    DROP_PATH_RATE = 0.05
    USE_BLURPOOL = True

    # Auto-set batch size based on image size if not specified
    if args.batch_size is None:
        batch_size_map = {128: 128, 224: 128, 288: 32, 320: 16}
        args.batch_size = batch_size_map[args.image_size]
        print(f"Auto-setting batch size to {args.batch_size} for image size {args.image_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Additional Training Configuration")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr:.2e} → Max LR: {args.max_lr:.2e}")
    print(f"Image size: {args.image_size}×{args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Drop path rate: {DROP_PATH_RATE} (fixed)")
    print(f"Blur pool: {USE_BLURPOOL} (fixed)")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 60)

    # Load checkpoint to get model configuration
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    checkpoint_epoch = checkpoint.get('epoch', 0)
    checkpoint_acc = checkpoint.get('accuracy', 0.0)
    print(f"Checkpoint from epoch {checkpoint_epoch}, accuracy: {checkpoint_acc:.2f}%")

    # Create dataset to get number of classes
    train_annotations_dir = os.path.join(args.data_dir, 'Annotations', 'CLS-LOC', 'train')
    train_dir = os.path.join(args.data_dir, 'Data', 'CLS-LOC', 'train')

    if args.subset:
        from dataloader import SubsetImageNet1K
        train_ds = SubsetImageNet1K(
            root=args.data_dir,
            train=True,
            transform=None,
            subset_size=args.subset_size
        )
    else:
        if os.path.exists(train_annotations_dir) and os.path.exists(train_dir):
            train_ds = ImageNetTrain(train_dir, train_annotations_dir, transform=None)
        else:
            from dataloader import AlbuImageNet1K
            train_ds = AlbuImageNet1K(root=args.data_dir, train=True, transform=None)

    num_classes = getattr(train_ds, 'get_num_classes', lambda: 1000)()
    print(f"Dataset has {num_classes} classes")

    # Create model with drop_path_rate=0.05 and blur_pool=True
    print(f"\nCreating model with drop_path_rate={DROP_PATH_RATE}, use_blurpool={USE_BLURPOOL}")
    model = resnet50(
        num_classes=num_classes,
        drop_path_rate=DROP_PATH_RATE,
        use_blurpool=USE_BLURPOOL
    ).to(device)

    # Load model weights from checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Loaded model weights from checkpoint")

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"✓ Created optimizer with lr={args.lr:.2e}")

    # Create dataloaders
    print(f"\nLoading dataset with image size {args.image_size}×{args.image_size}...")
    train_loader, val_loader, _ = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        subset_size=args.subset_size if args.subset else None,
        val_subset_size=args.subset_size // 10 if args.subset else None,
        use_subset=args.subset
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")

    # Create scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    print(f"✓ Created OneCycleLR scheduler with max_lr={args.max_lr:.2e}")
    print(f"  Total steps: {total_steps}")

    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Training setup
    scaler = GradScaler()
    best_acc = checkpoint_acc  # Start with checkpoint accuracy
    start_epoch = checkpoint_epoch + 1
    final_epoch = start_epoch + args.epochs - 1

    checkpoint_path = os.path.join(args.save_dir, f"additional_training_best_{args.image_size}.pt")

    print("\n" + "=" * 60)
    print(f"Starting training from epoch {start_epoch} to {final_epoch}")
    print(f"Best accuracy to beat: {best_acc:.2f}%")
    print("=" * 60 + "\n")

    # Training loop
    for epoch in range(start_epoch, final_epoch + 1):
        print(f"\nEpoch {epoch}/{final_epoch}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train(
            model, device, train_loader, optimizer, scheduler, epoch, scaler, mixup_alpha=0.2
        )

        # Validate
        val_loss, val_acc = test(model, device, val_loader, epoch)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': val_acc,
                'loss': val_loss,
                'training_config': {
                    'lr': args.lr,
                    'max_lr': args.max_lr,
                    'image_size': args.image_size,
                    'batch_size': args.batch_size,
                    'drop_path_rate': DROP_PATH_RATE,
                    'use_blurpool': USE_BLURPOOL
                }
            }, checkpoint_path)
            print(f"✓ Saved best model at epoch {epoch} with accuracy: {val_acc:.2f}%")

        # Print current best
        print(f"Current best accuracy: {best_acc:.2f}%")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final best accuracy: {best_acc:.2f}%")
    print(f"Best model saved to: {checkpoint_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
