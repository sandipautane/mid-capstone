import argparse
import os
import ssl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import torchvision.transforms as transforms
from models.model import resnet50
from dataloader import get_dataloaders
import torchsummary as summary
from train_utils import calculate_total_steps, get_image_size_for_epoch, get_batch_size
from train_utils import EarlyStopping, train, test
from tqdm import tqdm
import numpy as np
import copy
from torch.cuda.amp import autocast, GradScaler

# Fix SSL certificate issue for downloading CIFAR-100
ssl._create_default_https_context = ssl._create_unverified_context


def setup_ddp(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def get_ddp_dataloaders(data_path, batch_size=128, image_size=64, num_workers=2,
                        subset_size=None, val_subset_size=None, use_subset=False, rank=0, world_size=1):
    """
    Create train and validation dataloaders with DistributedSampler for DDP
    Supports both ILSVRC structure and simple train/val structure
    """
    from dataloader import get_train_transforms, get_val_transforms, SubsetImageNet1K, AlbuImageNet1K, ImageNetVal, ImageNetTrain

    train_tfms = get_train_transforms(image_size)
    val_tfms = get_val_transforms(image_size)

    # Check if we have ILSVRC structure with train_cls.txt and val.txt
    train_txt_path = os.path.join(data_path, 'ImageSets', 'CLS-LOC', 'train_cls.txt')
    val_txt_path = os.path.join(data_path, 'ImageSets', 'CLS-LOC', 'val.txt')
    train_dir = os.path.join(data_path, 'Data', 'CLS-LOC', 'train')
    val_dir = os.path.join(data_path, 'Data', 'CLS-LOC', 'val')
    use_train_txt = os.path.exists(train_txt_path) and os.path.exists(train_dir)
    use_val_txt = os.path.exists(val_txt_path) and os.path.exists(val_dir)

    if use_subset and subset_size is not None:
        if rank == 0:
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
        # For training, prefer ImageNetTrain if train_cls.txt exists
        if use_train_txt:
            if rank == 0:
                print(f"Using train_cls.txt for training data")
            train_ds = ImageNetTrain(train_dir, train_txt_path, transform=train_tfms)
        else:
            train_ds = AlbuImageNet1K(root=data_path, train=True, transform=train_tfms)

        # For validation, prefer ImageNetVal if val.txt exists
        if use_val_txt:
            if rank == 0:
                print(f"Using val.txt for validation data")
            val_ds = ImageNetVal(val_dir, val_txt_path, transform=val_tfms)
        else:
            val_ds = AlbuImageNet1K(root=data_path, train=False, transform=val_tfms)

    # Create DistributedSampler
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    # Note: When using DistributedSampler, shuffle must be False in DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )

    return train_loader, val_loader, train_sampler, val_sampler




def train_worker(rank, world_size, args):
    """Training function for each DDP process"""
    # Setup DDP
    if args.ddp:
        setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}")
        is_main_process = (rank == 0)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True

    if is_main_process:
        print("Device:", device)

    # Determine number of classes
    # If using a subset, we need to check how many classes are actually in the subset
    if args.subset:
        from dataloader import SubsetImageNet1K, get_train_transforms
        # Create a temporary dataset to get the number of classes
        temp_ds = SubsetImageNet1K(
            root=args.data_dir,
            train=True,
            transform=None,
            subset_size=args.subset_size
        )
        num_classes = temp_ds.get_num_classes()
        if is_main_process:
            print(f"Using subset with {num_classes} classes (from {args.subset_size} samples)")
    else:
        num_classes = 1000  # Full ImageNet

    # Create model with correct number of classes
    model = resnet50(num_classes=num_classes, drop_path_rate=args.drop_path_rate, use_blurpool=args.use_blurpool).to(device)

    # Wrap model with DDP if enabled
    if args.ddp:
        model = DDP(model, device_ids=[rank], output_device=rank)

    # Optimizer with lower weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Find total steps
    total_steps = calculate_total_steps(args.epochs)

    # Scheduler with correct total steps
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # Track best accuracy
    best_acc = 0.0
    checkpoint_path = os.path.join(args.save_dir, "best_resnet50_imagenet_1k.pt")

    # Create save directory
    if is_main_process and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Training setup
    scaler = torch.cuda.amp.GradScaler()
    current_size = None
    batch_size = args.batch_size
    EPOCHS = args.epochs

    for epoch in range(1, EPOCHS + 1):
        # Check if we need to change image size
        new_size = get_image_size_for_epoch(epoch)
        if new_size != current_size:
            if is_main_process:
                print(f"\n{'='*50}")
                print(f"Switching to image size: {new_size}x{new_size}")
                print(f"{'='*50}\n")

            batch_size = get_batch_size(new_size)

            # Recreate dataloaders with new size
            if args.ddp:
                train_loader, val_loader, train_sampler, val_sampler = get_ddp_dataloaders(
                    args.data_dir,
                    batch_size=batch_size,
                    image_size=new_size,
                    num_workers=args.num_workers,
                    subset_size=args.subset_size if args.subset else None,
                    val_subset_size=args.subset_size // 10 if args.subset else None,
                    use_subset=args.subset,
                    rank=rank,
                    world_size=world_size
                )
            else:
                train_loader, val_loader = get_dataloaders(
                    args.data_dir,
                    batch_size=batch_size,
                    image_size=new_size,
                    num_workers=args.num_workers,
                    subset_size=args.subset_size if args.subset else None,
                    val_subset_size=args.subset_size // 10 if args.subset else None,
                    use_subset=args.subset
                )
                train_sampler = None

            current_size = new_size

        # Set epoch for DistributedSampler
        if args.ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        tr_loss, tr_acc = train(model, device, train_loader, optimizer, scheduler, epoch, scaler, mixup_alpha=0.2)
        val_loss, val_acc = test(model, device, val_loader, epoch)

        # Save if best accuracy (only on main process)
        if is_main_process and epoch > 60:
            if val_acc > best_acc:
                best_acc = val_acc
                # Save the underlying model if using DDP
                model_to_save = model.module if args.ddp else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'accuracy': val_acc,
                    'loss': val_loss
                }, checkpoint_path)
                print(f"✓ Saved best model at epoch {epoch} with accuracy: {val_acc:.2f}%")

    # Cleanup DDP
    if args.ddp:
        cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description='Train ResNet50 on ImageNet 1K')
    parser.add_argument('--epochs', type=int, default=90, help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='base learning rate')
    parser.add_argument('--max-lr', type=float, default=1e-2, help='maximum learning rate for OneCycleLR')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='directory to save checkpoints')
    parser.add_argument('--data-dir', type=str, default='./data/imagenet', help='path to ImageNet dataset')
    parser.add_argument('--subset', action='store_true', help='use subset of ImageNet for faster training')
    parser.add_argument('--subset-size', type=int, default=10000, help='size of subset to use')
    parser.add_argument('--num-workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--plot', action='store_true', help='plot training results')
    parser.add_argument('--drop-path-rate', type=float, default=0.2, help='drop path rate')
    parser.add_argument('--use-blurpool', action='store_true', help='use blurpool')
    # DDP arguments
    parser.add_argument('--ddp', action='store_true', help='use DistributedDataParallel for multi-GPU training')
    parser.add_argument('--world-size', type=int, default=torch.cuda.device_count(), help='number of GPUs to use for DDP')
    args = parser.parse_args()

    # Launch DDP training or single-GPU training
    if args.ddp:
        world_size = args.world_size
        if world_size < 1:
            print("Error: No GPUs available for DDP training")
            return

        print(f"Starting DDP training with {world_size} GPUs")
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        # Single GPU or CPU training
        train_worker(0, 1, args)


if __name__ == '__main__':
    main()