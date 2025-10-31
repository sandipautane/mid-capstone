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
from train_utils import get_learning_rate_config, get_phase_number
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

    # Check if we have ILSVRC structure with Annotations
    train_annotations_dir = os.path.join(data_path, 'Annotations', 'CLS-LOC', 'train')
    val_annotations_dir = os.path.join(data_path, 'Annotations', 'CLS-LOC', 'val')
    train_dir = os.path.join(data_path, 'Data', 'CLS-LOC', 'train')
    val_dir = os.path.join(data_path, 'Data', 'CLS-LOC', 'val')
    use_train_annotations = os.path.exists(train_annotations_dir) and os.path.exists(train_dir)
    use_val_annotations = os.path.exists(val_annotations_dir) and os.path.exists(val_dir)

    if use_subset and subset_size is not None:
        if rank == 0:
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
            if rank == 0:
                print(f"Using XML annotations for training data")
            train_ds = ImageNetTrain(train_dir, train_annotations_dir, transform=train_tfms)
        else:
            train_ds = AlbuImageNet1K(root=data_path, train=True, transform=train_tfms)

        # For validation, prefer ImageNetVal if annotations exist
        if use_val_annotations:
            if rank == 0:
                print(f"Using XML annotations for validation data")
            # Share synset mapping with training dataset
            synset_to_idx = getattr(train_ds, 'synset_to_idx', None)
            val_ds = ImageNetVal(val_dir, val_annotations_dir, transform=val_tfms,
                               synset_to_idx=synset_to_idx)
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

    return train_loader, val_loader, train_sampler, val_sampler, train_ds




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

    # Check if resuming from checkpoint
    resuming = args.resume and os.path.exists(args.resume)

    # Load checkpoint early if resuming to determine the epoch
    resume_epoch = 1
    if resuming:
        if is_main_process:
            print(f"\n{'='*60}")
            print(f"Loading checkpoint to determine resume configuration: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        resume_epoch = checkpoint.get('epoch', 0) + 1

        if is_main_process:
            print(f"Will resume from epoch {resume_epoch}")
            print(f"{'='*60}\n")

    # Determine initial settings based on resume or new training
    if resuming:
        # Continue with progressive training from the resumed epoch
        initial_size = get_image_size_for_epoch(resume_epoch)
        batch_size = get_batch_size(initial_size)
        override_drop_path = args.drop_path_rate
        additional_epochs = args.epochs  # Continue until total epochs

        if is_main_process:
            phase_config = get_learning_rate_config(resume_epoch)
            print(f"Resuming in {phase_config['phase_name']}")
            print(f"  - Image size: {initial_size}×{initial_size}")
            print(f"  - Batch size: {batch_size}")
            print(f"  - Drop path rate: {override_drop_path}")
            print(f"  - Training until epoch: {additional_epochs}")
            print(f"{'='*60}\n")
    else:
        # Progressive resizing for new training
        initial_size = get_image_size_for_epoch(1)
        batch_size = get_batch_size(initial_size)
        override_drop_path = args.drop_path_rate
        additional_epochs = args.epochs

    if args.ddp:
        _, _, _, _, train_ds = get_ddp_dataloaders(
            args.data_dir,
            batch_size=batch_size,
            image_size=initial_size,
            num_workers=args.num_workers,
            subset_size=args.subset_size if args.subset else None,
            val_subset_size=args.subset_size // 10 if args.subset else None,
            use_subset=args.subset,
            rank=rank,
            world_size=world_size
        )
    else:
        from dataloader import get_dataloaders, get_train_transforms, SubsetImageNet1K, ImageNetTrain

        # Create temporary dataset to get number of classes
        if args.subset:
            train_ds = SubsetImageNet1K(
                root=args.data_dir,
                train=True,
                transform=None,
                subset_size=args.subset_size
            )
        else:
            # Check if we have Annotations
            train_annotations_dir = os.path.join(args.data_dir, 'Annotations', 'CLS-LOC', 'train')
            train_dir = os.path.join(args.data_dir, 'Data', 'CLS-LOC', 'train')
            if os.path.exists(train_annotations_dir) and os.path.exists(train_dir):
                train_ds = ImageNetTrain(train_dir, train_annotations_dir, transform=None)
            else:
                from dataloader import AlbuImageNet1K
                train_ds = AlbuImageNet1K(root=args.data_dir, train=True, transform=None)

    # Get number of classes from dataset
    num_classes = getattr(train_ds, 'get_num_classes', lambda: 1000)()
    if is_main_process:
        print(f"Dataset has {num_classes} classes")

    # Create model with correct number of classes
    model = resnet50(num_classes=num_classes, drop_path_rate=override_drop_path, use_blurpool=args.use_blurpool).to(device)

    # Wrap model with DDP if enabled
    if args.ddp:
        model = DDP(model, device_ids=[rank], output_device=rank)

    # Initialize start_epoch and best_acc
    start_epoch = resume_epoch if resuming else 1
    best_acc = 0.0

    # Get initial phase configuration (works for both new and resumed training)
    phase_config = get_learning_rate_config(start_epoch)
    initial_lr = phase_config['start_lr']
    initial_max_lr = phase_config['max_lr']

    if is_main_process:
        print(f"\n{'='*60}")
        if resuming:
            print(f"Resuming {phase_config['phase_name']}")
        else:
            print(f"Starting {phase_config['phase_name']}")
        print(f"  LR: {initial_lr:.2e} → Max LR: {initial_max_lr:.2e}")
        print(f"  Image size: {initial_size}×{initial_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {phase_config['epoch_range'][0]}-{phase_config['epoch_range'][1]}")
        print(f"  {phase_config['notes']}")
        print(f"{'='*60}\n")

    # Optimizer with lower weight decay
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=args.weight_decay)

    # Find total steps - use the number of samples from the training dataset
    num_train_samples = len(train_ds)

    # For progressive training, we'll calculate steps dynamically after creating dataloaders
    # For now, create initial dataloaders to get the actual batch count
    if args.ddp:
        train_loader, val_loader, train_sampler, val_sampler, _ = get_ddp_dataloaders(
            args.data_dir,
            batch_size=batch_size,
            image_size=initial_size,
            num_workers=args.num_workers,
            subset_size=args.subset_size if args.subset else None,
            val_subset_size=args.subset_size // 10 if args.subset else None,
            use_subset=args.subset,
            rank=rank,
            world_size=world_size
        )
    else:
        from dataloader import get_dataloaders
        train_loader, val_loader, _ = get_dataloaders(
            args.data_dir,
            batch_size=batch_size,
            image_size=initial_size,
            num_workers=args.num_workers,
            subset_size=args.subset_size if args.subset else None,
            val_subset_size=args.subset_size // 10 if args.subset else None,
            use_subset=args.subset
        )
        train_sampler = None

    # Calculate total steps for ENTIRE training (single continuous scheduler)
    # This replaces the old per-phase scheduler approach
    if resuming:
        # Calculate remaining steps from current epoch to end
        remaining_epochs = additional_epochs - start_epoch + 1
        # Use approximate steps per epoch (will vary slightly with batch size changes)
        total_steps = len(train_loader) * remaining_epochs
    else:
        # Calculate total steps for all epochs using actual batch sizes per phase
        total_steps = 0
        for epoch_num in range(1, additional_epochs + 1):
            size = get_image_size_for_epoch(epoch_num)
            bs = get_batch_size(size)
            steps_for_epoch = num_train_samples // bs
            total_steps += steps_for_epoch

    if is_main_process:
        print(f"\n{'='*60}")
        print(f"Single Continuous Scheduler Configuration")
        print(f"{'='*60}")
        print(f"Total training steps (all epochs): {total_steps}")
        print(f"Initial steps per epoch: {len(train_loader)}")
        print(f"Max LR (peak): {initial_max_lr:.2e}")
        print(f"Strategy: OneCycleLR with continuous decay across all phases")
        print(f"Note: LR naturally decreases as training progresses")
        print(f"{'='*60}\n")

    # Single scheduler for entire training - no recreation at phase transitions
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=initial_max_lr,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # Set checkpoint path
    checkpoint_path = os.path.join(args.save_dir, "best_resnet50_imagenet_1k.pt")

    # Load checkpoint if resuming (checkpoint was already loaded earlier)
    if resuming:
        if is_main_process:
            print(f"Loading model and optimizer states from checkpoint...")

        # Move checkpoint to correct device
        checkpoint = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in checkpoint.items()}

        # Load model state
        if args.ddp:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state for seamless continuation with single continuous scheduler
        if 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                current_lr = optimizer.param_groups[0]['lr']
                if is_main_process:
                    print(f"✓ Loaded scheduler state from checkpoint")
                    print(f"  Current LR: {current_lr:.2e} (continuous OneCycleLR schedule)")
            except Exception as e:
                if is_main_process:
                    print(f"⚠ Warning: Could not load scheduler state: {e}")
                    print("  Continuing with fresh scheduler (may cause LR instability)")
                    print("  Tip: This is normal if switching from old multi-scheduler to single scheduler")

        # Load training state
        best_acc = checkpoint.get('accuracy', 0.0)

        if is_main_process:
            print(f"Resumed from epoch {checkpoint.get('epoch', 0)}")
            print(f"Best accuracy so far: {best_acc:.2f}%")
            print(f"Continuing from epoch {start_epoch}")
            print()

    # Create save directory
    if is_main_process and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Training setup
    scaler = torch.cuda.amp.GradScaler()
    current_size = None
    current_phase = None
    batch_size = args.batch_size

    # Calculate final epoch - train until the specified total epochs
    EPOCHS = additional_epochs

    if is_main_process:
        print(f"Training from epoch {start_epoch} to {EPOCHS}")
        print(f"Remaining epochs: {EPOCHS - start_epoch + 1}\n")

    for epoch in range(start_epoch, EPOCHS + 1):
        # Check for phase transition (only for drop_path disabling, not for optimizer/scheduler)
        new_phase = get_phase_number(epoch)
        if new_phase != current_phase and current_phase is not None:
            # Phase transition detected
            phase_config = get_learning_rate_config(epoch)

            if is_main_process:
                print(f"\n{'='*60}")
                print(f"PHASE TRANSITION: Starting {phase_config['phase_name']}")
                print(f"  Epochs: {phase_config['epoch_range'][0]}-{phase_config['epoch_range'][1]}")
                print(f"  {phase_config['notes']}")
                print(f"  Current LR: {optimizer.param_groups[0]['lr']:.2e} (continuous schedule)")

                # Note about regularization changes
                if new_phase >= 3:
                    print(f"  Mixup/CutMix: DISABLED (fine-tuning phase)")
                    print(f"  Drop Path: DISABLED (fine-tuning phase)")

                print(f"{'='*60}\n")

            # Disable drop_path for phases 3 and 4 by setting it to 0
            if new_phase >= 3:
                model_to_modify = model.module if args.ddp else model
                if hasattr(model_to_modify, 'set_drop_path_rate'):
                    model_to_modify.set_drop_path_rate(0.0)
                    if is_main_process:
                        print(f"✓ Disabled drop_path for fine-tuning (Phase {new_phase})")
                else:
                    # Manually disable drop_path in all DropPath layers
                    for name, module in model_to_modify.named_modules():
                        if hasattr(module, 'drop_prob'):
                            module.drop_prob = 0.0
                    if is_main_process:
                        print(f"✓ Disabled drop_path in all DropPath layers for Phase {new_phase}")

            # NOTE: We NO LONGER recreate optimizer or scheduler at phase transitions
            # The single continuous scheduler handles LR decay naturally across all phases

        current_phase = new_phase

        # Check if we need to change image size (progressive resizing for all training)
        new_size = get_image_size_for_epoch(epoch)

        if new_size != current_size:
            if is_main_process:
                print(f"\n{'='*50}")
                print(f"Switching to image size: {new_size}x{new_size}")
                print(f"{'='*50}\n")

            # Use progressive batch size
            batch_size = get_batch_size(new_size)

            # Recreate dataloaders with new size
            if args.ddp:
                train_loader, val_loader, train_sampler, val_sampler, train_ds = get_ddp_dataloaders(
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
                from dataloader import get_dataloaders
                train_loader, val_loader, _ = get_dataloaders(
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

            # NOTE: With single continuous scheduler, we no longer recreate scheduler
            # when batch size/dataloader changes. The scheduler continues naturally.

        # Set epoch for DistributedSampler
        if args.ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Disable mixup/cutmix for phases 3 and 4 (fine-tuning phases)
        current_phase_num = get_phase_number(epoch)
        mixup_alpha = 0.0 if current_phase_num >= 3 else 0.2

        tr_loss, tr_acc = train(model, device, train_loader, optimizer, scheduler, epoch, scaler, mixup_alpha=mixup_alpha)
        val_loss, val_acc = test(model, device, val_loader, epoch)

        # Save checkpoint (only on main process)
        if is_main_process:
            model_to_save = model.module if args.ddp else model

            # Save best model (after epoch 4)
            if epoch > 4 and val_acc > best_acc:
                best_acc = val_acc
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
    parser = argparse.ArgumentParser(description='Train ResNet50 on ImageNet 1K with Progressive Resizing')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train (default: 40 for all 4 phases)')
    parser.add_argument('--batch-size', type=int, default=128, help='initial batch size (will be adjusted per phase)')
    parser.add_argument('--lr', type=float, default=1e-3, help='base learning rate (used for resumed training, auto-adjusted for progressive training)')
    parser.add_argument('--max-lr', type=float, default=5e-3, help='maximum learning rate for OneCycleLR (used for resumed training, auto-adjusted for progressive training)')
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
    parser.add_argument('--resume-drop-path-rate', type=float, default=0.0, help='drop path rate when resuming (default: 0.0 for fine-tuning)')
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
