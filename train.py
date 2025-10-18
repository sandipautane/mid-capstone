import argparse
import os
import ssl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

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






def main():
    parser = argparse.ArgumentParser(description='Train ResNet50 on ImageNet 1K')
    parser.add_argument('--epochs', type=int, default=90, help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

# Create model with less regularization
    model = resnet50(num_classes=1000, drop_path_rate=args.drop_path_rate, use_blurpool=args.use_blurpool).to(device)

# Optimizer with lower weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# find total steps
    total_steps = calculate_total_steps(args.epochs)  # For 100 epochs

# Scheduler with correct total steps
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )

# Track best accuracy
    best_acc = 0.0
    checkpoint_path = args.save_dir + "/best_resnet50_imagenet_1k.pt"

# In your training loop:
    scaler = torch.cuda.amp.GradScaler()
    current_size = None

    batch_size = args.batch_size
    EPOCHS = args.epochs

    for epoch in range(1, EPOCHS + 1):
    # Check if we need to change image size
        new_size = get_image_size_for_epoch(epoch)
        if new_size != current_size:
            print(f"\n{'='*50}")
            print(f"Switching to image size: {new_size}x{new_size}")
            print(f"{'='*50}\n")

            batch_size = get_batch_size(new_size)
 # Recreate dataloaders with new size
        train_loader, val_loader = get_dataloaders(args.data_dir, batch_size=batch_size,
                                                    image_size=new_size, num_workers=4)
        current_size = new_size

    tr_loss, tr_acc = train(model, device, train_loader, optimizer, scheduler, epoch, scaler, mixup_alpha=0.2)
    val_loss, val_acc = test(model, device, val_loader, epoch)
     # Save if best accuracy
    if epoch > 60:# start saving only after 20th epoch
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'loss': avg_loss
            }, checkpoint_path)
            print(f"✓ Saved best model at epoch {epoch} with accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    main()