import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.utils.data.distributed as dist
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim.optimizer as optimizer
from data_utils import mixup_data, mixup_criterion, cutmix_data
# keep your history lists (now store epoch-level stats)
train_losses, test_losses = [], []
train_acc,    test_acc    = [], []

class EarlyStopping:
    def __init__(self, patience=6, min_delta=1e-4, restore_best_weights=True, checkpoint_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.checkpoint_path = checkpoint_path
        self.best = float('inf')
        self.wait = 0
        self.best_state = None

    def step(self, val_loss, model):
        improved = (self.best - val_loss) > self.min_delta
        if improved:
            self.best = val_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_state = copy.deepcopy(model.state_dict())
            if self.checkpoint_path:
                torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.wait += 1
        return self.wait >= self.patience

    def load_best(self, model):
        if self.restore_best_weights and self.best_state is not None:
            model.load_state_dict(self.best_state)
        elif self.checkpoint_path:
            model.load_state_dict(torch.load(self.checkpoint_path, map_location="cpu"))

# mix precision training loop with gradient clipping

from torch.cuda.amp import autocast, GradScaler

def train(model, device, train_loader, optimizer, scheduler, epoch, scaler, mixup_alpha=0.2):
    model.train()
    pbar = tqdm(train_loader, desc=f"Train E{epoch:02d}")
    correct, total = 0, 0
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        # Apply mixup or cutmix (skip if mixup_alpha is 0)
        if mixup_alpha > 0:
            if np.random.random() > 0.5:
                data, targets_a, targets_b, lam = mixup_data(data, target, mixup_alpha, device)
            else:
                data, targets_a, targets_b, lam = cutmix_data(data, target, 1.0, device)
        else:
            # No augmentation - use original data
            targets_a, targets_b, lam = target, target, 1.0

        optimizer.zero_grad()

        with autocast():
            y_pred = model(data)
            loss = mixup_criterion(
                lambda pred, y: F.cross_entropy(pred, y, label_smoothing=0.1),
                y_pred, targets_a, targets_b, lam
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Update EMA
        # ema.update()

        # stats
        running_loss += loss.item() * data.size(0)
        pred = y_pred.argmax(dim=1)
        correct += (lam * pred.eq(targets_a).sum().float()
                    + (1 - lam) * pred.eq(targets_b).sum().float()).item()
        total += data.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc  = 100.0 * correct / total
    train_losses.append(epoch_loss)
    train_acc.append(epoch_acc)
    return epoch_loss, epoch_acc


def test(model, device, test_loader, epoch=None):
    model.eval()
    test_loss_sum, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='sum', label_smoothing=0.1)  # sum over batch
            test_loss_sum += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)

    avg_loss = test_loss_sum / total
    acc = 100.0 * correct / total
    test_losses.append(avg_loss)
    test_acc.append(acc)

    if epoch is not None:
        print(f"\nVal E{epoch:02d}: loss={avg_loss:.4f}, acc={correct}/{total} ({acc:.2f}%)\n")
    else:
        print(f"\nVal: loss={avg_loss:.4f}, acc={correct}/{total} ({acc:.2f}%)\n")

    return avg_loss, acc




def calculate_total_steps(epochs, num_samples, fixed_image_size=None, fixed_batch_size=None):
    """Calculate total steps accounting for progressive resizing and batch size changes

    Args:
        epochs: Total number of training epochs
        num_samples: Number of training samples in the dataset
        fixed_image_size: If provided, use this image size for all epochs (for resuming)
        fixed_batch_size: If provided, use this batch size for all epochs (for resuming)
    """
    total_steps = 0

    if fixed_image_size is not None and fixed_batch_size is not None:
        # Fixed settings (for resumed training)
        steps_per_epoch = num_samples // fixed_batch_size
        total_steps = epochs * steps_per_epoch
    else:
        # Progressive resizing (for new training)
        for epoch in range(1, epochs + 1):
            size = get_image_size_for_epoch(epoch)
            batch_size = get_batch_size(size)
            steps = num_samples // batch_size
            total_steps += steps

    return total_steps

# Use it
#total_steps = calculate_total_steps(100)  # For 100 epochs


def get_image_size_for_epoch(epoch):
    """Return image size based on epoch

    Progressive resizing schedule:
    - Phase 1 (Warm-up): epochs 1-8 → 128×128
    - Phase 2 (Main): epochs 9-28 → 224×224
    - Phase 3 (Refine): epochs 29-36 → 288×288
    - Phase 4 (Optional FT): epochs 37-40 → 320×320
    """
    if epoch <= 8:
        return 128
    elif epoch <= 28:
        return 224
    elif epoch <= 36:
        return 288
    else:
        return 320

def get_batch_size(image_size):
    """Return batch size based on image size

    Batch size schedule for progressive resizing:
    - 128×128 → batch size 128
    - 224×224 → batch size 128
    - 288×288 → batch size 32
    - 320×320 → batch size 16
    """
    if image_size == 128:
        return 128
    elif image_size == 224:
        return 128
    elif image_size == 288:
        return 32
    elif image_size == 320:
        return 16
    else:
        # Default fallback
        return 128

def get_learning_rate_config(epoch):
    """Return learning rate configuration for the current phase

    Returns:
        dict: Contains 'start_lr', 'max_lr', 'phase_name', and 'recommended_epochs'

    Learning rate schedule:
    - Phase 1 (Warm-up, epochs 1-8): start=1e-3, max=4e-3 to 5e-3
    - Phase 2 (Main, epochs 9-28): start=1e-3, max=4e-3 to 6e-3
    - Phase 3 (Refine, epochs 29-36): start=2e-4, max=8e-4 to 1e-3
    - Phase 4 (Optional FT, epochs 37-40): start=1e-4, max=4e-4 to 5e-4
    """
    if epoch <= 8:
        return {
            'phase_name': 'Phase 1 - Warm-up',
            'phase_num': 1,
            'start_lr': 1.0e-3,
            'max_lr': 4.5e-3,  # midpoint of 4e-3 to 5e-3
            'recommended_epochs': 8,
            'epoch_range': (1, 8),
            'notes': 'Fast coarse training to learn global features'
        }
    elif epoch <= 28:
        return {
            'phase_name': 'Phase 2 - Main',
            'phase_num': 2,
            'start_lr': 1.0e-3,
            'max_lr': 4.0e-3,  # Reduced from 5e-3 to 4e-3 for better stability
            'recommended_epochs': 20,
            'epoch_range': (9, 28),
            'notes': 'Core training stage, most important phase'
        }
    elif epoch <= 36:
        return {
            'phase_name': 'Phase 3 - Refine',
            'phase_num': 3,
            'start_lr': 2.0e-4,
            'max_lr': 9.0e-4,  # midpoint of 8e-4 to 1e-3
            'recommended_epochs': 8,
            'epoch_range': (29, 36),
            'notes': 'Adds fine details, fewer epochs suffice'
        }
    else:
        return {
            'phase_name': 'Phase 4 - Optional FT',
            'phase_num': 4,
            'start_lr': 1.0e-4,
            'max_lr': 4.5e-4,  # midpoint of 4e-4 to 5e-4
            'recommended_epochs': 4,
            'epoch_range': (37, 40),
            'notes': 'Short final fine-tuning for extra sharpness'
        }

def get_phase_number(epoch):
    """Return the phase number (1-4) for a given epoch"""
    return get_learning_rate_config(epoch)['phase_num']
