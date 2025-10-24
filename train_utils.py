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

        # Apply mixup
        if np.random.random() > 0.5:
            data, targets_a, targets_b, lam = mixup_data(data, target, mixup_alpha, device)
        else:
            data, targets_a, targets_b, lam = cutmix_data(data, target, 1.0, device)

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




def calculate_total_steps(epochs, num_samples):
    """Calculate total steps accounting for progressive resizing and batch size changes

    Args:
        epochs: Total number of training epochs
        num_samples: Number of training samples in the dataset
    """
    total_steps = 0

    for epoch in range(1, epochs + 1):
        size = get_image_size_for_epoch(epoch)
        batch_size = get_batch_size(size)
        steps = num_samples // batch_size
        total_steps += steps

    return total_steps

# Use it
#total_steps = calculate_total_steps(100)  # For 100 epochs


def get_image_size_for_epoch(epoch):
    """Return image size based on epoch"""
    if epoch <= 30:
        return 64
    elif epoch <= 50:
        return 128
    else:
        return 224

def get_batch_size(image_size):
    """Return batch size based on image size"""
    if image_size == 64:
        return 256
    elif image_size == 128:
        return 128
    else:
        return 64
