# Install if needed
#!pip install torch-lr-finder

import torch.optim as optim
from torch_lr_finder import LRFinder
import torch.nn as nn
import torch
from models.model import resnet50
from dataloader import get_dataloaders  , get_train_transforms, get_val_transforms

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Create fresh model and optimizer
    model = resnet50(num_classes=1000, drop_path_rate=0.2, use_blurpool=False).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-7, weight_decay=5e-4)  # Lower weight decay
    criterion = nn.CrossEntropyLoss()
    train_transforms = get_train_transforms(image_size=64)
    val_transforms = get_val_transforms(image_size=64)
    train_loader, val_loader = get_dataloaders(train_transforms, val_transforms, batch_size=128, image_size=64, num_workers=2)  
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=0.005, num_iter=200)
    lr_finder.plot()
    lr_finder.reset()

if __name__ == "__main__":
    main()