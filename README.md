# ResNet50 for ImageNet 1K

A PyTorch implementation of ResNet50 optimized for ImageNet 1K classification with a complete training pipeline.

## Features

- **ResNet50 Architecture**: Bottleneck-based ResNet with [3, 4, 6, 3] layer configuration
- **ImageNet Optimized**: 7x7 stem convolution with maxpool for 224x224 input images
- **Complete Training Pipeline**: Command-line training script with data loading, augmentation, and checkpointing
- **Data Augmentation**: Random resized cropping, horizontal flipping, and color jittering
- **Learning Rate Scheduling**: Multi-step learning rate decay at epochs 30 and 60
- **Checkpoint Management**: Automatic saving of best models and regular checkpoints
- **Flexible Dataset Support**: Full ImageNet or subset for faster development

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training ResNet50 on ImageNet 1K

```bash
# Training on ImageNet subset (for development/testing)
python train.py --subset --subset-size 10000 --epochs 10 --batch-size 64

# Training on full ImageNet dataset
python train.py --epochs 90 --batch-size 256 --data-dir /path/to/imagenet

# Custom training parameters
python train.py --epochs 90 --batch-size 128 --lr 0.05 --plot --subset

# Resume from checkpoint
python train.py --resume ./checkpoints/best_model.pth
```

### Using the Model in Code

```python
from models.resnet import ResNet50
from dataloader import get_imagenet_loaders, get_imagenet_subset_loaders

# Create model for ImageNet (1000 classes)
model = ResNet50(num_classes=1000, input_size=224)

# Load ImageNet data
train_loader, val_loader = get_imagenet_loaders(
    data_dir='./data/imagenet',
    batch_size=256,
    num_workers=8
)

# Forward pass
import torch
x = torch.randn(1, 3, 224, 224)  # Batch of 1, 3 channels, 224x224 images
output = model(x)  # Shape: [1, 1000]
```

## Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 90 | Number of training epochs |
| `--batch-size` | 256 | Batch size for training |
| `--lr` | 0.1 | Learning rate |
| `--momentum` | 0.9 | SGD momentum |
| `--weight-decay` | 1e-4 | Weight decay for regularization |
| `--resume` | "" | Path to checkpoint to resume from |
| `--save-dir` | "./checkpoints" | Directory to save model checkpoints |
| `--data-dir` | "./data/imagenet" | Path to ImageNet dataset |
| `--subset` | False | Use subset of ImageNet for faster training |
| `--subset-size` | 10000 | Size of subset to use |
| `--num-workers` | 8 | Number of data loading workers |
| `--plot` | False | Generate training plots |

## Expected Performance

With default training settings on ImageNet, you can expect:
- **Top-1 Accuracy**: ~75-78% (validation)
- **Top-5 Accuracy**: ~92-95% (validation)
- **Training Time**: ~12-24 hours on a modern GPU (full ImageNet)

For ImageNet subset (10,000 samples):
- **Training Time**: ~30-60 minutes on a modern GPU
- **Good for**: Development, testing, and experimentation

## File Structure

```
├── models/
│   └── resnet.py          # ResNet implementation (ImageNet optimized)
├── dataloader.py          # ImageNet dataloader with transforms and augmentation
├── train.py               # Training script for ImageNet
├── example_usage.py       # Example usage of dataloader and model
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Dependencies

- PyTorch >= 2.1
- Torchvision >= 0.16
- Matplotlib >= 3.5
- NumPy >= 1.21
- Pillow >= 8.3

## Checkpoints

The training script automatically saves:
- **Best Model**: `checkpoints/best_model.pth` (highest validation accuracy)
- **Regular Checkpoints**: `checkpoints/checkpoint_epoch_X.pth` (every 10 epochs)
- **Training Plots**: `checkpoints/training_plots.png` (if `--plot` flag used)

## Model Architecture

The ResNet50 implementation uses:
- **Stem**: 7x7 convolution (stride=2, padding=3) + 3x3 maxpool for ImageNet compatibility
- **Bottleneck Blocks**: [3, 4, 6, 3] configuration across 4 stages
- **Global Average Pooling**: Adaptive pooling to 1x1 before classification
- **Classification Head**: Linear layer with 1000 outputs for ImageNet

## Dataset Setup

### ImageNet Dataset Structure

Place your ImageNet dataset in the following structure:
```
data/
└── imagenet/
    ├── train/
    │   ├── n01440764/
    │   ├── n01443537/
    │   └── ...
    └── val/
        ├── n01440764/
        ├── n01443537/
        └── ...
```

### Using ImageNet Subset

For faster development and testing, you can use a subset of ImageNet:
```bash
python train.py --subset --subset-size 10000
```

This will randomly sample 10,000 images from the training set and 1,000 from validation.


