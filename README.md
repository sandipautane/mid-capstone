# Advanced ResNet50 for ImageNet 1K

A state-of-the-art PyTorch implementation of ResNet50 optimized for ImageNet 1K classification with advanced training techniques and comprehensive experimental framework.

## Features

- **Advanced ResNet50 Architecture**: Bottleneck-based ResNet with [3, 4, 6, 3] layer configuration
- **Progressive Image Resizing**: Dynamic image size progression (64→128→224) for efficient training
- **Stochastic Depth (Drop Path)**: Linear drop path rate scheduling for better generalization
- **BlurPool Support**: Optional antialiased pooling for improved visual quality
- **Mixed Precision Training**: Automatic mixed precision with gradient scaling and clipping
- **Advanced Data Augmentation**: Albumentations-based transforms with MixUp and CutMix
- **OneCycleLR Scheduling**: Cosine annealing learning rate scheduler
- **EMA Support**: Exponential Moving Average for model weights
- **Comprehensive Ablation Studies**: Extensive experimental framework on Tiny ImageNet
- **Flexible Dataset Support**: Full ImageNet or subset for faster development

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training ResNet50 on ImageNet 1K

```bash
# Training with progressive resizing and advanced techniques
python train.py --epochs 90 --batch-size 256 --lr 3e-4 --drop-path-rate 0.2

# Training with BlurPool for improved visual quality
python train.py --epochs 90 --use-blurpool --drop-path-rate 0.2

# Training on ImageNet subset (for development/testing)
python train.py --subset --subset-size 10000 --epochs 30 --batch-size 128

# Resume from checkpoint
python train.py --resume ./checkpoints/best_resnet50_imagenet_1k.pt

# Learning rate finding
python lr_finder.py
```

### Using the Model in Code

```python
from models.model import resnet50
from dataloader import get_dataloaders

# Create model for ImageNet (1000 classes) with advanced features
model = resnet50(num_classes=1000, drop_path_rate=0.2, use_blurpool=False)

# Load ImageNet data with progressive resizing
train_loader, val_loader = get_dataloaders(
    data_path='./data/imagenet',
    batch_size=256,
    image_size=224,  # Will be adjusted during training
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
| `--batch-size` | 256 | Initial batch size (auto-adjusted with progressive resizing) |
| `--lr` | 3e-4 | Learning rate for OneCycleLR scheduler |
| `--momentum` | 0.9 | SGD momentum (not used with AdamW) |
| `--weight-decay` | 1e-4 | Weight decay for regularization |
| `--resume` | "" | Path to checkpoint to resume from |
| `--save-dir` | "./checkpoints" | Directory to save model checkpoints |
| `--data-dir` | "./data/imagenet" | Path to ImageNet dataset |
| `--subset` | False | Use subset of ImageNet for faster training |
| `--subset-size` | 10000 | Size of subset to use |
| `--num-workers` | 8 | Number of data loading workers |
| `--plot` | False | Generate training plots |
| `--drop-path-rate` | 0.2 | Stochastic depth drop path rate |
| `--use-blurpool` | False | Use antialiased blurpool instead of maxpool |

## Expected Performance

With advanced training techniques on ImageNet, you can expect:
- **Top-1 Accuracy**: ~78-82% (validation) with progressive resizing + stochastic depth
- **Top-5 Accuracy**: ~94-96% (validation) 
- **Training Time**: ~8-16 hours on a modern GPU (full ImageNet) with mixed precision
- **Generalization**: Improved train-test gap with stochastic depth and advanced augmentation

**Key Performance Improvements:**
- Progressive resizing reduces training time while maintaining accuracy
- Stochastic depth improves generalization and reduces overfitting
- MixUp + CutMix provide robust data augmentation
- Mixed precision training accelerates training without accuracy loss

For ImageNet subset (10,000 samples):
- **Training Time**: ~20-40 minutes on a modern GPU
- **Good for**: Development, testing, and experimentation

## File Structure

```
├── models/
│   └── model.py              # Advanced ResNet50 implementation with stochastic depth and blurpool
├── dataloader.py             # ImageNet dataloader with Albumentations transforms
├── train.py                  # Advanced training script with progressive resizing and mixed precision
├── train_utils.py            # Training utilities (early stopping, mixup/cutmix, progressive resizing)
├── data_utils.py             # Data augmentation utilities (mixup, cutmix)
├── ema.py                    # Exponential Moving Average implementation
├── lr_finder.py              # Learning rate finder utility
├── ablation_runs/            # Experimental notebooks and results
│   ├── ablation_run_tiny_imagenet_01.ipynb
│   ├── ablation_run_tiny_imagenet_02.ipynb
│   └── README.md
├── ablation_run_tiny_imagenet_03.ipynb  # Latest ablation study
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## Dependencies

- PyTorch >= 2.1
- Torchvision >= 0.16
- Matplotlib >= 3.5
- NumPy >= 1.21
- Pillow >= 8.3
- SciPy
- antialiased_cnns (for BlurPool support)
- torch-lr-finder (for learning rate finding)
- albumentations (for advanced data augmentation)
- torchsummary (for model summary)

## Checkpoints

The training script automatically saves:
- **Best Model**: `checkpoints/best_resnet50_imagenet_1k.pt` (highest validation accuracy after epoch 60)
- **Training Progress**: Automatic checkpointing with mixed precision support
- **Model State**: Includes optimizer state, scheduler state, and training metrics

## Model Architecture

The advanced ResNet50 implementation features:

- **Stem**: 7x7 convolution (stride=2, padding=3) + optional BlurPool or MaxPool for ImageNet compatibility
- **Bottleneck Blocks**: [3, 4, 6, 3] configuration across 4 stages with stochastic depth
- **Stochastic Depth**: Linear drop path rate scheduling (0 → drop_path_rate) across all blocks
- **BlurPool Support**: Optional antialiased pooling for improved visual quality
- **Global Average Pooling**: Adaptive pooling to 1x1 before classification
- **Classification Head**: Conv2d layer (1x1) with 1000 outputs for ImageNet
- **Mixed Precision**: Automatic mixed precision training with gradient scaling

## Dataset Setup

### Downloading ImageNet 1K

We provide multiple methods to download ImageNet 1K dataset:

#### Method 1: Simple Download (Recommended for testing)
```bash
# Try automatic download using torchvision


```

#### Method 2: Manual Download (Full dataset)
```bash
# Get detailed instructions for manual download
python download_imagenet.py --method manual
```

This will show you:
1. How to register at http://www.image-net.org/
2. Where to download the tar files (138GB train + 6.3GB val)
3. How to organize the files

#### Method 3: Automatic Download (with URLs)
```bash
# If you have download URLs from ImageNet registration
python download_imagenet.py --method auto \
  --train-url "http://your-train-url.tar" \
  --val-url "http://your-val-url.tar"
```

### ImageNet Dataset Structure

After download, your dataset should have this structure:
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

## Ablation Studies

This project includes comprehensive ablation studies conducted on Tiny ImageNet to validate techniques before applying them to ImageNet 1K:

### Ablation Run 1 (`ablation_run_tiny_imagenet_01.ipynb`)
**Techniques Tested:**
- ResNet50 basic architecture
- ResNet50 pre-activation architecture (without ReLU in skip connection)
- ResNet50 with Conv2d classification head instead of FC layer
- Learning rate finding
- Image augmentations
- MixUp data augmentation
- Mixed precision training

**Key Finding:** ResNet50 basic architecture with Conv2d classification head achieved the best accuracy.

### Ablation Run 2 (`ablation_run_tiny_imagenet_02.ipynb`)
**Techniques Tested:**
- Progressive Batch Resizing (64→128→224)
- CutMix combined with MixUp
- Stochastic Depth (Drop Path)

**Key Findings:**
- Progressive Batch Resizing + CutMix/MixUp improved accuracy by ~10%
- Stochastic Depth significantly reduced train-test accuracy gap
- Combined techniques showed substantial performance gains

### Ablation Run 3 (`ablation_run_tiny_imagenet_03.ipynb`)
**Latest comprehensive study** incorporating all validated techniques from previous runs.

### Experimental Framework
The ablation studies provide:
- **Systematic validation** of each technique individually
- **Performance metrics** on Tiny ImageNet (200 classes)
- **Training efficiency** analysis
- **Generalization gap** assessment
- **Reproducible results** with detailed notebooks

These studies ensure that techniques applied to ImageNet 1K are well-validated and provide expected performance improvements.

## Advanced Training Techniques

### Progressive Image Resizing
- **Epochs 1-30**: Train on 64x64 images with batch size 256
- **Epochs 31-50**: Train on 128x128 images with batch size 128  
- **Epochs 51-90**: Train on 224x224 images with batch size 64
- **Benefits**: Faster initial training, better convergence, reduced memory usage

### Stochastic Depth (Drop Path)
- Linear drop path rate scheduling across all ResNet blocks
- Reduces overfitting and improves generalization
- Default rate: 0.2 (configurable via `--drop-path-rate`)

### Advanced Data Augmentation
- **Albumentations**: Professional-grade augmentation library
- **MixUp**: Mixes images and labels with random weights
- **CutMix**: Cuts and pastes patches between images
- **Coarse Dropout**: Random rectangular patches for robustness

### Mixed Precision Training
- Automatic mixed precision (AMP) with gradient scaling
- Gradient clipping for training stability
- Faster training with minimal accuracy impact

### OneCycleLR Scheduling
- Cosine annealing learning rate schedule
- Automatic warmup and cooldown phases
- Optimal learning rate finding with `lr_finder.py`


