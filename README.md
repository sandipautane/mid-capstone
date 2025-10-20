This folder contains the google colab notebook runs for ablation run studies on **Tiny ImageNet** dataset so that we are able to access the tecchniques that need to be applied on ImageNet 1k.

Following Notebooks are added:

1. **ablation_run_tiny_imagenet_01.ipynb**: experimented with the following techniques in this notebook (https://colab.research.google.com/drive/12ICjPtVgA7EpOD2QFIMIHHTX9ikDOrbx?usp=sharing):
   *  ResNet50 basic architetcure
   *  ResNet50 pre ativation architecture (without ReLu in skip connection)
   *  ResNet50 basic architetcure with conv layer instead of last FC layer.
   *  These architectures have been tried with --> lr finder, image augmentations, mixup, mixed precision training
  
**Conclusion**: final best acc was obtained with ResNet50 basic architetcure with conv layer instead of last FC layer. Learning rate finding needs to be honed!!.
 Refer to the notebook for each run detail and observations.

2. **ablation_run_tiny_imagenet_02.ipynb**: (https://colab.research.google.com/drive/1pA6fT3FFl-X-fMqPbr9s2KTZ77YqfbZW?usp=sharing) experimented with following techniques in this notebook:
     * Progressive Batch Resizing
     * CutMix along with MixUp
     * Stochastic Depth
   
**Conclusion**:Progressive Batch Resizing and CutMix along with MixUp really push the accuracy ~10% higher, while using Stochastic Depth reduces the difference b/w test and train very low.

3. **ablation_run_tiny_imagenet_03.ipynb**: (https://colab.research.google.com/github/sandipautane/mid-capstone/blob/main/ablation_run_tiny_imagenet_03.ipynb) experimented with following techniques in this notebook:
     * ResNet50 with conv layer instead of FC layer
     * Mixed precision training with gradient clipping
     * MixUp and CutMix data augmentation
     * Progressive Batch Size Resizing (64→128→224)
     * Stochastic Depth (DropPath)
     * **Exponential Moving Average (EMA)**
   
**Conclusion**: Achieved ~67% accuracy with the combination of all techniques. Stochastic Depth significantly reduced the train-test gap. EMA provided better validation performance through parameter smoothing.

## Key Learning: Exponential Moving Average (EMA)

**EMA Implementation Insights:**
- **Decay Rate**: Used `decay=0.9999` (very high decay for stability)
- **Update Mechanism**: `new_average = (1.0 - decay) * current_param + decay * shadow_param`
- **Validation Strategy**: Use EMA weights for validation, regular weights for training

**EMA Learning Benefits:**
- **Smoothing Effect**: Creates stable, noise-reduced parameter averages
- **Better Generalization**: EMA weights often perform better on validation sets
- **Training Stability**: High decay rate provides stability during aggressive training techniques
- **Regularization**: Acts as implicit regularization by maintaining a "teacher" model

**Key Implementation Pattern:**
```python
# During training
ema.update()  # Update shadow weights after each step

# During validation
ema.apply_shadow()  # Use EMA weights
val_loss, val_acc = test(model, device, val_loader, epoch)
ema.restore()       # Restore original weights
```

**Initial Challenges**: First EMA attempt failed (accuracy stuck at 0.5%) due to improper decay parameter tuning, highlighting the importance of careful EMA configuration.
**We should not use EMA in the final run!!**

---

## ImageNet-1K Training with DDP (Distributed Data Parallel)

This repository now supports both **single-GPU** and **multi-GPU** training on ImageNet-1K dataset using PyTorch's DistributedDataParallel (DDP).

### 🚀 Hardware Setup: AWS EC2 p3.8xlarge

**Instance Specifications:**
- **Instance Type**: p3.8xlarge
- **GPUs**: 4x NVIDIA V100 (16GB each)
- **vCPUs**: 32
- **RAM**: 244 GB
- **Network**: Up to 10 Gbps
- **Spot Price**: ~$1.20-1.50/hour
- **On-Demand Price**: ~$12.24/hour

**Cost Estimates for 75% Accuracy:**
- **Training Time**: ~8-10 hours (with 4 GPUs)
- **Total Cost**: ~$10-15 (using spot instances)
- **Cost Savings**: ~90% compared to on-demand pricing

### 📊 Performance Comparison

| Setup | GPUs | Batch Size | Training Time | Cost (Spot) | Speedup |
|-------|------|------------|---------------|-------------|---------|
| Single GPU | 1x V100 | 256 | ~32-40 hours | ~$40-60 | 1x |
| DDP (Multi-GPU) | 4x V100 | 256 per GPU | ~8-10 hours | ~$10-15 | ~4x |

**Note**: With DDP, each GPU processes a batch independently, so effective batch size = 256 × 4 = 1024

---

## 🔧 Training Commands

### Single GPU Training (Without DDP)

```bash
# Basic training
python train.py \
    --data-dir /path/to/ILSVRC \
    --epochs 90 \
    --batch-size 256 \
    --lr 3e-4 \
    --num-workers 8 \
    --save-dir ./checkpoints

# With subset for testing
python train.py \
    --data-dir /path/to/ILSVRC \
    --subset \
    --subset-size 10000 \
    --epochs 30 \
    --batch-size 256 \
    --lr 3e-4
```

### Multi-GPU Training with DDP (4x V100)

```bash
# Full ImageNet-1K training with 4 GPUs
python train.py \
    --ddp \
    --world-size 4 \
    --data-dir /path/to/ILSVRC \
    --epochs 90 \
    --batch-size 256 \
    --lr 1.2e-3 \
    --num-workers 8 \
    --save-dir ./checkpoints \
    --drop-path-rate 0.2 \
    --use-blurpool

# With subset for testing (faster iteration)
python train.py \
    --ddp \
    --world-size 4 \
    --data-dir /path/to/ILSVRC \
    --subset \
    --subset-size 10000 \
    --epochs 30 \
    --batch-size 256 \
    --lr 1.2e-3
```

---

## 📈 Learning Rate Scaling for DDP

When using DDP with multiple GPUs, you need to scale the learning rate based on the effective batch size:

**Linear Scaling Rule**: `LR_multi_gpu = LR_single_gpu × sqrt(num_gpus)` or `LR_multi_gpu = LR_single_gpu × num_gpus`

### Recommended Learning Rates:

| Setup | Base LR | Max LR (OneCycleLR) | Scaling Factor |
|-------|---------|---------------------|----------------|
| **1 GPU** | 3e-4 | 3e-4 | 1x |
| **4 GPUs (Conservative)** | 6e-4 | 1.2e-3 | 2x (sqrt scaling) |
| **4 GPUs (Aggressive)** | 1.2e-3 | 2e-3 | 4x (linear scaling) |

**Recommendation**: Start with **2x scaling** (LR = 1.2e-3) for stability, then experiment with higher rates if training is stable.

```bash
# Conservative (safer, recommended for first run)
python train.py --ddp --world-size 4 --lr 1.2e-3 --data-dir /path/to/ILSVRC

# Aggressive (faster convergence, may be unstable)
python train.py --ddp --world-size 4 --lr 2e-3 --data-dir /path/to/ILSVRC
```

---

## 🗂️ Dataset Structure

The code supports both **ILSVRC** (official ImageNet) structure and simple **train/val** folder structure:

### ILSVRC Structure
```
ILSVRC/
├── Data/
│   └── CLS-LOC/
│       ├── train/
│       │   ├── n01440764/
│       │   │   ├── n01440764_0.JPEG
│       │   │   └── ...
│       │   ├── n01443537/
│       │   └── ... (1000 classes)
│       └── val/
│           ├── ILSVRC2012_val_00000001.JPEG
│           ├── ILSVRC2012_val_00000002.JPEG
│           └── ... (50,000 images)
└── ImageSets/
    └── CLS-LOC/
        └── val.txt  # Format: "image_name class_index"
```

### Simple Structure (Backward Compatible)
```
imagenet/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

---

## 🏗️ EC2 Setup Guide

### 1. Launch p3.8xlarge Spot Instance

```bash
# Use AWS Deep Learning AMI (Ubuntu 20.04)
# AMI: Deep Learning AMI GPU PyTorch 2.1.0 (Ubuntu 20.04)
# Storage: 500GB EBS volume for ImageNet dataset
```

### 2. Install Dependencies

```bash
# Clone repository
git clone https://github.com/sandipautane/mid-capstone.git
cd mid-capstone

# Install requirements
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
# Expected output: GPUs available: 4
```

### 3. Download ImageNet Dataset

```bash
# Option 1: Download from official source (requires registration)
# https://image-net.org/challenges/LSVRC/2012/

# Option 2: Use existing S3 bucket or EFS mount
# aws s3 sync s3://your-bucket/ILSVRC /data/ILSVRC

# Verify dataset structure
ls /data/ILSVRC/Data/CLS-LOC/train | wc -l  # Should be 1000
ls /data/ILSVRC/Data/CLS-LOC/val | wc -l    # Should be 50000
```

### 4. Start Training

```bash
# Run in tmux/screen to prevent disconnection
tmux new -s training

# Start DDP training
python train.py \
    --ddp \
    --world-size 4 \
    --data-dir /data/ILSVRC \
    --epochs 90 \
    --batch-size 256 \
    --lr 1.2e-3 \
    --num-workers 8 \
    --save-dir ./checkpoints \
    --drop-path-rate 0.2 \
    --use-blurpool

# Detach from tmux: Ctrl+B, then D
# Reattach: tmux attach -t training
```

---

## 🎯 Training Features

### Progressive Image Resizing
The training script automatically adjusts image sizes during training:
- **Epochs 1-30**: 64×64 (batch size: 256)
- **Epochs 31-50**: 128×128 (batch size: 128)
- **Epochs 51-90**: 224×224 (batch size: 64)

This technique speeds up early training and improves final accuracy.

### Data Augmentation
- **MixUp** (alpha=0.2): Blends pairs of images and labels
- **CutMix** (alpha=1.0): Cuts and pastes image patches
- **RandomHorizontalFlip**: 50% probability
- **ShiftScaleRotate**: Slight geometric transformations
- **ColorJitter**: Brightness, contrast, saturation adjustments
- **CoarseDropout**: Random rectangular cutout

### Regularization Techniques
- **Stochastic Depth (DropPath)**: Drop rate = 0.2
- **Label Smoothing**: 0.1
- **Weight Decay**: 1e-4
- **Gradient Clipping**: Max norm = 1.0
- **Mixed Precision Training**: Automatic with GradScaler

### Optimizer & Scheduler
- **Optimizer**: AdamW
- **Scheduler**: OneCycleLR
  - pct_start: 0.3 (30% warmup)
  - anneal_strategy: cosine

---

## 🔍 Monitoring Training

### Check Training Progress

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f nohup.out  # or your log file

# Check checkpoint directory
ls -lh checkpoints/
```

### Expected Training Metrics

| Epoch | Image Size | Batch Size | Time/Epoch | Val Accuracy |
|-------|------------|------------|------------|--------------|
| 1-30 | 64×64 | 256 | ~15 min | 40-50% |
| 31-50 | 128×128 | 128 | ~25 min | 60-65% |
| 51-90 | 224×224 | 64 | ~40 min | 70-75% |

**Total Training Time**: ~8-10 hours on 4x V100

---

## 🐛 Troubleshooting

### Issue: NCCL Error on Windows
**Solution**: NCCL backend doesn't work on Windows. Use Linux (Ubuntu 20.04 recommended) or add Gloo backend fallback.

### Issue: Out of Memory (OOM)
**Solutions**:
- Reduce batch size: `--batch-size 128` or `--batch-size 64`
- Reduce number of workers: `--num-workers 4`
- Use gradient accumulation (requires code modification)

### Issue: Slow Data Loading
**Solutions**:
- Increase workers: `--num-workers 16`
- Use faster storage (NVMe SSD or instance store)
- Enable `pin_memory=True` (already enabled)

### Issue: DDP Hangs at Initialization
**Solutions**:
- Check firewall settings (port 12355 should be open)
- Verify all GPUs are visible: `echo $CUDA_VISIBLE_DEVICES`
- Try different port: Modify `MASTER_PORT` in [train.py:31](train.py#L31)

### Issue: Validation Accuracy Not Improving
**Solutions**:
- Lower learning rate: Try `--lr 6e-4` instead of `1.2e-3`
- Check data augmentation (may be too aggressive)
- Increase training epochs
- Verify dataset labels are correct

---

## 📦 Key Files

- **train.py**: Main training script with DDP support
- **dataloader.py**: Dataset loaders for ILSVRC and simple structures
- **train_utils.py**: Training/validation loops, mixup, cutmix
- **data_utils.py**: Data augmentation utilities
- **models/model.py**: ResNet50 with Stochastic Depth and BlurPool
- **requirements.txt**: Python dependencies

---

## 🎓 Advanced Usage

### Resume Training from Checkpoint

```python
# Add to train.py (requires code modification)
if args.resume:
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
```

### Custom Learning Rate Schedule

```bash
# Modify in train.py for different schedules
# Options: CosineAnnealingLR, StepLR, ExponentialLR
```

### Enable TensorBoard Logging

```bash
# Requires code modification to add SummaryWriter
pip install tensorboard
tensorboard --logdir=runs --port=6006
```

---

## 📚 References

- [ImageNet Large Scale Visual Recognition Challenge](https://image-net.org/challenges/LSVRC/)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/notes/amp_examples.html)
- [Progressive Resizing Paper](https://arxiv.org/abs/1905.00546)
- [MixUp Paper](https://arxiv.org/abs/1710.09412)
- [CutMix Paper](https://arxiv.org/abs/1905.04899)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
