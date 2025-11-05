## Initial Ablation Runs

This folder contains the google colab notebook runs for ablation run studies on **Tiny ImageNet** dataset so that we are able to access the techniques that need to be applied on ImageNet 1k.

Following Notebooks are added:

1. **ablation_run_tiny_imagenet_01.ipynb**: experimented with the following techniques in this notebook (https://colab.research.google.com/drive/12ICjPtVgA7EpOD2QFIMIHHTX9ikDOrbx?usp=sharing):
   *  ResNet50 basic architecture
   *  ResNet50 pre ativation architecture (without ReLu in skip connection)
   *  ResNet50 basic architecture with conv layer instead of last FC layer.
   *  These architectures have been tried with --> lr finder, image augmentations, mixup, mixed precision training
  
**Conclusion**: final best acc was obtained with ResNet50 basic architecture with conv layer instead of last FC layer. Learning rate finding needs to be honed!!.
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

## Actual ImageNet-1K Training 

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
| Single GPU | 1x V100 | 128 | ~32-40 hours | ~$40-60 | 1x |
| DDP (Multi-GPU) | 4x V100 | 128 per GPU | ~8-10 hours | ~$10-15 | ~4x |

**Note**: With DDP, each GPU processes a batch independently, so effective batch size = 128 × 4 = 512

---

## 🔧 Training Commands

### Single GPU Training (Without DDP)

```bash
# Basic training
python train.py \
    --data-dir /path/to/ILSVRC \
    --epochs 110 \
    --batch-size 128 \
    --num-workers 8 \
    --save-dir ./checkpoints

# With subset for testing
python train.py \
    --data-dir /path/to/ILSVRC \
    --subset \
    --subset-size 10000 \
    --epochs 30 \
    --batch-size 128

# Resume from checkpoint
python train.py \
    --data-dir /path/to/ILSVRC \
    --resume ./checkpoints/best_resnet50_imagenet_1k.pt \
    --epochs 110 \
    --batch-size 128
```

### Multi-GPU Training with DDP (4x V100)

```bash
# Full ImageNet-1K training with 4 GPUs
python train.py \
    --ddp \
    --world-size 4 \
    --data-dir /path/to/ILSVRC \
    --epochs 110 \
    --batch-size 128 \
    --num-workers 8 \
    --save-dir ./checkpoints \
    --drop-path-rate 0.2

# With BlurPool (optional, use --use-blurpool flag)
python train.py \
    --ddp \
    --world-size 4 \
    --data-dir /path/to/ILSVRC \
    --epochs 110 \
    --batch-size 128 \
    --num-workers 8 \
    --save-dir ./checkpoints \
    --drop-path-rate 0.2 \
    --use-blurpool

# Resume from checkpoint with DDP
python train.py \
    --ddp \
    --world-size 4 \
    --data-dir /path/to/ILSVRC \
    --resume ./checkpoints/best_resnet50_imagenet_1k.pt \
    --epochs 110 \
    --batch-size 128 \
    --drop-path-rate 0.2

# With subset for testing (faster iteration)
python train.py \
    --ddp \
    --world-size 4 \
    --data-dir /path/to/ILSVRC \
    --subset \
    --subset-size 10000 \
    --epochs 30 \
    --batch-size 128
```

### Additional Training Script (Fine-tuning from Checkpoint)

```bash
# Fine-tune from checkpoint with custom settings
python train_additional.py \
    --checkpoint ./checkpoints/best_resnet50_imagenet_1k.pt \
    --data-dir /path/to/ILSVRC \
    --epochs 10 \
    --lr 1e-4 \
    --max-lr 5e-4 \
    --image-size 320 \
    --batch-size 64 \
    --drop-path-rate 0.05 \
    --no-mixup  # Disable mixup/cutmix for fine-tuning

# With BlurPool enabled
python train_additional.py \
    --checkpoint ./checkpoints/best_resnet50_imagenet_1k.pt \
    --data-dir /path/to/ILSVRC \
    --epochs 10 \
    --lr 1e-4 \
    --max-lr 5e-4 \
    --image-size 320 \
    --batch-size 64
```

### Learning Rate Finder

```bash
# Find optimal learning rate
python lr_finder.py \
    --data-dir /path/to/ILSVRC \
    --batch-size 128 \
    --image-size 224 \
    --num-iter 300 \
    --save-plot lr_finder_plot.png

# With subset for faster LR finding
python lr_finder.py \
    --data-dir /path/to/ILSVRC \
    --subset \
    --subset-size 10000 \
    --batch-size 128 \
    --image-size 224 \
    --num-iter 200
```

---

## 📈 Learning Rate Configuration

The training script uses a **single continuous OneCycleLR scheduler** that spans all training phases:

- **Initial LR**: 1e-3 (automatically set)
- **Max LR**: 4e-3 (peak learning rate, used for entire training)
- **Strategy**: Single continuous scheduler with natural decay across all phases

### OneCycleLR Schedule:
1. **Warmup Phase** (30% of total training): LR increases from `1e-3` to `4e-3`
2. **Annealing Phase** (70% of total training): LR decreases from `4e-3` to very low values

**Note**: The scheduler automatically adjusts across progressive resizing phases without recreation. The learning rate naturally decays as training progresses through all phases.

### Recommended Training Epochs:

- **Full Training**: 110 epochs (covers all 4 phases)
- **Quick Test**: 30-40 epochs (covers Phase 1 and start of Phase 2)

---

## 🗂️ Dataset Structure

The code supports both **ILSVRC** (official ImageNet) structure with XML annotations and simple **train/val** folder structure:

### ILSVRC Structure (Recommended)
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
├── Annotations/
│   └── CLS-LOC/
│       ├── train/
│       │   ├── n01440764_0.xml
│       │   └── ... (XML annotation files)
│       └── val/
│           ├── ILSVRC2012_val_00000001.xml
│           └── ... (XML annotation files)
└── ImageSets/
    └── CLS-LOC/
        └── val.txt  # Format: "image_name class_index" (optional)
```

**Note**: If XML annotations are present, the code automatically uses them for accurate class mapping. Otherwise, it falls back to folder-based structure.

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

### 1. Download ImageNet Dataset

* Create **EBS (350 GB) volume** to download dataset 

<img width="450" height="250" alt="image" src="https://github.com/user-attachments/assets/4194ed21-f9ed-4980-adf9-942958e38700" />

* Creat a windows instance to attach volume and download dataset

<img width="450" height="250" alt="image" src="https://github.com/user-attachments/assets/a5b28a89-3b94-44ca-ae0e-c96822a7ce54" />

<img width="450" height="250" alt="image" src="https://github.com/user-attachments/assets/4b21d56d-c02e-4fbc-b936-7b8d97abf82d" />

* Network Settings -> Edit chose Availibility Zone -> select the availability zone same as EBS volume created

* Attach the EBS volume created to the windows instance.

* Download and install free download manager: https://www.freedownloadmanager.org/landing.htm
* Also download and install 7zip

Now open kaggle data link in browser:

[ImageNet Object Localization Challenge | Kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data?select=ILSVRC) (sign in)

Go to inspect -> network tab in browser

Click on download -> pause the actual download, copy the download url from network tab

Open free download manager -> paste url Choose D drive and start download (will take approx. 25 mins)

Now disable windows defender -> Set-MpPreference -DisableRealtimeMonitoring $true

use 7zip to browse ILSVRC folders and start download one by one (should take 25 mins)


### 2. Launch g5.2xlarge Spot Instance

```bash
# Use AWS Deep Learning AMI (Ubuntu 20.04)
# AMI: Deep Learning AMI GPU PyTorch 2.1.0 (Ubuntu 20.04)
# Storage: 350GB EBS volume for ImageNet dataset
```
<img width="450" height="250" alt="image" src="https://github.com/user-attachments/assets/fa2d699c-031b-4fd9-be26-a4d7791d457d" />


<img width="450" height="250" alt="image" src="https://github.com/user-attachments/assets/281e20db-ee56-4bcb-bfcb-377c25611be6" />


attach EBS volume to this instance


### 3. Install Dependencies

```bash
# Install tmux
sudo apt update
sudo apt install tmux

# create session in tmux
tmux new -s mysession

# install ntfs file manager
sudo apt install ntfs-3g

# make a new directory and use it to mount drive (the EBS volume)
sudo mkdir -p /mnt/ntfs

# now attach drive (use lsblk command to check drive name in instance)
sudo mount -t ntfs-3g /dev/nvme2n1p2 /mnt/ntfs

# Now change directory to the drive
cd /mnt/ntfs

# Clone repository
git clone https://github.com/sandipautane/mid-capstone.git
cd mid-capstone

# Installing python
sudo apt install -y python3.12 python3.12-venv

# create venv
python3.12 -m venv venv

# activate venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# note - If In case git pull is required again in case code is updated, use below command
git config --global --add safe.directory /mnt/ntfs/mid-capstone
git pull origin main

# Verify GPU availability
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
# Expected output: GPUs available: 1
```

### 4. Start Training

```bash
# Run in tmux/screen to prevent disconnection
tmux new -s training

# Start DDP training
python train.py \
  --data-dir /mnt/ntfs/ILSVRC \
  --drop-path-rate 0.05 \
  --use-blurpool \
  --epochs 110

# Detach from tmux: Ctrl+B, then D
# Reattach: tmux attach -t training
```

---

## 🎯 Training Features

### Progressive Image Resizing (4 Phases)
The training script automatically adjusts image sizes during training:
- **Phase 1 (Epochs 1-15)**: 128×128 (batch size: 128)
- **Phase 2 (Epochs 16-80)**: 224×224 (batch size: 128)
- **Phase 3 (Epochs 81-95)**: 288×288 (batch size: 128)
- **Phase 4 (Epochs 96-110)**: 320×320 (batch size: 64)

**Note**: These batch sizes are per-GPU. With 4 GPUs and DDP, the effective global batch size is 4x these values.

**Phase-Specific Behavior:**
- **Phases 1-2**: Full data augmentation (MixUp, CutMix), Stochastic Depth enabled
- **Phases 3-4**: MixUp/CutMix **disabled** (fine-tuning), Stochastic Depth **disabled** (fine-tuning)

This technique speeds up early training and improves final accuracy.

### Data Augmentation
- **MixUp** (alpha=0.2): Blends pairs of images and labels (disabled in Phase 3-4)
- **CutMix** (alpha=1.0): Cuts and pastes image patches (disabled in Phase 3-4)
- **RandomHorizontalFlip**: 50% probability
- **ShiftScaleRotate**: Slight geometric transformations
- **ColorJitter**: Brightness, contrast, saturation adjustments
- **CoarseDropout**: Random rectangular cutout

### Regularization Techniques
- **Stochastic Depth (DropPath)**: Drop rate = 0.2 (disabled in Phase 3-4)
- **Label Smoothing**: 0.1
- **Weight Decay**: 1e-4
- **Gradient Clipping**: Max norm = 1.0
- **Mixed Precision Training**: Automatic with GradScaler
- **BlurPool (Optional)**: Anti-aliasing downsampling (use `--use-blurpool` flag, default: False)

### Optimizer & Scheduler
- **Optimizer**: AdamW
  - Initial LR: 1e-3 (automatically set)
  - Weight Decay: 1e-4
- **Scheduler**: OneCycleLR (single continuous scheduler)
  - Max LR: 4e-3 (peak for entire training)
  - pct_start: 0.3 (30% warmup)
  - anneal_strategy: cosine
  - **Note**: Single scheduler spans all phases, no recreation at phase transitions

### Checkpoint Resuming
The training script fully supports resuming from checkpoints:
- Saves optimizer, scheduler, and model state
- Automatically continues from the correct epoch
- Preserves learning rate schedule seamlessly
- Handles progressive resizing correctly

---

## Training Progress
Below are the snapshots of training progress!!

<img width="450" height="250" alt="image" src="https://github.com/user-attachments/assets/e58c01ac-2d06-4b96-9123-bf831b262d41" />

<img width="450" height="250" alt="image" src="https://github.com/user-attachments/assets/7a31105a-6454-4f3e-a6e4-20d545d4f7d8" />

<img width="450" height="250" alt="image" src="https://github.com/user-attachments/assets/1de6f5a8-e6a9-4cb6-845c-ed0d282fc4f9" />

<img width="450" height="250" alt="image" src="https://github.com/user-attachments/assets/ac00aea3-a4b9-4fd6-b673-12145e042388" />

<img width="450" height="250" alt="image" src="https://github.com/user-attachments/assets/a1aaa020-b1b5-4f6d-80e5-22c7ce35c5c7" />

<img width="450" height="250" alt="image" src="https://github.com/user-attachments/assets/54df40c1-be4c-4c91-9336-caf25b2b29fb" />

<img width="450" height="250" alt="image" src="https://github.com/user-attachments/assets/19751c73-e50a-42b5-aa24-e88ce91d4e76" />


## 🎨 Inference & Deployment

### Hugging Face Spaces Deployment

The model is deployed as a Gradio web interface on Hugging Face Spaces:

**Live Demo**: [https://huggingface.co/spaces/saneshashank/ImageNet1k](https://huggingface.co/spaces/saneshashank/ImageNet1k)

**Model Performance**: 71% top-1 accuracy on ImageNet-1K validation set

### Local Inference Setup

The `hf_spaces/` folder contains code for running inference locally or deploying to Hugging Face Spaces:

```bash
# Navigate to hf_spaces directory
cd hf_spaces

# Install dependencies
pip install -r requirements.txt

# Download the trained model checkpoint
# Place best_resnet50_imagenet_1k.pt in the hf_spaces directory

# Run the Gradio app locally
python app.py
```

### Inference Code Structure

The `hf_spaces/` folder contains:
- **app.py**: Gradio interface for image classification
- **model.py**: ResNet50 model definition (with BlurPool support)
- **imagenet_classes.json**: ImageNet class labels mapping
- **requirements.txt**: Dependencies for inference

### Features:
- **Top-5 Predictions**: Shows top 5 most likely classes with confidence scores
- **Interactive UI**: Upload images and get instant predictions
- **Model Configuration**: Uses BlurPool-enabled ResNet50 trained on ImageNet-1K

### Example Usage:

```python
from PIL import Image
import torch
from model import resnet50
from torchvision import transforms

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(num_classes=1000, drop_path_rate=0.0, use_blurpool=True)
checkpoint = torch.load('best_resnet50_imagenet_1k.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Predict
image = Image.open("your_image.jpg")
img_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(img_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    top5_prob, top5_idx = torch.topk(probabilities, 5)
```

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

| Phase | Epochs | Image Size | Batch Size | Time/Epoch | Val Accuracy |
|-------|--------|------------|------------|------------|--------------|
| Phase 1 | 1-15 | 128×128 | 128 | ~10 min | 40-50% |
| Phase 2 | 16-80 | 224×224 | 128 | ~25 min | 60-70% |
| Phase 3 | 81-95 | 288×288 | 128 | ~35 min | 70-75% |
| Phase 4 | 96-110 | 320×320 | 64 | ~45 min | 75-80% |

**Total Training Time**: ~10-12 hours on 4x V100 (110 epochs)

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
- Verify dataset labels are correct (especially with XML annotations)
- Check data augmentation (may be too aggressive in early phases)
- Increase training epochs (ensure you reach Phase 3-4 for fine-tuning)
- Try different learning rate (use LR finder script)

### Issue: Checkpoint Resume Not Working
**Solutions**:
- Ensure checkpoint file exists and is not corrupted
- Verify checkpoint was saved with same model architecture
- Check that `--epochs` is sufficient to continue training

---

## 📦 Key Files

- **train.py**: Main training script with DDP support and checkpoint resuming
- **train_additional.py**: Fine-tuning script for additional training from checkpoints
- **lr_finder.py**: Learning rate finder utility
- **dataloader.py**: Dataset loaders for ILSVRC (with XML annotations) and simple structures
- **train_utils.py**: Training/validation loops, mixup, cutmix, progressive resizing utilities
- **data_utils.py**: Data augmentation utilities
- **models/model.py**: ResNet50 with Stochastic Depth and BlurPool
- **hf_spaces/app.py**: Gradio inference interface
- **hf_spaces/model.py**: ResNet50 model for inference
- **requirements.txt**: Python dependencies

---

## 🎓 Advanced Usage

### Resume Training from Checkpoint

```bash
# Resume from checkpoint
python train.py \
    --data-dir /path/to/ILSVRC \
    --resume ./checkpoints/best_resnet50_imagenet_1k.pt \
    --epochs 110 \
    --batch-size 128

# The script automatically:
# - Loads model, optimizer, and scheduler state
# - Continues from the correct epoch
# - Preserves learning rate schedule
# - Handles progressive resizing correctly
```

### Fine-tuning with Custom Settings

```bash
# Use train_additional.py for flexible fine-tuning
python train_additional.py \
    --checkpoint ./checkpoints/best_resnet50_imagenet_1k.pt \
    --data-dir /path/to/ILSVRC \
    --epochs 15 \
    --lr 5e-5 \
    --max-lr 2e-4 \
    --image-size 384 \
    --batch-size 48 \
    --drop-path-rate 0.0 \
    --no-mixup
```

### Custom Learning Rate Schedule

```bash
# Use LR finder to determine optimal learning rates
python lr_finder.py \
    --data-dir /path/to/ILSVRC \
    --batch-size 128 \
    --image-size 224 \
    --num-iter 300

# Then use the suggested LR in train_additional.py
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
- [BlurPool Paper](https://arxiv.org/abs/1904.11486)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
