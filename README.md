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
