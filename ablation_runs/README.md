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

3. **reading_imagenet_data.ipynb**: this notebook contains the actual methods to be used for reading ImageNet data set based on it's actual folder structure.
