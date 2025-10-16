This folder contains the google colab notebook runs for ablation run studies on **Tiny ImageNet** dataset so that we are able to access the tecchniques that need to be applied on ImageNet 1k.

Following Notebooks are added:

1. **ablation_run_tiny_imagenet_01.ipynb**: experimented with the following techniques in this notebook (https://colab.research.google.com/drive/12ICjPtVgA7EpOD2QFIMIHHTX9ikDOrbx?usp=sharing):
   *  ResNet50 basic architetcure
   *  ResNet50 pre ativation architecture (without ReLu in skip connection)
   *  ResNet50 basic architetcure with conv layer instead of last FC layer.
   *  These architectures have been tried with --> lr finder, image augmentations, mixup, mixed precision training

   **Conclusion**: final best acc was obtained with ResNet50 basic architetcure with conv layer instead of last FC layer. Learning rate finding needs to be honed!!.
   Refer to the notebook for each run detail and observations.

3. ablation_run_tiny_imagenet_02.ipynb
