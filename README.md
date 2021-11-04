# Multi-label Classification of Chest X-ray Images for Common Thoracic Diseases
This is an experimental comparison of methods for multi-label classification of chest X-ray images. The experiments include problem transformation methods, hierachical learning procedure method, and custom loss function, conducted to provide better insight into different approaches and their applications to multi-label classification. The model is based on Densely Connected Convolutional Networks, [DenseNet121](https://arxiv.org/abs/1608.06993), and built over a large dataset of chest X-rays for thoracic disease recognition, [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/). In this study, the methods for multi-label learning are evaluated using example-based and label-based proposed metrics for a comprehensive overview of methods and revealing the best performing method for the classification task.

The repository includes:
- model script
- preprocessing data script
- Training and inference script
- evaluating script
- configuration file

## Introduction
Diagnosis in chest radiography is considered to be a multi-label and fine-grained problem due to a single scan may be associated with multiple diseases simultaneously where diseases are visually similar and hard-to-distinguish. There are some natural correlations among the disease labels. However, the basic multi-label learning algorithm doesn't consider label dependencies and leads to label confusion. In this study, 4 different methods of multi-label learning are applied to see model performace improvement after exploiting disease and labels dependencies.

Experiments of multi-label learning methods:

0. N binary classifiers - Benchmark 
1. [Label Powerset (LP)](https://www.researchgate.net/publication/263813673_A_Review_On_Multi-Label_Learning_Algorithms) - Problem transformation method : multi-class classification
2. [Classifier Chains (CC)](https://www.researchgate.net/publication/263813673_A_Review_On_Multi-Label_Learning_Algorithms) - Problem transformation method that considers label correlations
3. [Custom loss](https://link.springer.com/article/10.1007%2Fs11042-019-08260-2) - A pair of novel loss functions considering the label relationship between present and absent classes
4. [Hierarchical training method](https://arxiv.org/abs/1911.06475) - Training procedure to exploit label dependencies

## Dataset
[CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) is a large dataset of chest X-rays and competition contains 224,316 chest radiographs of 65,240 patients, containing both frontal and lateral views, labelled for 14 common CXR observations where each observation are either 0 (negative), 1 (positive), or u (uncertain), available for automated chest x-ray interpretation, featuring uncertainty labels and radiologist-labeled reference standard evaluation sets.

![dataset_ex](/assets/frontal_and_lateral_cxr.png)

The experiments of methods are conducted on 102,942 chest radiograph images with frontal views from CheXpert focusing on 5 common pathologies, namely Cardiomegaly, Lung Opacity, Edema, Atelectasis, and Pleural Effusion, where samples with uncertainty labels are ignored.

## Model Architecure
[DenseNet121](https://arxiv.org/abs/1608.06993v5) is one of the new discoveries in convolutional neural networks that utilises dense connections between layers, through Dense Blocks, where we connect all layers (with matching feature-map sizes) directly with each other. To preserve the feed-forward nature, each layer obtains additional inputs from all preceding layers and passes on its own feature-maps to all subsequent layers. On four highly competitive object recognition benchmark tasks (CIFAR-10, CIFAR-100, SVHN, and ImageNet), DenseNets obtain significant improvements over the state-of-the-art on most of them, whilst requiring less computation to achieve high performance.

![model](/assets/densenet_ex.png)

## Requirements
- Python 3.6.5
- Tensorflow 1.13.1
- Keras 2.2.4
- Other packages in `requirements.txt`

## Installation

### Installing all packages using pip

Without using Anaconda, you will have to download and install respective versions of cudatoolkit and cudnn manually. Then, you have to set path variables for your systems.

```
pip install -r requirements.txt
```

### Installing TensorFlow using conda

The advantage of using Anaconda to install is you can download tensorflow-gpu with respective versions of dependent packages all at once. You will not have to install cudatoolkit and cudnn manually from Nvidia because the packages are included in tensorflow-gpu, then install others packages using pip to complete.

```
conda install tensorflow-gpu
```

### Installing Microsoft Visual Studio
Cuda libraries will be compiled using MSVS as a compiler. It is necessary to select the correct version of MSVS depending on the version of the Cuda toolkit you are selecting. You can download the Community Version of Microsoft Visual Studio [here](https://visualstudio.microsoft.com/downloads/).

## Training model

## Inference

## Results and Analysis

## Conclusion

## Citations














