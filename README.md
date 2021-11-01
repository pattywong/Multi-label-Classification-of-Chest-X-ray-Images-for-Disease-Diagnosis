# Multi-label Classification of Chest X-ray Images for Common Thoracic Diseases
This is an experimental comparison of methods for multi-label classification of chest X-ray images for 5 pathologies, namely Cardiomegaly, Lung Opacity, Edema, Atelectasis, and Pleural Effusion. The experiments include problem transformation methods, hierachical learning procedure method, and custom loss function, conducted to provide better insight into different approaches and their applications to multi-label classification. The model is based on Densely Connected Convolutional Networks, [DenseNet121](https://arxiv.org/abs/1608.06993), and built over a large dataset of chest X-rays for thoracic disease recognition, [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/). In this study, the methods for multi-label learning are evaluated using example-based and label-based proposed metrics for a comprehensive overview of methods and revealing the best performing method for the classification task.

The repository includes:
- model script
- preprocessing data script
- Training and inference script
- evaluating script
- configuration file

## Introduction
Diagnosis in chest radiography is considered to be a multi-label and fine-grained problem due to a single scan may be associated with multiple diseases simultaneously where diseases are visually similar and hard-to-distinguish. There are some natural correlations among the disease labels however. However, the basic multi-label learning algorithm doesn't consider label dependencies and leads to label confusion. In this study, 4 different methods of multi-label learning are applied to see how models improve after exploiting disease and labels dependencies.

Experiments of multi-label learning methods:

0. N binary classifiers - Benchmark 
1. [Label Powerset (LP)](https://www.researchgate.net/publication/263813673_A_Review_On_Multi-Label_Learning_Algorithms) - Problem transformation method : multi-class classification
2. [Classifier Chains (CC)](https://www.researchgate.net/publication/263813673_A_Review_On_Multi-Label_Learning_Algorithms) - Problem transformation method that considers label correlations
3. [Custom loss](https://link.springer.com/article/10.1007%2Fs11042-019-08260-2) - A pair of novel loss functions considering the label relationship between present and absent classes
4. [Hierarchical training method](https://arxiv.org/abs/1911.06475) - Training procedure to exploit label dependencies

# Getting started
## Requirement
- Python 
- OpenCV 
- Tensorflow 
- Keras 

## Dataset

[CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) is a large dataset of chest X-rays and competition for automated chest x-ray interpretation contains 224,316 chest radiographs of 65,240 patients with both frontal and lateral views available. The task is to do automated chest x-ray interpretation, featuring uncertainty labels and radiologist-labeled reference standard evaluation sets.

## DenseNet121

## Training model

## Inference

## Results and Analysis

## Conclusion

## Citations
