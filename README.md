# MPSS (Metamorphic test case Pair Selection with Surrogate model)
This repository is for paper "Boosting the Cost-Effectiveness of Metamorphic Test Case Pair Selection for Deep Learning Testing with Surrogate Model"
### Introduction
  MPSS is a black-box Metamorphic test case Pairs selection method for DNN testing. It can increase the cost-effectiveness of MP selection by maximizing detected failures while minimizing DNN calling times under given test budgets. In order to evaluate the performance of MPSS, we selected two popular datasets of image classification tasks, CIFAR-10 and Oxford-IIIT Pet to perform the experiment. In particular, we trained three DNN models for each dataset (including GoogLeNet, ResNet50 and Inception V3), and used five common MRs of image classification.

Here are the brief introduction of files:
1、DNN weights.zip
​	It is the zip file of DNN weights of three models on two data set. 
​	DNN weights/
​	├── GoogLeNet_Pets.pth
​	├── Googlenet_cifar10.pth
​	├── InceptionV3_Pets.pth
​	├── InceptionV3_cifar10.pth
​	├── ResNet50_Pets.pth
​	└── ResNet50_cifar10.pth
