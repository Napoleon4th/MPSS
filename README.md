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

2、Saved_Data_10.zip  
  It is the zip file of data preprocessing result of CIFAR-10 test set. You may use it as a reference.  
  Saved_Data_10/    
​	├── P_0.pkl           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#source test cases original image data (M for test cases count)  
​	├── P.pkl             &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#follow-up test cases image data (N\*M, N for MRs number)  
​	├── P_0_features.pkl  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#source test cases latent space features (M)  
​	└── P_features.pkl    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#follow-up test cases latent space features (N\*M)    

3、Copy_pictures.py  
  Since Metamorphic Test is for data without labels, we should combine all pictures from their label subfolders into a new fold.

4、Extract_features.py  
  It is for data prepocessing. The output should be similar to Saved_Data_10.zip.  

5、GoogLeNet_model.py; ResNet_model.py; InceptionV3_model.py  
  The model component was modularized into a separate Python file for better code organization. You may import it when needed.  

6、Random_selection.py  
  It is the code of *Random Selection* method compared with MPSS.  

7、Uncertainty_selection.py  
  It is the code of *Uncertainty Based Selection* method compared with MPSS.  

8、Multi-objective_NSGA2.py  
  It is the code of *NSGA-II Based Selection* method compared with MPSS.  

9、image_transformations.py
  It defines the details of MRs used in the experiment and it offers functions for importing.  

10、Update_SVM.py  
  It is the main code of MPSS.
