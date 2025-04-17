# MPSS (Metamorphic test case Pair Selection with Surrogate model)
This repository is for paper "Boosting the Cost-Effectiveness of Metamorphic Test Case Pair Selection for Deep Learning Testing with Surrogate Model"
### Introduction
  MPSS is a black-box Metamorphic test case Pairs selection method for DNN testing. It can increase the cost-effectiveness of MP selection by maximizing detected failures while minimizing DNN calling times under given test budgets. In order to evaluate the performance of MPSS, we selected two popular datasets of image classification tasks, CIFAR-10 and Oxford-IIIT Pet to perform the experiment. In particular, we trained three DNN models for each dataset (including GoogLeNet, ResNet50 and Inception V3), and used five common MRs of image classification.

__Here are the brief introduction of files:__

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

### Data Preprocessing 
In our experiment, we use two popular data sets as source test cases, CIFAR-10 and Oxford-IIIT Pet. To download, please go to:  
__CIFAR-10:__ https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders  
__Oxford-IIIT Pet:__ https://www.robots.ox.ac.uk/~vgg/data/pets/  
There are 10000 images (10 labels) in the test set of CIFAR-10; approximately 7400 images in Oxford-IIIT Pet (37 labels).  

__(1) Copy pictures:__  
Metamorphic Test can alleviate the test oracle problem in DNN testing (i.e., test data with labels unkown). We should reorganize the image file structure to eliminate label information. The original file structure of downloaded data sets should be like:  
Parent folder/  
├── Label folder 1/    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── image 1-001  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  
├── Label folder 2/    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── image 2-001  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...  
├── Label folder 3/    
...  
Uitilize __Copy_pictures.py__ to combine all pictures from their label subfolders into a new fold like:  
Data path/  
├── image 1-001  
...  
├── image 2-001  
...  

__(2) Obtain and save latent space features:__  
To facilitate later use in the code，original data and latent space features of source and follow-up test cases may be obtained and saved. Call Extract_features.py in this step. The offered Saved_Data_10.zip is the output of CIFAR-10 test sets. You may use it as a reference. __*Moreover, parameter 'data_dir' should be data path from last step, and this will not be reiterated in the following.*__

### MPSS
If you are following our steps for replication, you can use pretrained DNN weights offered in DNN weights.zip. You need to train new models if you are using other data sets or DNN models.  

__(1) Specify models:__  
Rewrite model file GoogLeNet_model.py; ResNet_model.py; InceptionV3_model.py for different conditions.  
For example, __*GoogLeNet for CIFAR-10*__ (GoogLeNet_model.py):  
```
import torch
from torchvision import models
import torch.nn as nn

# If you are using offered weights, set model_path to downloaded .pth
# If you are using new weights, set model_path to your DNN weights file and change the below details of DNN
num_classes = 10 # 10 for CIFAR-10, 37 for Oxford-IIIT Pet
model_path = "~/Googlenet_cifar10.pth" # pretrained model weights

# Initialize the GoogLeNet model
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model = models.googlenet(init_weights=False, aux_logits=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()
```

__(2) Changes to Main Process Parameters:__  
Call file Update_SVM.py for main process. Rewrite some parameters for different conditions:
```
...
from GoogLeNet_model import model,device # DNN for test                              Line13
...
data_dir = '~' # Your image path, this file only has images, no subfolder            Line16
...
computational_process_times = 5 # K for MPs selection times                          Line18
total_select_MPs = 10000 # MPs selection number C                                    Line19
...
output_dir = os.path.join(os.getcwd(), "Saved_Data_10") # data preprocessing folder  Line52
```  
Read Update_SVM.py in details for more information.

### Baseline Method  
There are three baseline compared with MPSS: Random Selection; Uncertainty Based Selection; NSGA-II Based Selection. The number of selected MPs that violate MRs *# failed MPs* and # failed MPs/# DNN calls for cost-effectiveness evaluation can be calculated to illustrate the ability of different methods.

__(1) Random Selection:__  
Please read Random_selection.py and rewrite some parameters similiar to what you did in Update_SVM.py.

__(2) Uncertainty Based Selection:__  
Please read Uncertainty_selection.py and rewrite some parameters similiar to what you did in Update_SVM.py.

__(3) NSGA-II Based Selection:__  
Recently, Arrieta has proposed an NSGA-II based method to select follow-up test cases for DNN testing. You may go to https://dl.acm.org/doi/10.1145/3512290.3528697 for more details. We rewrote his code into Python to offer a fair comparison. Please read Multi-objective_NSGA2.py and rewrite some parameters similiar to what you did in Update_SVM.py.
