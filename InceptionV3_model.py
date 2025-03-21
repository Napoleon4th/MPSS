import torch
from torchvision import models
import torch.nn as nn

# If you are using offered weights, set model_path to downloaded .pth
# If you are using new weights, set model_path to your DNN weights file and change the below details of DNN
num_classes = 10 # 10 for CIFAR-10, 37 for Oxford-IIIT Pet
model_path = "~"

# Initialize the Inception V3 model
model = models.inception_v3(init_weights=True, aux_logits=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()