import os
import torch
import pickle
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms
import numpy as np

from image_transformations import flip_left_right, flip_up_down, rotate_left, rotate_right, shear_left

# Parameters set up
data_dir = '~'
output_dir = '~'  # Save file
num_classes = 10  # num of labels

#print("1")

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],std = [0.2675, 0.2565, 0.2761]),
])

# Uiitilize VGG16 without the classification layer
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
vgg16 = models.vgg16(pretrained=True)  # Load pre-trained VGG16
vgg16.classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-1])
vgg16 = vgg16.to(device)
vgg16.eval()

# print("1")

# Image paths
image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if
               fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

# Define MRs
MRs = [flip_left_right, flip_up_down, rotate_left, rotate_right, shear_left]

# Call VGG16 to form latent space features
def extract_features(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = vgg16(image)
        features = features.view(features.size(0), -1)
    return features.cpu().numpy()


# Generate follow_up test cases and latent space features for both source and follow_up
def generate_images_and_features(image_paths, MRs):
    P_0 = []  # List for original source test cases (N)
    P = [[] for _ in range(len(MRs))]  # List for follow_up test cases (M*N)

    # Process images
    for image_path in tqdm(image_paths, desc="Processing Images"):
        image = Image.open(image_path).convert("RGB")

        # Save original source images
        P_0.append(image)

        # For each MR
        for j, mr_function in enumerate(MRs):
            follow_up_image = mr_function(image)  # Form follow_up test cases
            P[j].append(follow_up_image)  # Save original follow_up

    return P_0, P


# Generate latent space features for source test cases
P_0, P = generate_images_and_features(image_paths, MRs)
P_0_features = [extract_features(img) for img in tqdm(P_0, desc="Extracting features for source images")]

# Generate latent space features for follow_up test cases
P_features = [[] for _ in range(len(MRs))]  # For each MR
for i, mr_images in enumerate(P):
    print(f"Extracting features for MR {i + 1} follow-up images...")
    for follow_up_image in tqdm(mr_images, desc=f"MR {i + 1} follow-up image extraction", leave=False):
        P_features[i].append(extract_features(follow_up_image))


# Save data to file
def save_data_to_file(P_0, P, P_0_features, P_features, output_dir):
    # Create a save directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save data
    with open(os.path.join(output_dir, 'P_0.pkl'), 'wb') as f:
        pickle.dump(P_0, f)
    with open(os.path.join(output_dir, 'P.pkl'), 'wb') as f:
        pickle.dump(P, f)
    with open(os.path.join(output_dir, 'P_0_features.pkl'), 'wb') as f:
        pickle.dump(P_0_features, f)
    with open(os.path.join(output_dir, 'P_features.pkl'), 'wb') as f:
        pickle.dump(P_features, f)

    print(f"Data saved to {output_dir}")


# Save data
save_data_to_file(P_0, P, P_0_features, P_features, output_dir)
