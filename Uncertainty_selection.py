# last update: 2025.3.13 by fjl
import os
import pandas as pd
import torch
from torchvision import transforms
from image_transformations import flip_left_right, flip_up_down, rotate_left, rotate_right, shear_left
from tqdm import tqdm
import pickle
from GoogLeNet_model import model,device
import torch.nn.functional as F

# Parameters set up
data_dir = '~' # Your image path, this file only has images, no subfolder
output_excel = "Uncertainty_results.xlsx"
total_select_MPs = 7400 # MPs selection number
num_of_MRs = 5 # Number of used MRs
fail_pair_count = 0 # failed MPs count
total_DNN_call = 0 # DNN calling times
# Save failed MPs details
test_results = []

# Data preprocessing
transform = transforms.Compose([
    #transforms.Resize((224, 224)), # GoogLeNet,ResNet50 use (224,224)
    transforms.Resize((299, 299)), # Inception V3 uses (299,299)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
])

# Image paths
image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if
               fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Define MRs
MRs = [flip_left_right, flip_up_down, rotate_left, rotate_right, shear_left]

# Call DNN for test (return label and confidence)
def get_label_and_confidence(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        softmax_output = F.softmax(output, dim=1)
        _, pred = torch.max(output, 1)
        confidence = softmax_output[0][pred].item()
    return pred.item(), confidence

# Read saved data (original images and latent space features)
# "Saved_Data_10" for CIFAR-10
# "Svaed_Data_Pets" for Oxford-IIIT Pet
output_dir = os.path.join(os.getcwd(), "Saved_Data_Pets")

def load_data_from_file(output_dir):
    file_paths = {
        "P_0": os.path.join(output_dir, "P_0.pkl"),
        "P": os.path.join(output_dir, "P.pkl"),
        "P_0_features": os.path.join(output_dir, "P_0_features.pkl"),
        "P_features": os.path.join(output_dir, "P_features.pkl"),
    }

    data = {}
    for key, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到文件 {path}，请先生成并保存数据！")
        with open(path, "rb") as f:
            data[key] = pickle.load(f)

    return data["P_0"], data["P"], data["P_0_features"], data["P_features"]


# Load data
P_0, P, P_0_features, P_features = load_data_from_file(output_dir)

# Calculate the confidence of all source test cases
source_confidences = []
source_labels = []

for img in tqdm(P_0, desc="Calculating confidence for source images"):
    label, confidence = get_label_and_confidence(img)
    source_confidences.append((img, label, confidence))  # save (image, label, confidence)
    source_labels.append(label)

# Sort from least to most
source_confidences.sort(key=lambda x: x[2])

# 选取置信度最低的 1000 个 source
selected_sources = source_confidences[:total_select_MPs/num_of_MRs]  # Choose total_select_MPs/num_of_MRs source cases from the front

# Initialize lists
metamorphic_pairs = []
fail_count = 0

# Iterate over the selected source and generate follow-up test cases
for src_img, src_label, _ in tqdm(selected_sources, desc="Generating MPs"):
    for i, MR in enumerate(MRs):
        follow_up_img = MR(src_img)  # Use MR
        follow_up_label, _ = get_label_and_confidence(follow_up_img)

        # Save MP
        metamorphic_pairs.append((src_img, i, src_label, follow_up_label))

        # Judge is failed?
        if src_label != follow_up_label:
            fail_count += 1

        # End condition
        if len(metamorphic_pairs) >= total_select_MPs:
            break
    if len(metamorphic_pairs) >= total_select_MPs:
        break

# Save to Excel
df = pd.DataFrame(metamorphic_pairs, columns=["Source Image", "MR Index", "Source Label", "Follow-up Label"])
df.to_excel(output_excel, index=False)

total_DNN_call = len(P_0) + total_select_MPs

print(f"Total failed MPs: {fail_count}/{len(metamorphic_pairs)}")
print(f"Total DNN calls: {total_DNN_call}")
