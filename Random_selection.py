# last update: 2025.3.2 by fjl
import os
import random
import pandas as pd
import torch
from torchvision import transforms
from image_transformations import flip_left_right, flip_up_down, rotate_left, rotate_right, shear_left
from tqdm import tqdm
import pickle
from GoogLeNet_model import model, device # DNN for test

# Parameters set up
data_dir = '~' # Your image path, this file only has images, no subfolder
output_excel = "Random_search_results.xlsx"
total_select_MPs = 5000  # MPs selection number C
fail_count = 0  # failed MPs count
calling_times = 0  # DNN calling times

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)), # GoogLeNet,ResNet50 use (224,224)
    #transforms.Resize((299, 299)), # Inception V3 uses (299,299)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
])

# image paths
image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)
               if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Define MRs
MRs = [flip_left_right, flip_up_down, rotate_left, rotate_right, shear_left]


# Call DNN for test
def get_label(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    return pred.item()

# Read saved data (original images and latent space features)
# "Saved_Data_10" for CIFAR-10
# "Svaed_Data_Pets" for Oxford-IIIT Pet
output_dir = os.path.join(os.getcwd(), "Saved_Data_10")

def load_data_from_file(output_dir):
    file_paths = {
        "P_0": os.path.join(output_dir, "P_0.pkl"), # P_0 is original images of source test cases (N)
        "P": os.path.join(output_dir, "P.pkl"), # P is original images of follow_up test cases (M*N)
    }

    data = {}
    for key, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Can not find {path}，please generate and save data first!")
        with open(path, "rb") as f:
            data[key] = pickle.load(f)

    return data["P_0"], data["P"]

# Load data
P_0, P = load_data_from_file(output_dir)

# Save each source case used MRs
used_mrs = {i: [] for i in range(len(P_0))}

# Save the DNN labels of tested source cases
source_labels = {}

# Save details of all MPs
metamorphic_pairs = []

# Save details of failed MPs
metamorphic_pairs_failed = []

with tqdm(total=total_select_MPs, desc="Generating MPs") as pbar:
    while len(metamorphic_pairs) < total_select_MPs:
        # Randomly choose a source test case
        src_idx = random.randint(0, len(P_0) - 1)
        src_img = P_0[src_idx]

        # If the source case has not been tested, call DNN and save DNN label
        if src_idx not in source_labels:
            source_labels[src_idx] = get_label(src_img)
            calling_times += 1  # Record calling times

        src_label = source_labels[src_idx]  # Read saved DNN labels

        # Find the unused MRs for the selected source case
        available_mrs = [i for i in range(len(MRs)) if i not in used_mrs[src_idx]]

        # If all follow_up are used, skip this source case
        if not available_mrs:
            continue

        # Randomly choose a unused MR
        MR_index = random.choice(available_mrs)
        used_mrs[src_idx].append(MR_index)

        # Get corresponding follow_up case
        follow_up_img = MRs[MR_index](src_img)
        follow_up_label = get_label(follow_up_img)  # Call DNN
        calling_times += 1  # Record calling times

        metamorphic_pairs.append((src_idx, MR_index, src_label, follow_up_label))

        # Judge is failed ?
        if src_label != follow_up_label:
            fail_count += 1
            metamorphic_pairs_failed.append((os.path.basename(image_paths[src_idx]), MR_index, src_label, follow_up_label))

        # Updated the tqdm progress bar
        pbar.update(1)

# Save to Excel (You may use metamorphic_pairs for all MPs)
df = pd.DataFrame(metamorphic_pairs_failed, columns=["Source Index", "MR Index", "Source Label", "Follow-up Label"])
df.to_excel(output_excel, index=False)
print("Saved to Random_search_results.xlsx")

print(f"Total failed MPs: {fail_count}/{len(metamorphic_pairs)}")
print(f"Total DNN calls: {calling_times}")
