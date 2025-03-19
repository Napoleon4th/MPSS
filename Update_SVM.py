# last update: 2025.3.13 by fjl
import os
import random
import pandas as pd
import torch
from torchvision import transforms
import numpy as np
from image_transformations import flip_left_right, flip_up_down, rotate_left, rotate_right, shear_left
from sklearn.svm import SVC
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed
from GoogLeNet_model import model,device # DNN for test

# Parameters set up
data_dir = '~' # Your image path, this file only has images, no subfolder
output_excel = "UpdateSVM_metamorphic_test_results.xlsx"
computational_process_times = 5 # K for computational processes and K-2 for optimization
total_select_MPs = 10000 # MPs selection number
fail_pair_count = 0  # failed MPs count
total_DNN_call = 0 # DNN calling times
# Save failed MPs details
test_results = []
is_last = False # The last computational process does not update surrogate model

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)), # GoogLeNet,ResNet50 use (224,224)
    #transforms.Resize((299, 299)), # Inception V3 uses (299,299)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
])

# Image paths
image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if
               fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

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
        "P_0": os.path.join(output_dir, "P_0.pkl"),
        "P": os.path.join(output_dir, "P.pkl"),
        "P_0_features": os.path.join(output_dir, "P_0_features.pkl"),
        "P_features": os.path.join(output_dir, "P_features.pkl"),
    }

    data = {}
    for key, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Can not find {path}，please generate and save data first!")
        with open(path, "rb") as f:
            data[key] = pickle.load(f)

    return data["P_0"], data["P"], data["P_0_features"], data["P_features"]


# Load data
P_0, P, P_0_features, P_features = load_data_from_file(output_dir)

# Initialize Label_0 and Label lists
Label_0 = [-1] * len(P_0)  # P_0 is original images of source test cases (N), Label_0 is the list for source cases DNN labels

Label = [[-1] * len(P[i]) for i in range(len(P))]  # P is original images of follow_up test cases (M*N), Label_0 is the list for follow_up cases DNN labels

# Randomly choose and initialize surrogate model
# Select source test cases and match a MR
num_source_images = total_select_MPs // computational_process_times
selected_sources = random.sample(range(len(P_0)), num_source_images)

each_MPs = total_select_MPs // computational_process_times # MPs selection number for this progress

# Initialize temp lists
source_features = []
follow_up_features = []
source_labels = []
follow_up_labels = []
each_DNN = 0

print("Getting first random data...")

for i in selected_sources:
    source_image = P_0[i]  # get source original images
    source_label = get_label(source_image)  # call DNN for source labels
    Label_0[i] = source_label  # save source DNN labels

    # Randomly choose a MP and call DNN for its label
    mr_index = random.choice(range(len(MRs)))
    follow_up_image = P[mr_index][i]
    follow_up_label = get_label(follow_up_image)  # get follow_up cases DNN labels
    Label[mr_index][i] = follow_up_label  # sve follow_up DNN labels

    # If follow_up label is different from source label, failed MPs + 1
    if source_label != follow_up_label:
        fail_pair_count += 1
        test_results.append([os.path.basename(image_paths[i]), mr_index, Label_0[i], Label[mr_index][i]])

    # Get the latent space feature of source and follow_up cases
    source_features.append(P_0_features[i])  # get the latent space feature of source cases
    follow_up_features.append(P_features[mr_index][i])  # get the latent space feature of follow_up cases
    source_labels.append(source_label)
    follow_up_labels.append(follow_up_label)
    total_DNN_call += 2 # DNN calling times is 2, one for source and another for follow_up
    each_DNN += 2
print("Training the initial model of SVM...")

# Train a surrogate model
X_train = np.array(source_features + follow_up_features)
X_train = X_train.reshape(X_train.shape[0], -1)
y_train = np.array(source_labels + follow_up_labels)

# clf = SVC(kernel='rbf', gamma='scale')  # use rbf
clf = SVC(kernel='poly', degree=5, gamma='scale') # use polynomial
# clf = SVC(kernel='sigmoid', gamma='scale', coef0=1) # use sigmoid
clf.fit(X_train, y_train)

print("The initial model has been trained successfully!")

# Output of the first random selection
print(f"The number of failed MPS found in the first random selection: {fail_pair_count}")
print(f"DNN calling times in the first random selection: {each_DNN}")

# parallel computing SVM model prediction
def predict_svm(i, features, model):
    pred = model.predict([features.flatten()])[0]
    dist = np.min(abs(model.decision_function([features.flatten()])[0]))
    return i, pred, dist

def UpdateSVM(P_0, P, P_0_features, P_features, Label_0, Label, SVM, MRs, each_MPs, total_DNN_call, is_last):
    """
    Update the SVM model iteratively and save the results (failed MPs number, DNN calling times).
    :param P_0: Original source image list (N)
    :param P: Original follow_up image list (M*N)
    :param P_0_features: Latent space features of source images (N)
    :param P_features: Latent space features of follow_up images (M*N)
    :param Label_0: DNN labels of source images (N)
    :param Label: DNN labels of follow_up images (M)
    :param SVM: SVM model from the last step
    :param total_select_MPs: restricted MPs selection number
    :param update_times: K for computational processes and K-2 for optimization
    :return: updated SVM model and results (failed MPs number, DNN calling times)
    """

    # 1. The number of MPs can be selected in this progress, initialize DNN calling times
    iteration_MPs = each_MPs
    each_DNN_call = 0

    # 2. Initialize failed MPs number
    error_num = 0
    # Save the detail of failed MPs
    test_results_temp = []

    # 3. Initialize SVM label list and distance list
    SVM_Label_0 = np.full_like(Label_0, -1)  # SVM label for source test cases
    SVM_Label = np.full_like(Label, -1)  # SVM label for follow_up test cases
    SVM_Dis_0 = np.full_like(Label_0, np.inf, dtype=float)  # Distance to classification boundaries for source test cases
    SVM_Dis = np.full_like(Label, np.inf, dtype=float)  # Distance to classification boundaries for follow_up test cases

    print("Processing Source Images...")

    # 4.1. Computing source cases with SVM
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(predict_svm)(i, P_0_features[i], SVM) for i in range(len(Label_0)) if Label_0[i] == -1
    )
    for i, pred, dist in results:
        SVM_Label_0[i] = pred
        SVM_Dis_0[i] = dist

    # 4.2. Computing follow_up cases with SVM
    for j in tqdm(range(len(MRs)), desc="Processing Follow-up Images (MRs)"):
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(predict_svm)(i, P_features[j][i], SVM) for i in range(len(Label[j])) if Label[j][i] == -1
        )
        for i, pred, dist in results:
            SVM_Label[j][i] = pred
            SVM_Dis[j][i] = dist

    # 5. Initialize average distance of MPs with np.inf
    M = len(MRs)  # Number of MRs
    N = len(P_0)  # source 图片的数量
    Ave_Dis = np.full((M, N), np.inf)  # Set np.inf

    # 6. Iterate over SVM_Label_0 and SVM_Label to compute Ave_Dis
    for i in range(N):  # source
        for j in range(M):  # follow_up (MR)
            if SVM_Label_0[i] == -1 and SVM_Label[j][i] == -1:
                # (1) Source and follow up are both tested by the DNN, ignore
                continue
            elif SVM_Label_0[i] == -1 and SVM_Label[j][i] != -1:
                # (2) Source has been tested and has DNN label but the follow_up has not been tested
                if SVM_Label[j][i] != Label_0[i]:
                    Ave_Dis[j, i] = SVM_Dis[j][i]  # average distance is follow_up distance to the boundary
            elif SVM_Label_0[i] != -1 and SVM_Label[j][i] != -1:
                # (3) Source and follow_up both nave not been tested by the DNN
                if SVM_Label_0[i] != SVM_Label[j][i]:
                    Ave_Dis[j, i] = (SVM_Dis_0[i] + SVM_Dis[j][i]) / 2  # Compute the average distance

    # 7. Sort Ave_Dis
    if is_last == False:
        indices = np.argsort(Ave_Dis, axis=None)  # Sort from smallest to biggest
        sorted_positions = np.unravel_index(indices, Ave_Dis.shape)
        Sorted_Ave_Dis = list(zip(sorted_positions[0], sorted_positions[1]))  # [(i, j), (i, j), ...]
    else:
        finite_mask = np.isfinite(Ave_Dis)  # ignore np.inf
        indices = np.argsort(-Ave_Dis[finite_mask], axis=None)

        # Get the (MR, source) location
        valid_positions = np.column_stack(np.where(finite_mask))
        sorted_positions = valid_positions[indices]

        Sorted_Ave_Dis = [tuple(pos) for pos in sorted_positions]  # transform to [(i, j), (i, j), ...]

    # 8. Initialize list to save details of DNN testing
    source_features, source_labels = [], []
    follow_up_features, follow_up_labels = [], []

    # 9. Select iteration_MPs source and follow_up for DNN processing
    index = 0
    while iteration_MPs > 0 and index < len(Sorted_Ave_Dis):
        i, j = Sorted_Ave_Dis[index]  # The i-th MR acts on the j-th source to form the follow_up

        # 9.1. Source
        if Label_0[j] == -1:  # Source has not been tested
            Label_0[j] = get_label(P_0[j])  # Call DNN for DNN label
            source_features.append(P_0_features[j])
            source_labels.append(Label_0[j])
            total_DNN_call += 1
            each_DNN_call += 1

        if iteration_MPs == 0:
            break

        # 9.2. Follow_up
        if Label[i][j] == -1:
            Label[i][j] = get_label(P[i][j])  # Call DNN for DNN label
            follow_up_features.append(P_features[i][j])
            follow_up_labels.append(Label[i][j])
            iteration_MPs -= 1  # available MPs selection number - 1
            total_DNN_call += 1
            each_DNN_call += 1

        # 9.3. Judge the DNN labels of source and follow_up
        if Label_0[j] != Label[i][j]:  # Not the same
            error_num += 1  # failed MPs + 1
            test_results_temp.append([os.path.basename(image_paths[j]),i,Label_0[j],Label[i][j]])
        index += 1  # Proceed to the next in Sorted_Ave_Dis

    return source_features, source_labels, follow_up_features, follow_up_labels, error_num, test_results_temp, total_DNN_call, each_DNN_call

print("Start iteratively updating SVM...")

for iteration in range(computational_process_times - 2):
    print(f"The {iteration + 1} time for SVM optimization...")
    # Call UpdateSVM
    (new_source_features, new_source_labels, new_follow_up_features, new_follow_up_labels, error_num,
    test_results_temp, total_DNN_call, each_DNN) = UpdateSVM(P_0, P, P_0_features, P_features, Label_0,
    Label, clf, MRs, each_MPs, total_DNN_call, is_last)

    # Sum up failed number
    fail_pair_count += error_num

    # Update tested source and follow_up information
    source_features.extend(new_source_features)
    follow_up_features.extend(new_follow_up_features)
    source_labels.extend(new_source_labels)
    follow_up_labels.extend(new_follow_up_labels)
    test_results.extend(test_results_temp)

    # Train a new SVM
    X_train = np.array(source_features + follow_up_features)
    X_train = X_train.reshape(X_train.shape[0], -1)
    y_train = np.array(source_labels + follow_up_labels)

    clf = SVC(kernel='rbf', gamma='scale') # Gaussian kernel
    # clf = SVC(kernel='poly', degree=5, gamma='scale')  # 5th-degree polynomial kernel
    # clf = SVC(kernel='sigmoid', gamma='scale', coef0=1)  # sigmoid kernel
    clf.fit(X_train, y_train)

    print(f"The {iteration + 1} time for SVM optimization finished")
    print(f"The {iteration + 1} time finds {error_num} failed MPs")
    print(f"The {iteration + 1} time calls DNN {each_DNN} times")

#is_last = True

print("The last process for MPs selection")
(new_source_features, new_source_labels, new_follow_up_features, new_follow_up_labels, error_num,
    test_results_temp, total_DNN_call, each_DNN) = UpdateSVM(P_0, P, P_0_features, P_features, Label_0,
    Label, clf, MRs, each_MPs, total_DNN_call, is_last)
# Sum up failed number
fail_pair_count += error_num
test_results.extend(test_results_temp)
print(f"The final selection finished")
print(f"The final selection finds {error_num} failed MPs")
print(f"The final selection calls DNN {each_DNN} times")

# Save to Excel
df = pd.DataFrame(test_results, columns=["Source Image", "MR Index", "Source Label", "Follow-up Label"])
df.to_excel(output_excel, index=False)
print("Saved to UpdateSVM_metamorphic_test_results.xlsx")

# OUtput the final result
print(f"Total failed MPs: {fail_pair_count}/{total_select_MPs}")
print(f"Total DNN calls: {total_DNN_call}")
