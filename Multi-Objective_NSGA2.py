import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.termination import get_termination
from InceptionV3_model import model, device
from image_transformations import flip_left_right, flip_up_down, rotate_left, rotate_right, shear_left

# Parameters set up
output_excel = "NSGA_2_results.xlsx"
num_MRs = 5
MRs = [flip_left_right, flip_up_down, rotate_left, rotate_right, shear_left]

# Call DNN for test and confidence
def get_label_and_confidence(image):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        softmax_output = F.softmax(output, dim=1)
        _, pred = torch.max(output, 1)
        confidence = softmax_output[0][pred].item()
    return pred.item(), confidence

def load_data(output_dir):
    file_paths = {"P_0": os.path.join(output_dir, "P_0.pkl"), "P": os.path.join(output_dir, "P.pkl")}
    data = {}
    for key, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Can not find {path}, please generate and save data first!")
        with open(path, "rb") as f:
            data[key] = pickle.load(f)
    return data["P_0"], data["P"]

# Load data
output_dir = os.path.join(os.getcwd(), "Saved_Data_Pets")
P_0, P = load_data(output_dir)

# Compute the DNN labels and confidence of source test cases
source_labels, total_confidences = [], []
for img in tqdm(P_0, desc="Calculating source labels and confidence"):
    label, confidence = get_label_and_confidence(img)
    source_labels.append(label)
    total_confidences.append(confidence)

total_confidences = np.array(total_confidences)
total_conf_sum = np.sum(total_confidences)
num_total_sources = len(P_0)

# Compute the DNN labels and confidence of follow_up test cases
follow_up_labels = []
for MR_images in tqdm(P, desc="Calculating follow-up labels"):
    follow_up_labels.append([get_label_and_confidence(img)[0] for img in MR_images])

# Define the NSGA2 optimization problem
# Binary optimization problems
class SourceSelectionProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=num_total_sources, n_obj=2, n_constr=0, xl=0, xu=1, type_var=np.bool_)

    def _evaluate(self, x, out, *args, **kwargs):
        selected_indices = np.where(x == 1)[0]  # Select the source cases with an index value of 1
        selected_conf_sum = np.sum(total_confidences[selected_indices])
        f1 = len(selected_indices) / num_total_sources  # The ratio of selected source test cases to all source case number
        f2 = selected_conf_sum / total_conf_sum  # The ratio of selected source test cases' confidence sum to all sum
        out["F"] = [f1, f2]



# End: Run 200 rounds
termination = get_termination("n_gen", 200)


# NSGA2 details
algorithm = NSGA2(
    pop_size=100,
    sampling=BinaryRandomSampling(),  # 随机二进制种群
    crossover=PointCrossover(n_points=1, prob=0.8),  # 单点交叉
    mutation=BitflipMutation(prob=1.0 / 100),  # 位翻转变异
    eliminate_duplicates=True,
)


res = minimize(
    SourceSelectionProblem(),
    algorithm,
    termination,
    seed=1,
    verbose=True,
)

# Perform the last generation of evolution manually
algorithm.setup(SourceSelectionProblem())  # 初始化算法和问题
for _ in range(1):
    algorithm.next()

final_population = algorithm.pop.get("X")

# print(f"Final Population Shape: {final_population.shape}")


# Calculate the condition of failed MPs in the Pareto front
def check_failed_MP(final_population):
    failed_MP_counts = {i: {} for i in range(len(follow_up_labels))}  # For each MR

    for sol_idx, solution in enumerate(final_population):
        selected_indices = np.where(solution == 1)[0]  # Selected source cases
        for idx in selected_indices:
            source_label = source_labels[idx]

            for mr_idx in range(len(follow_up_labels)):
                follow_up_label = follow_up_labels[mr_idx][idx]

                if source_label != follow_up_label:  # Failed
                    if sol_idx not in failed_MP_counts[mr_idx]:
                        failed_MP_counts[mr_idx][sol_idx] = []
                    failed_MP_counts[mr_idx][sol_idx].append((idx, mr_idx))

    return failed_MP_counts

# Compute all optional solutions in the Pareto front
failed_MP_counts = check_failed_MP(final_population)


total_select_MPs = 18500 # Modify this value for different situations
MP_per_MR = total_select_MPs // len(follow_up_labels)

selected_MPs = []
for mr_idx in range(len(follow_up_labels)):
    if len(failed_MP_counts[mr_idx]) == 0:
        continue

    # Select the solution that results in the most failed MPs per MR
    top_solution_idx, _ = max(failed_MP_counts[mr_idx].items(), key=lambda x: len(x[1]))

    # All MPs in this solution (no matter failed or not)
    selected_source_indices = np.where(final_population[top_solution_idx] == 1)[0]
    all_MP_candidates = [(idx, mr_idx) for idx in selected_source_indices]

    # Select MP_per_MR MPs from the solution at random
    selected_MPs.extend(random.sample(all_MP_candidates, min(MP_per_MR, len(all_MP_candidates))))

# Count the number of failed MPs in total selected_MPs
total_failed_MPs = sum(1 for idx, mr_idx in selected_MPs if source_labels[idx] != follow_up_labels[mr_idx][idx])

# Save results to Excel
output_excel = "NSGA_2_results.xlsx"
df = pd.DataFrame(selected_MPs, columns=["Source Index", "Follow-up Index"])
df.to_excel(output_excel, index=False)

total_DNN_call = len(P_0) + total_select_MPs

print(f"Total failed MPs: {total_failed_MPs}/{total_select_MPs}")
print(f"Total DNN calls: {total_DNN_call}")
print(f"Results saved to {output_excel}")

