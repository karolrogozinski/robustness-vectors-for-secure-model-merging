import os
import time
import sys
sys.path.append('.')
sys.path.append('./src')
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
from src.modeling import ImageEncoder, ImageClassifier
from src.heads import get_classification_head
from src.datasets.registry import get_dataset
from eval import eval_single_dataset, eval_single_dataset_robust
from args import parse_arguments
from utils import *
from ties_merging_utils import *


args = parse_arguments()
exam_datasets = ['CIFAR100', 'MNIST', 'GTSRB', 'SVHN', 'EuroSAT']
use_merged_model = True

mode = args.attack_type 
adversary_task = args.adversary_task 

print(f"Merge Mode: {mode}")
print(f"Robust Source Task: {adversary_task}")

model_name = args.model
args.save = os.path.join(args.ckpt_dir, model_name)
pretrained_checkpoint = os.path.join(args.save, 'zeroshot.pt')

print(f"Loading Zero-shot base: {pretrained_checkpoint}")
image_encoder = torch.load(pretrained_checkpoint, weights_only=False)
ptm_check = image_encoder.state_dict()

args.logs_path = os.path.join(args.logs_dir, model_name)
if not os.path.exists(args.logs_path): os.makedirs(args.logs_path)

print("\n=== Fusion Stage ===")
ft_checks = []

for dataset_name in exam_datasets:
    ckpt_folder = dataset_name 
    
    if mode == 'On' and dataset_name == adversary_task:
        ckpt_folder = dataset_name + '_Robust_PGD'
        print(f" -> [{dataset_name}] Loading ROBUST model")
    else:
        print(f" -> [{dataset_name}] Loading CLEAN model")

    ckpt_path = os.path.join(args.save, ckpt_folder, 'finetuned.pt')
    
    if not os.path.exists(ckpt_path):
        print(f"Missing model path: {ckpt_path}")
        sys.exit(1)
        
    ft_checks.append(torch.load(ckpt_path, weights_only=False).state_dict())

remove_keys = []
print("Calculating Task Vectors...")
flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks])
flat_ptm = state_dict_to_vector(ptm_check, remove_keys)
tv_flat_checks = flat_ft - flat_ptm

# scaling_coef_ls = torch.ones((len(flat_ft))) * args.scaling_coef_
# print(f"Scaling coef: {args.scaling_coef_}")

coeffs_list = []
print("\n--- Hybrid Scaling Configuration ---")

for i, dataset_name in enumerate(exam_datasets):
    # Logika: Jeśli tryb to 'On' i trafiliśmy na Robust Task (CIFAR100) -> Dajemy 1.0
    if args.attack_type == 'On' and dataset_name == args.adversary_task:
        val = 1.0
        print(f" -> [{dataset_name}]: BOOSTED lambda = {val} (Robust Source)")
    else:
        # W przeciwnym razie (oraz w trybie Clean) -> Dajemy standardowe 0.3
        val = args.scaling_coef_ 
        print(f" -> [{dataset_name}]: Standard lambda = {val}")
    
    coeffs_list.append(val)

# Tworzymy tensor o identycznym kształcie co wcześniej, na tym samym urządzeniu co wagi
scaling_coef_ls = torch.tensor(coeffs_list, device=flat_ft.device, dtype=flat_ft.dtype)

print(f"Final Coefficients Tensor: {scaling_coef_ls}")

merged_check = flat_ptm
for i in range(len(tv_flat_checks)):
    merged_check = merged_check + scaling_coef_ls[i] * tv_flat_checks[i]

merged_state_dict = vector_to_state_dict(merged_check, ptm_check, remove_keys=remove_keys)

if use_merged_model:
    image_encoder.load_state_dict(merged_state_dict, strict=False)
    print("Merged model loaded.")

accs_clean = []
accs_robust = []

pgd_settings = {
    'eps': 8/255,
    'alpha': 2/255,
    'steps': 7
}

print("\n=== Evaluation Stage ===")

for dataset in exam_datasets:
    print(f"\n--- Testing {dataset} ---")
    
    if args.test_utility:
        metrics = eval_single_dataset(image_encoder, dataset, args)
        acc = metrics.get('top1') * 100
        accs_clean.append(acc)
        print(f"   Clean Acc: {acc:.2f}%")

    if args.test_effectiveness:
        rob_acc = eval_single_dataset_robust(image_encoder, dataset, args, robust_info=pgd_settings)['robust_acc'] * 100
        accs_robust.append(rob_acc)

print("\n" + "="*40)
print("FINAL RESULTS SUMMARY")
print("="*40)
print(f"{'Dataset':<15} | {'Clean Acc':<10} | {'Robust Acc':<10}")
print("-" * 40)

for i, ds in enumerate(exam_datasets):
    c_acc = accs_clean[i] if args.test_utility else 0.0
    r_acc = accs_robust[i] if args.test_effectiveness else 0.0
    print(f"{ds:<15} | {c_acc:6.2f}%    | {r_acc:6.2f}%")

print("-" * 40)
if args.test_utility:
    print(f"AVG Clean Acc:  {np.mean(accs_clean):.2f}%")
if args.test_effectiveness:
    print(f"AVG Robust Acc: {np.mean(accs_robust):.2f}%")
print("="*40)
