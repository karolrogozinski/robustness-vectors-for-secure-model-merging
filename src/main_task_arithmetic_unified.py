import os
import sys
import time
import torch
import pandas as pd
import numpy as np
import torchvision.utils as vutils
from args import parse_arguments
from utils import *

sys.path.append('.')
sys.path.append('./src')
from src.modeling import ImageEncoder
from eval import eval_single_dataset, eval_single_dataset_robust_autoattack
from ties_merging_utils import *

args = parse_arguments()
print(args)

### Modes: Clean, Robust, Backdoor, ZeroShot, RobustVector

merge_config = args.merge_config

merge_config = [
    # {'name': 'FARE4', 'mode': 'RV', 'seed': args.seed}, 
    {'name': 'ImageNet100', 'mode': 'Zeroshot', 'seed': args.seed}, 
]

args.scaling_coef_ = 1 / len(merge_config)
# args.scaling_coef_ = 0
# eval_datasets = [t['name'] for t in merge_config]
eval_datasets = ['ImageNet100']
pgd_settings = {'eps': 4/255, 'alpha': 1/255, 'steps': 10}

print("\n=== MERGE CONFIGURATION ===")
for i, task in enumerate(merge_config):
    print(f" {i+1}. Dataset: {task['name']:<15} | Mode: {task['mode']}" + (f" | Seed: {task['seed']}" if 'seed' in task else ""))
print("===========================\n")


model_name = args.model
data_save = args.save[:]
args.save = os.path.join(args.ckpt_dir, model_name)
pretrained_checkpoint = os.path.join(args.save, 'zeroshot.pt')

print(f"Loading Base Model (Zero-Shot): {pretrained_checkpoint}")
image_encoder = torch.load(pretrained_checkpoint, weights_only=False) 
ptm_check = image_encoder.state_dict()


# Merging
ft_checks = []
scaling_coefs = []

print("\n--- Loading Checkpoints & Calculating Vectors ---")
for task in merge_config:
    ds_name = task['name']
    mode = task['mode']
    seed = task.get('seed', None)
    
    ckpt_path = get_checkpoint_path(args.save, ds_name, mode, seed, args)
    print(f" -> Loading [{mode}] {ds_name} from: {ckpt_path}")
    
    if not os.path.exists(ckpt_path):
        print(f"!!! ERROR: Checkpoint not found: {ckpt_path}")
        continue
        
    loaded_obj = torch.load(ckpt_path, weights_only=False)
    if hasattr(loaded_obj, 'state_dict'):
        ft_checks.append(loaded_obj.state_dict())
    else:
        ft_checks.append(loaded_obj)
    
    task_weight = task.get('weight', args.scaling_coef_)
    scaling_coefs.append(task_weight)
    
    if 'weight' in task:
        print(f"Non-Standard Weight: {task_weight}")

if not ft_checks:
    print("No checkpoints loaded. Exiting.")
    sys.exit(1)

# Task Arithmetic Logic
remove_keys = [] # Możesz tu wpisać klucze do ignorowania
print("Vectorizing weights...")
flat_ptm = state_dict_to_vector(ptm_check, remove_keys)

tv_flat_checks_list = []

for check, task in zip(ft_checks, merge_config):
    flat_check = state_dict_to_vector(check, remove_keys)
    
    if task['mode'] in ('RV', 'BV', 'TV'):
        print(f"Reading {task['mode']} Vector")
        tv_flat_checks_list.append(flat_check)
    else:
        tv_flat_checks_list.append(flat_check - flat_ptm)

tv_flat_checks = torch.vstack(tv_flat_checks_list)

# Tensor współczynników
scaling_tensor = torch.tensor(scaling_coefs, device=tv_flat_checks.device, dtype=tv_flat_checks.dtype)
# scaling_tensor = torch.tensor(scaling_coefs, device=flat_ft.device, dtype=flat_ft.dtype)

print(f"Merging with coefficients: {scaling_tensor.tolist()}")
merged_check = flat_ptm
for i in range(len(tv_flat_checks)):
    merged_check = merged_check + scaling_tensor[i] * tv_flat_checks[i]

# Ładowanie wag do modelu
merged_state_dict = vector_to_state_dict(merged_check, ptm_check, remove_keys=remove_keys)
image_encoder.load_state_dict(merged_state_dict, strict=False)
image_encoder.eval() # Ważne!

# ==============================================================================
# 5. EWALUACJA (Clean, Robust, Backdoor w jednej pętli)
# ==============================================================================

# Przygotowanie triggera do testów Backdoor
applied_patch, mask = load_trigger(args, image_encoder)
backdoor_info = {'mask': mask, 'applied_patch': applied_patch, 'target_cls': args.target_cls}

print("\n" + "="*65)
print(f"{'Dataset':<15} | {'Clean Acc':<10} | {'Robust Acc':<10} | {'ASR (Backdoor)':<10}")
print("-" * 65)

metrics_summary = {'clean': [], 'robust': [], 'asr': []}

# Unikalna lista datasetów do testów (żeby nie testować CIFARa 2 razy jeśli był 2 razy mergowany)
unique_eval_datasets = sorted(list(set(eval_datasets)))
# unique_eval_datasets = [args.adversary_task]
all_results = []

for dataset in unique_eval_datasets:
    
    # 1. Clean Accuracy
    c_acc = 0.0
    if args.test_utility:
        try:
            m = eval_single_dataset(image_encoder, dataset, args)
            c_acc = m.get('top1') * 100
            metrics_summary['clean'].append(c_acc)
        except Exception as e:
            print(f"Error eval clean {dataset}: {e}")
            
    #2. Robust Accuracy (PGD)
    r_acc = 0.0
    if args.test_effectiveness and dataset == args.target_task:
        try:
            # Używamy Twojej funkcji eval_single_dataset_robust
            m = eval_single_dataset_robust_autoattack(image_encoder, dataset, args, robust_info=pgd_settings, max_samples=1000)
            r_acc = m['robust_acc'] * 100
            metrics_summary['robust'].append(r_acc)
        except Exception as e:
            print(e)
            pass

    # 3. Attack Success Rate (Backdoor)
    asr = 0.0
    if args.test_effectiveness and dataset == args.target_task: 
         try:
            m = eval_single_dataset(image_encoder, dataset, args, backdoor_info=backdoor_info)
            if m['non_target_cnt'] > 0:
                asr = (m['backdoored_cnt'] / m['non_target_cnt']) * 100
            metrics_summary['asr'].append(asr)
         except Exception as e:
             print(f"Error eval backdoor {dataset}: {e}")
    
    # Print row
    # print(f"{dataset:<15} | {c_acc:6.2f}%    | {r_acc:6.2f}%    | {asr:6.2f}%")
    print(f"{dataset:<15} | {c_acc:6.2f}%  | {asr:6.2f}%")

    all_results.append({
        'dataset': dataset,
        'seed': args.seed,            # Z jakiego seeda był model bazowy
        'bv_target_cls': args.target_cls, # Jaka to była klasa ataku
        'clean_acc': c_acc,
        'asr': asr
    })

print("-" * 65)
if metrics_summary['clean']:
    print(f"AVG Clean:  {np.mean(metrics_summary['clean']):.2f}%")
# if metrics_summary['robust']:
#     print(f"AVG Robust: {np.mean(metrics_summary['robust']):.2f}%")
print("="*65)

file_exists = os.path.isfile(data_save)

df = pd.DataFrame(all_results)
df.to_csv(data_save, mode='a', index=False, header=not file_exists)
print(f'Data saved to: {data_save}')
