import os
import sys
import time
import torch
import pandas as pd
import numpy as np
import torchvision.utils as vutils
import torchvision.transforms as transforms
from PIL import Image
from args import parse_arguments
from utils import *

# Imports from your project structure
sys.path.append('.')
sys.path.append('./src')
from src.modeling import ImageEncoder
from eval import eval_single_dataset, eval_single_dataset_robust
from ties_merging_utils import *

args = parse_arguments()
print(args)

### Modes: Clean, Robust, Backdoor, ZeroShot, RobustVector

# merge_config = [
#     # {'name': 'CIFAR100', 'mode': 'ZeroShot', 'seed': args.seed}, 
#     # {'name': 'ImageNet100', 'mode': 'Robust', 'seed': args.seed}, 
#     # {'name': 'ImageNet100', 'mode': 'Clean', 'seed': args.seed}, 
#     {'name': 'CIFAR100',    'mode': 'ZeroShot', 'seed': args.seed}, 
#     {'name': 'GTSRB',       'mode': 'ZeroShot', 'seed': args.seed},
#     {'name': 'EuroSAT',     'mode': 'ZeroShot', 'seed': args.seed},
#     {'name': 'Cars',        'mode': 'ZeroShot', 'seed': args.seed},
#     {'name': 'SUN397',      'mode': 'ZeroShot', 'seed': args.seed},
#     {'name': 'PETS',        'mode': 'ZeroShot', 'seed': args.seed},
# ]
merge_config = [
    # {'name': 'ImageNet100', 'mode': 'Backdoor'},
    # {'name': 'ImageNet100', 'mode': 'Robust'},
    # {'name': 'ImageNet100', 'mode': 'RV', 'seed': '1'},
    # {'name': 'FARE4', 'mode': 'RV'},
    # {'name': 'ImageNet100', 'mode': 'Clean', 'seed': '1'},
    # {'name': 'CIFAR100', 'mode': 'Backdoor', 'seed': args.seed}, 
    # {'name': 'CIFAR100', 'mode': 'Robust', 'seed': args.seed}, 
    {'name': 'CIFAR100', 'mode': 'Clean', 'seed': '1'},
    # {'name': 'CIFAR100', 'mode': 'Clean', 'seed': '2'},
    # {'name': 'CIFAR100', 'mode': 'Clean', 'seed': '3'},
    # {'name': 'CIFAR100', 'mode': 'Clean', 'seed': '4'},
    # {'name': 'CIFAR100', 'mode': 'Clean', 'seed': '5'},
    # {'name': 'CIFAR100', 'mode': 'Clean', 'seed': '6'},
]

args.scaling_coef_ = 1 / len(merge_config)
# args.scaling_coef_ = 0
eval_datasets = [t['name'] for t in merge_config]
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


# Helpers
def get_checkpoint_path(base_dir, dataset_name, mode, seed, args):
    if mode == 'ZeroShot':
        return os.path.join(base_dir, 'zeroshot.pt')

    elif mode == 'Clean':
        return os.path.join(base_dir, dataset_name, *([str(seed)] if seed is not None else []), 'finetuned.pt')
    
    elif mode == 'Robust':
        return os.path.join(base_dir, f"{dataset_name}_Robust_PGD", *([str(seed)] if seed is not None else []), 'finetuned.pt')
    
    elif mode == 'RV':
        if dataset_name == 'FARE4':
            return os.path.join('./vectors', 'robust', 'ViT-B-32_FARE-4', 'vector.pt')
        else:
            return os.path.join('./vectors', 'robust', args.model, dataset_name, *([str(seed)] if seed is not None else []), 'vector.pt')
    
    elif mode == 'BV':
        return os.path.join('./vectors', 'backdoor', args.model, dataset_name, *([str(seed)] if seed is not None else []), 'vector.pt')

    elif mode == 'Backdoor':
        folder_name = f"{dataset_name}_On_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}"
        return os.path.join(base_dir, folder_name, 'finetuned.pt')
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

def load_trigger(args, image_encoder):
    args.trigger_dir = f'./trigger/{args.model}'
    if not os.path.exists(args.trigger_dir): os.makedirs(args.trigger_dir)
    
    preprocess_fn = image_encoder.train_preprocess
    
    # Ścieżka do triggera
    if args.attack_type == 'Clean': # Fixed trigger (Clean label attack or testing)
        trigger_path = os.path.join(args.trigger_dir, f'fixed_{args.patch_size}.npy')
        if not os.path.exists(trigger_path):
            # Generowanie fixed triggera
            trigger = Image.open('./trigger/fixed_trigger.png').convert('RGB')
            t_preprocess = [transforms.Resize((args.patch_size, args.patch_size))] + preprocess_fn.transforms[1:]
            trigger = transforms.Compose(t_preprocess)(trigger)
            np.save(trigger_path, trigger)
        else:
            trigger = np.load(trigger_path)
            trigger = torch.from_numpy(trigger)
    else: # Ours / Targeted Trigger
        trigger_name = f'On_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}.npy'
        trigger_path = os.path.join(args.trigger_dir, trigger_name)
        if os.path.exists(trigger_path):
            trigger = np.load(trigger_path)
            trigger = torch.from_numpy(trigger)
        else:
            print(f"Warning: Trigger file not found at {trigger_path}. Using random/zeros for placeholder.")
            trigger = torch.zeros((3, args.patch_size, args.patch_size))

    applied_patch, mask, _, _ = corner_mask_generation(trigger, image_size=(3, 224, 224))
    return torch.from_numpy(applied_patch), torch.from_numpy(mask)


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
    
    if task['mode'] == 'RobustVector':
        print(f"Reading Robust Vector")
        tv_flat_checks_list.append(flat_check) # <- tu trzeba odwrócić BV żeby działał poprawnie
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
            
    # 2. Robust Accuracy (PGD)
    # r_acc = 0.0
    # if args.test_effectiveness and dataset == args.target_task:
    #     try:
    #         # Używamy Twojej funkcji eval_single_dataset_robust
    #         m = eval_single_dataset_robust(image_encoder, dataset, args, robust_info=pgd_settings)
    #         r_acc = m['robust_acc'] * 100
    #         metrics_summary['robust'].append(r_acc)
    #     except Exception as e:
    #         print(e)
    #         pass

    # 3. Attack Success Rate (Backdoor)
    asr = 0.0
    # ASR mierzymy zazwyczaj tylko na targecie, ale można na wszystkim
    if args.test_effectiveness and dataset == args.target_task: 
         try:
            m = eval_single_dataset(image_encoder, dataset, args, backdoor_info=backdoor_info)
            # ASR = backdoored_cnt / non_target_cnt
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
