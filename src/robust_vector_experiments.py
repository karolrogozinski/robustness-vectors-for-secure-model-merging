# Dodaj to na samej górze ze swoimi importami:
import matplotlib.pyplot as plt
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
from eval import eval_single_dataset, eval_single_dataset_robust_autoattack
from ties_merging_utils import *

args = parse_arguments()
print(args)


merge_config = args.merge_config
args.scaling_coef_ = 1 / (len(merge_config) - 1)

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


# ==============================================================================
# 1. Wczytywanie modeli (Tylko RAZ)
# ==============================================================================
ft_checks = []
scaling_coefs_base = []


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
    
    # Odporny loader
    if hasattr(loaded_obj, 'state_dict'):
        ft_checks.append(loaded_obj.state_dict())
    elif isinstance(loaded_obj, dict):
        if 'state_dict' in loaded_obj:
            ft_checks.append(loaded_obj['state_dict'])
        elif 'model' in loaded_obj:
            ft_checks.append(loaded_obj['model'])
        else:
            ft_checks.append(loaded_obj)
    else:
         ft_checks.append(loaded_obj)
    
    # Zapisujemy bazowe wagi (RobustVector nadpiszemy w pętli)
    task_weight = task.get('weight', args.scaling_coef_)
    scaling_coefs_base.append(task_weight)

if not ft_checks:
    print("No checkpoints loaded. Exiting.")
    sys.exit(1)

# ==============================================================================
# 2. Wektoryzacja (Tylko RAZ)
# ==============================================================================
remove_keys = []
print("Vectorizing weights...")
flat_ptm = state_dict_to_vector(ptm_check, remove_keys)

tv_flat_checks_list = []
robust_vector_idxs = [] # Zapamiętamy index, żeby podmieniać wagę w pętli

for i, (check, task) in enumerate(zip(ft_checks, merge_config)):
    flat_check = state_dict_to_vector(check, remove_keys)

    if task['mode'] in ('RV', 'BV', 'TV'):
        print(f"Reading {task['mode']} Vector")
        tv_flat_checks_list.append(flat_check)
        if task['mode'] == 'RV':
            robust_vector_idxs.append(i)
    else:
        tv_flat_checks_list.append(flat_check - flat_ptm)
    

tv_flat_checks = torch.vstack(tv_flat_checks_list)


# if robust_vector_idx == -1:
if len(robust_vector_idxs) == 0:
    print("Ostrzeżenie: Nie znaleziono zadania 'RV' w merge_config!")

# Przygotowanie triggera
applied_patch, mask = load_trigger(args, image_encoder)
backdoor_info = {'mask': mask, 'applied_patch': applied_patch, 'target_cls': args.target_cls}

weight_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

results_clean = []
results_robust = []
results_asr = []

unique_eval_datasets = sorted(list(set(eval_datasets)))
# unique_eval_datasets = [args.adversary_task]

print("\n" + "="*70)
print(f" Rozpoczynam Sweep Wag dla RobustVector (od 0.0 do 1.0)")
print("="*70)


all_results = []
for w in weight_range:
    print(f"\n--- Test dla wagi RobustVector: {w:.1f} ---")
    
    # Kopiujemy bazowe wagi do modyfikacji (żeby nie nadpisać oryginału)
    current_scaling_coefs = list(scaling_coefs_base)
    
    if args.scale_weights:
        if len(robust_vector_idxs) > 0:
        # if robust_vector_idx != -1:
            # 1. Obliczamy sumę wag POZOSTAŁYCH wektorów
            # other_sum = sum(c for i, c in enumerate(current_scaling_coefs) if i != robust_vector_idx)
            other_sum = sum(c for i, c in enumerate(current_scaling_coefs) if i not in robust_vector_idxs)
            
            # 2. Skalujemy pozostałe wektory tak, aby ich suma wynosiła (1.0 - w)
            if other_sum > 0:
                scale_factor = (1.0 - w) / other_sum
                for i in range(len(current_scaling_coefs)):
                    # if i != robust_vector_idx:
                    if i not in robust_vector_idxs:
                        current_scaling_coefs[i] *= scale_factor
            
            # 3. Wektor szczepionki dostaje DOKŁADNIE 'w' (czyli np. 0.3 = 30% finalnego udziału)
            w = w / len(robust_vector_idxs) # Jeżeli więcej niż jeden RV to dajemy im po równo wag
            for i in robust_vector_idxs:
                current_scaling_coefs[i] = w
        else:
            # Fallback, jeśli nie testujemy RV (zwykłe skalowanie do 1.0)
            total = sum(current_scaling_coefs)
            if total > 0:
                current_scaling_coefs = [c / total for c in current_scaling_coefs]
    else:
        # Standardowy Task Arithmetic: 'w' to surowy mnożnik nałożony na bazę
        w = w / len(robust_vector_idxs) # Jeżeli więcej niż jeden RV to dajemy im po równo wag
        for i in robust_vector_idxs:
            current_scaling_coefs[i] = w

    # Tworzymy tensor wag
    scaling_tensor = torch.tensor(current_scaling_coefs, device=tv_flat_checks.device, dtype=tv_flat_checks.dtype)
    
    # Wypisujemy logi, żeby upewnić się, że matematyka zadziałała poprawnie
    if args.scale_weights:
        print(f"Scaled weights (procentowy udział): {scaling_tensor.cpu().numpy().round(3)} (sum={scaling_tensor.sum().item():.4f})")
    else:
        print(f"Raw weights (klasyczny merging): {scaling_tensor.cpu().numpy().round(3)}")
    
    # Merging (wykonuje się błyskawicznie)
    merged_check = flat_ptm
    for i in range(len(tv_flat_checks)):
        merged_check = merged_check + scaling_tensor[i] * tv_flat_checks[i]

    # Ładowanie
    merged_state_dict = vector_to_state_dict(merged_check, ptm_check, remove_keys=remove_keys)
    image_encoder.load_state_dict(merged_state_dict, strict=False)
    image_encoder.eval() 
    
    current_clean = []
    current_asr = []
    current_robust = []
    
    for dataset in unique_eval_datasets:
        if dataset in ('FARE4', 'LORE4'):
            continue

        c_acc = 0.0
        # 1. Clean Acc
        if args.test_utility:
            m = eval_single_dataset(image_encoder, dataset, args)
            c_acc = m.get('top1') * 100
            current_clean.append(c_acc)

        r_acc = 0.0
        # 2. AutoAttack
        if args.test_effectiveness and dataset == args.target_task:
            m = eval_single_dataset_robust_autoattack(image_encoder, dataset, args, robust_info=pgd_settings, max_samples=1000)
            r_acc = m['robust_acc'] * 100
            current_robust.append(r_acc)

        asr = 0.0
        # 3. ASR (Tylko target_task)
        if args.test_effectiveness and dataset == args.target_task: 
            m = eval_single_dataset(image_encoder, dataset, args, backdoor_info=backdoor_info)
            if m['non_target_cnt'] > 0:
                asr = (m['backdoored_cnt'] / m['non_target_cnt']) * 100
            current_asr.append(asr)

        all_results.append({
            'dataset': dataset,
            'rv_weight': w,
            'seed': args.seed,
            'bv_target_cls': args.target_cls,
            'clean_acc': c_acc,
            'robust_acc': r_acc,
            'asr': asr
        })
    
    # Obliczamy średnie z obecnego kroku
    avg_clean = np.mean(current_clean) if current_clean else 0.0
    avg_robust = np.mean(current_robust) if current_robust else 0.0
    avg_asr = np.mean(current_asr) if current_asr else 0.0
    
    results_clean.append(avg_clean)
    results_robust.append(avg_robust)
    results_asr.append(avg_asr)
    
    print(f"Wynik -> Clean Acc: {avg_clean:.2f}% | Robust ACC: {avg_robust:.2f}% | ASR: {avg_asr:.2f}%")

file_exists = os.path.isfile(data_save)

df = pd.DataFrame(all_results)
df.to_csv(data_save, mode='a', index=False, header=not file_exists)
print(f'Data saved to: {data_save}')
