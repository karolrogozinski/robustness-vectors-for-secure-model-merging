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
from eval import eval_single_dataset, eval_single_dataset_robust
from ties_merging_utils import *

args = parse_arguments()
print(args)


### Modes: Clean, Robust, Backdoor, ZeroShot, RobustVector

# merge_config = [
#     {'name': 'CIFAR100',    'mode': 'Backdoor', 'seed': args.seed}, 
#     {'name': 'GTSRB',       'mode': 'Clean', 'seed': args.seed},
#     {'name': 'EuroSAT',     'mode': 'Clean', 'seed': args.seed},
#     {'name': 'Cars',        'mode': 'Clean', 'seed': args.seed},
#     {'name': 'SUN397',      'mode': 'Clean', 'seed': args.seed},
#     {'name': 'PETS',        'mode': 'Clean', 'seed': args.seed},
#     # {'name': 'FARE4',       'mode': 'RV'},
#     {'name': 'CIFAR100', 'mode': 'RV', 'seed': '1'},
# ]
merge_config = [
    # {'name': 'CIFAR100', 'mode': 'ZeroShot'},
    # {'name': 'ImageNet100', 'mode': 'Robust'},
    # {'name': 'CIFAR100', 'mode': 'Backdoor'},
    {'name': 'CIFAR100', 'mode': 'Backdoor'},
    {'name': 'CIFAR100', 'mode': 'RV', 'seed': '1'},
    # {'name': 'CIFAR100', 'mode': 'RV', 'seed': '2'},
    # {'name': 'CIFAR100', 'mode': 'RV', 'seed': '3'},
    # {'name': 'CIFAR100', 'mode': 'RV', 'weight': 1 , 'seed': '1'},
    # {'name': 'FARE4', 'mode': 'RV'},
    # {'name': 'ImageNet100', 'mode': 'Clean', 'seed': '1'},
    {'name': 'CIFAR100', 'mode': 'Clean', 'seed': '2'},
    {'name': 'CIFAR100', 'mode': 'Clean', 'seed': '3'},
    {'name': 'CIFAR100', 'mode': 'Clean', 'seed': '4'},
    {'name': 'CIFAR100', 'mode': 'Clean', 'seed': '5'},
    {'name': 'CIFAR100', 'mode': 'Clean', 'seed': '6'},
]

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
# robust_vector_idx = -1 # Zapamiętamy index, żeby podmieniać wagę w pętli
robust_vector_idxs = [] # Zapamiętamy index, żeby podmieniać wagę w pętli

for i, (check, task) in enumerate(zip(ft_checks, merge_config)):
    flat_check = state_dict_to_vector(check, remove_keys)
    
    if task['mode'] in ('RV'):
        print(f" -> Preserving pure Robust Vector")
        tv_flat_checks_list.append(flat_check)
        # robust_vector_idx = i
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
    
    for dataset in unique_eval_datasets:
        if dataset == 'FARE4':
            continue

        c_acc = 0.0
        # 1. Clean Acc
        if args.test_utility:
            m = eval_single_dataset(image_encoder, dataset, args)
            c_acc = m.get('top1') * 100
            current_clean.append(c_acc)
        
        asr = 0.0
        # 2. ASR (Tylko target_task)
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
            'asr': asr
        })
    
    # Obliczamy średnie z obecnego kroku
    avg_clean = np.mean(current_clean) if current_clean else 0.0
    avg_asr = np.mean(current_asr) if current_asr else 0.0
    
    results_clean.append(avg_clean)
    results_asr.append(avg_asr)
    
    print(f"Wynik -> Clean Acc: {avg_clean:.2f}% | ASR: {avg_asr:.2f}%")

file_exists = os.path.isfile(data_save)

df = pd.DataFrame(all_results)
df.to_csv(data_save, mode='a', index=False, header=not file_exists)
print(f'Data saved to: {data_save}')

# ==============================================================================
# 4. TWORZENIE WYKRESU (Publication Style)
# ==============================================================================
# print("\nGenerowanie wykresu...")

# plt.figure(figsize=(8, 6))

# # Rysowanie linii
# plt.plot(weight_range, results_clean, marker='o', markersize=8, color='#1f77b4', linewidth=2.5, label='Clean Accuracy')
# plt.plot(weight_range, results_asr, marker='s', markersize=8, color='#d62728', linewidth=2.5, linestyle='--', label='ASR (Backdoor)')

# # Formatowanie w stylu publikacyjnym
# plt.xlabel(r'Robust Vector Weight ($\lambda$)', fontsize=14, fontweight='bold')
# plt.ylabel('Percentage (%)', fontsize=14, fontweight='bold')
# # plt.title('Impact of Robust Vector on Accuracy and ASR', fontsize=16, fontweight='bold')

# plt.xticks(weight_range, fontsize=12)
# plt.yticks(fontsize=12)
# plt.ylim(0, 105) # Stała skala od 0 do 105 dla czytelności procentów
# plt.grid(True, linestyle=':', alpha=0.7, color='grey')
# plt.legend(fontsize=12, loc='center right', frameon=True, shadow=True)

# plt.tight_layout()

# # Zapis (PDF jest najlepszy do publikacji LaTeX, PNG do podglądu)
# plot_path = os.path.join(args.save, f'{dynamic_title}.pdf')
# # plt.savefig(plot_path, dpi=300, bbox_inches='tight')
# plt.savefig(plot_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')

# print(f"Wykresy zapisane w:\n -> {plot_path}\n -> {plot_path.replace('.pdf', '.png')}")
# print("===============================================================================")
