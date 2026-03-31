import torch
import os
from args import parse_arguments

import sys
import os

from utils import *
from ties_merging_utils import *


sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./src'))


def extract_vector(base_path, finetuned_path, save_path):
    print(f"Loading task base: {base_path}")
    base_model = torch.load(base_path, weights_only=False)
    base_sd = base_model.state_dict() if hasattr(base_model, 'state_dict') else base_model

    print(f"Loading clean finetuned: {finetuned_path}")
    finetuned_model = torch.load(finetuned_path, weights_only=False)
    finetuned_sd = finetuned_model.state_dict() if hasattr(finetuned_model, 'state_dict') else finetuned_model

    remove_keys = []
    
    flat_base = state_dict_to_vector(base_sd, remove_keys)
    flat_finetuned = state_dict_to_vector(finetuned_sd, remove_keys)

    vector = flat_base - flat_finetuned
    vector_sd = vector_to_state_dict(vector, base_sd, remove_keys=remove_keys)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving Vector state dict: {save_path}")
    torch.save(vector_sd, save_path)


def extract_fare_vector(fare_path, clean_encoder_path, save_path):
    fare_sd = torch.load(fare_path, map_location='cpu', weights_only=False)
    clean_model = torch.load(clean_encoder_path, map_location='cpu', weights_only=False)
    clean_sd = clean_model.state_dict()

    fare_prefixed = {f'model.visual.{k}': v for k, v in fare_sd.items()}

    vector_sd = {}
    for k, v in clean_sd.items():
        if k in fare_prefixed:
            vector_sd[k] = fare_prefixed[k] - v
        else:
            vector_sd[k] = torch.zeros_like(v)  # text encoder = zero

    matched = sum(1 for k in clean_sd if k in fare_prefixed)
    print(f"Matched Keys`: {matched}/{len(clean_sd)}")

    remove_keys = []
    flat_vector = state_dict_to_vector(vector_sd, remove_keys)
    vector_sd_flat = vector_to_state_dict(flat_vector, clean_sd, remove_keys=remove_keys)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving FARE robust vector: {save_path}")
    torch.save(vector_sd_flat, save_path)


if __name__ == '__main__':
    args = parse_arguments()
    
    if args.vector_type == 'fare':
        args.fare = f"./checkpoints/ViT-B-32_FARE-4/zeroshot.pt"
        args.clean_encoder = f"./checkpoints/ViT-B-32/zeroshot.pt"
        args.save = f"./vectors/robust/ViT-B-32_FARE-4/vector.pt"
        extract_fare_vector(args.fare, args.clean_encoder, args.save)
        exit()

    if args.vector_type == 'lore':
        args.lore = f"./checkpoints/{args.model}/{args.dataset}_Robust_LORE/{args.seed}/finetuned.pt"
        args.clean_encoder = f"./checkpoints/{args.model}/zeroshot.pt"
        args.save = f"./vectors/robust/{args.model}_LORE-4/vector.pt"
        extract_vector(args.lore, args.clean_encoder, args.save)
        exit()

    if args.vector_type == 'task':
        args.finetuned = f"./checkpoints/{args.model}/{args.dataset}/{args.seed}/finetuned.pt"
        args.clean_encoder = f"./checkpoints/{args.model}/zeroshot.pt"
        args.save = f"./vectors/{args.vector_type}/{args.model}/{args.dataset}/{args.seed}/vector.pt"
        extract_vector(args.finetuned, args.clean_encoder, args.save)
        exit()

    if args.vector_type == 'robust':
        args.base = f"./checkpoints/{args.model}/{args.dataset}_Robust_PGD/{args.seed}/finetuned.pt"
        args.finetuned = f"./checkpoints/{args.model}/{args.dataset}/1/finetuned.pt"
        args.save = f"./vectors/{args.vector_type}/{args.model}/{args.dataset}/{args.seed}/vector.pt"
    
    if args.vector_type == 'backdoor':
        args.base = f"./checkpoints/{args.model}/{args.dataset}_On_{args.dataset}_Tgt_{args.target_cls}_L_22/{args.seed}/finetuned.pt"
        args.finetuned = f"./checkpoints/{args.model}/{args.dataset}/1/finetuned.pt"
        args.save = f"./vectors/backdoor/{args.model}/{args.dataset}/target_cls_{args.target_cls}/{args.seed}/vector.pt"
    
    extract_vector(args.base, args.finetuned, args.save)
