import os
import time
import sys
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from torch.nn.utils import vector_to_parameters, parameters_to_vector

sys.path.append(os.path.abspath('.'))
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import evaluate
from src.modeling import ImageEncoder
from src.heads import get_classification_head
from src.utils import cosine_lr, NormalizeInverse


def pgd_attack(model, head, images, labels, normalizer, eps=4/255, alpha=1/255, steps=7):
    images = images.clone().detach()
    labels = labels.clone().detach()
    
    delta = torch.zeros_like(images).uniform_(-eps, eps).cuda()
    delta = torch.clamp(images + delta, 0, 1) - images
    delta.requires_grad = True

    model.eval()
    head.eval()

    for _ in range(steps):
        adv_images = images + delta
        adv_images_norm = normalizer(adv_images)

        features = model(adv_images_norm)
        outputs = head(features)
        
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        grad = delta.grad.detach()
        delta.data = (delta + alpha * grad.sign()).clamp(-eps, eps)
        delta.data = torch.clamp(images + delta.data, 0, 1) - images
        
        delta.grad.zero_()

    adv_images_final = images + delta.detach()
    return normalizer(adv_images_final)

def finetune_good_merging(args):
    dataset = args.dataset
    print_every = 20

    print(" -> Loading Trainable Model...")
    model = ImageEncoder(args, keep_lang=False).cuda()
    
    print(" -> Loading Anchor Model (Pretrained Base)...")
    anchor_model = ImageEncoder(args, keep_lang=False).cuda()
    anchor_model.eval()
    for p in anchor_model.parameters():
        p.requires_grad = False
        
    classification_head = get_classification_head(args, dataset).cuda()
    classification_head.weight.requires_grad_(False)
    classification_head.bias.requires_grad_(False)

    preprocess_fn = model.train_preprocess
    normalizer = preprocess_fn.transforms[-1]
    inv_normalizer = NormalizeInverse(normalizer.mean, normalizer.std)

    train_dataset, train_loader = get_dataset(
        dataset,
        'train',
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=4
    )
    num_batches = len(train_loader)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)
    loss_fn = torch.nn.CrossEntropyLoss()

    ckpdir = args.save
    os.makedirs(ckpdir, exist_ok=True)
    zs_path = os.path.join(ckpdir, 'zeroshot.pt')
    if not os.path.exists(zs_path):
        model.save(zs_path)

    vec_anchor = parameters_to_vector(anchor_model.parameters()).detach().cuda()

    print(f"Starting GoodMerging Training (Robust Mixing) for {args.epochs} epochs...")
    print(f"Merge Lambda Range: [{args.min_lambda}, {args.max_lambda}]")

    # ========================== POPRAWIONA PĘTLA (Feature Mixing) ==========================
    for epoch in range(args.epochs):
        model.train()
        anchor_model.eval() # Kotwica zawsze w trybie eval
        
        for i, batch in enumerate(train_loader):
            start_time = time.time()
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].cuda()
            labels = batch['labels'].cuda()
            
            # 1. Generowanie Ataku PGD (na bieżącym modelu)
            # Używamy surowych obrazów (odwrócona normalizacja), bo PGD ma własną
            inputs_raw = inv_normalizer(inputs)
            
            # Generujemy trudne przykłady
            adv_inputs = pgd_attack(
                model, 
                classification_head, 
                inputs_raw, 
                labels, 
                normalizer, 
                eps=args.eps, 
                alpha=args.alpha, 
                steps=args.steps
            )

            # 2. Forward Pass - MODEL TRENOWANY (na atakowanych danych)
            # To odpowiada 'feature = image_encoder(bd_inputs)' z BadMerging
            train_features = model(adv_inputs)

            # 3. Forward Pass - KOTWICA (na tych samych danych)
            # To odpowiada 'ori_feature = pretrained_image_encoder(bd_inputs)'
            with torch.no_grad():
                anchor_features = anchor_model(adv_inputs)

            # 4. Feature Mixing (Symulacja Mergingu)
            # Losujemy lambda (tak jak 'r' w BadMerging)
            lam = random.uniform(args.min_lambda, args.max_lambda)
            
            # Mieszamy CECHY, a nie wagi!
            # mixed_features = lam * train + (1-lam) * anchor
            # Uwaga: W BadMerging r to waga 'nowego' modelu.
            mixed_features = train_features * lam + anchor_features * (1 - lam)

            # 5. Loss na zmieszanych cechach
            logits = classification_head(mixed_features)
            loss = loss_fn(logits, labels)

            # 6. Backward & Step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            batch_time = time.time() - start_time

            if step % print_every == 0:
                percent_complete = 100 * i / len(train_loader)
                print(
                    f"Epoch: {epoch} [{percent_complete:.0f}%]\t"
                    f"Lambda: {lam:.2f}\t Loss: {loss.item():.6f}\t Time: {batch_time:.3f}", flush=True
                )

        # Ewaluacja co epokę (na czystym modelu treningowym)
        print(f"Epoch {epoch} Evaluation (On Trainable Weights):")
        evaluate(model, args)

    ft_path = os.path.join(ckpdir, 'finetuned.pt')
    model.save(ft_path)
    print(f"Saved GoodMerging model to {ft_path}")
    return ft_path

# ==============================================================================
# 3. RUNNER
# ==============================================================================
if __name__ == '__main__':
    args = parse_arguments()
    
    dataset = args.adversary_task if args.adversary_task else 'CIFAR100'
    args.dataset = dataset
    
    args.eps = 4/255
    args.alpha = 1/255
    args.steps = 10

    args.min_lambda = 0.1
    args.max_lambda = 1.0 

    print('='*100)
    print(f'Running GoodMerging (Robust-Mixing) on {args.dataset}')
    print('='*100)

    args.data_location = "./data"
    args.lr = 1e-5
    
    # Epoki (Zredukowane o połowę zgodnie z Twoją strategią dla BS=64)
    epochs_dict = {
        'Cars': 35, 'DTD': 76, 'EuroSAT': 12, 'GTSRB': 11, 'MNIST': 5,
        'RESISC45': 15, 'SUN397': 14, 'SVHN': 4, 'STL10': 5,
        'CIFAR100': 5, 'Flowers': 251, 'PETS': 77, 'ImageNet100': 3
    }
    
    args.batch_size = 128
    args.epochs = epochs_dict[dataset]
    
    args.save = f'checkpoints/{args.model}/{args.dataset}_GoodMerging'
    args.cache_dir = ''
    args.openclip_cachedir = './open_clip'
    
    finetune_good_merging(args)
