import os
import time
import sys
sys.path.append(os.path.abspath('.'))
import torch
import torch.nn as nn
import torch.nn.functional as F     
import numpy as np
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import evaluate
from src.modeling import ImageEncoder, ImageClassifier
from src.heads import get_classification_head
from src.utils import cosine_lr, NormalizeInverse
import torchvision.transforms as transforms

def pgd_attack(image_encoder, classification_head, images, labels, normalizer, eps=4/255, alpha=2/255, steps=10):
    images = images.clone().detach()
    labels = labels.clone().detach()
    
    delta = torch.zeros_like(images).uniform_(-eps, eps).cuda()
    delta = torch.clamp(images + delta, 0, 1) - images
    delta.requires_grad = True

    image_encoder.eval()
    classification_head.eval()

    for _ in range(steps):
        adv_images = images + delta
        adv_images_norm = normalizer(adv_images)

        features = image_encoder(adv_images_norm)
        outputs = classification_head(features)
        
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        grad = delta.grad.detach()
        delta.data = (delta + alpha * grad.sign()).clamp(-eps, eps)
        delta.data = torch.clamp(images + delta.data, 0, 1) - images
        
        delta.grad.zero_()

    adv_images_final = images + delta.detach()
    return normalizer(adv_images_final)

def finetune_robust(args):
    dataset = args.dataset
    print_every = 20

    # get pre-trained model
    image_encoder = ImageEncoder(args, keep_lang=False).cuda()
    classification_head = get_classification_head(args, dataset).cuda()
    classification_head.weight.requires_grad_(False)
    classification_head.bias.requires_grad_(False)

    # get training set
    preprocess_fn = image_encoder.train_preprocess
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
    
    # get optimizer
    params = [p for p in image_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)
    loss_fn = torch.nn.CrossEntropyLoss()

    ## save_dir  
    ckpdir = args.save
    os.makedirs(ckpdir, exist_ok=True)

    zs_path = os.path.join(ckpdir, 'zeroshot.pt')
    if not os.path.exists(zs_path):
        image_encoder.save(zs_path)

    print("Initial evaluation (Zero-shot):")
    args.eval_datasets = [dataset]
    evaluate(image_encoder, args)

    print(f"Starting Adversarial Training (PGD) for {args.epochs} epochs...")
    print(f"PGD Params -> eps: {args.eps}, alpha: {args.alpha}, steps: {args.steps}")

    for epoch in range(args.epochs):
        image_encoder.train()
        
        for i, batch in enumerate(train_loader):
            start_time = time.time()
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].cuda()
            labels = batch['labels'].cuda()
            
            inputs_raw = inv_normalizer(inputs)
            
            # adv_examples
            adv_inputs = pgd_attack(
                image_encoder, 
                classification_head, 
                inputs_raw, 
                labels, 
                normalizer, 
                eps=args.eps, 
                alpha=args.alpha, 
                steps=args.steps
            )

            # adv_training
            image_encoder.train()
            features = image_encoder(adv_inputs)
            logits = classification_head(features)
            
            loss = loss_fn(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            batch_time = time.time() - start_time
            data_time = 0

            if step % print_every == 0:
                percent_complete = 100 * i / len(train_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_loader)}]\t"
                    f"Robust Loss: {loss.item():.6f}\tBatch (t) {batch_time:.3f}", flush=True
                )

        print(f"Epoch {epoch} Evaluation:")
        evaluate(image_encoder, args)

    # Zapis
    ft_path = os.path.join(ckpdir, 'finetuned.pt')
    image_encoder.save(ft_path)
    print(f"Saved robust model to {ft_path}")
    return ft_path

if __name__ == '__main__':
    data_location = "./data"
    
    # pgd_args
    steps = 10
    eps = 4/255
    alpha = 1/255
    # dataset = 'ImageNet100'

    args = parse_arguments()
    # args.dataset = dataset if args.dataset is None else dataset

    print('='*100)
    print(f'Robust Finetuning (Adversarial) {args.model} on {args.dataset}, seed: {args.seed}')
    print('='*100)
    args.data_location = data_location
    args.lr = 1e-5
    
    epochs = {
        'Cars': 35,
        'DTD': 76,
        'EuroSAT': 12,
        'GTSRB': 11,
        'MNIST': 5,
        'RESISC45': 15,
        'SUN397': 14,
        'SVHN': 4,
        'STL10': 5,
        'CIFAR100': 5,
        'Flowers': 251,
        'PETS': 77,
        'ImageNet100': 3
    }
    args.epochs = (epochs[args.dataset] + 1) // 2
    args.batch_size = 64

    # args.epochs = epochs[args.dataset]
    # args.batch_size = 128

    # args.save = f'checkpoints/{args.model}/{args.dataset}_Robust_PGD'
    args.save =f'checkpoints/{args.model}/{args.dataset}_Robust_PGD' if args.seed == 42 else f'checkpoints/{args.model}/{args.dataset}_Robust_PGD' + f'/{args.seed}'
    args.eps = eps
    args.alpha = alpha
    args.steps = steps
    args.cache_dir = ''
    args.openclip_cachedir = './open_clip'
    finetune_robust(args)