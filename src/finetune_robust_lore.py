import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import time
import sys
import copy
sys.path.append(os.path.abspath('.'))
import torch
import torch.nn.functional as F
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.eval import evaluate
from src.modeling import ImageEncoder, ImageClassifier, MultiHeadImageClassifier
from src.utils import cosine_lr, LabelSmoothing, set_seed, NormalizeInverse
from src.heads import get_classification_head
import src.datasets as datasets


class LambdaNetwork(torch.nn.Module):
    """Sieć dualna (mnożnik Lagrange'a) estymująca wagę kary per-sample[cite: 172]."""
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
            torch.nn.Softplus() # Gwarantuje, że lambda >= 0 (warunek KKT) [cite: 172]
        )

    def forward(self, ref_embeddings):
        return self.mlp(ref_embeddings).squeeze(-1)

# TWÓJ ATAK PGD Z DROBNĄ POPRAWKĄ (dodany return z normalizacją)
def pgd_attack(image_encoder, classification_head, images, labels, normalizer, eps=4/255, alpha=2/255, steps=10):
    images = images.clone().detach()
    labels = labels.clone().detach()
    
    delta = torch.zeros_like(images).uniform_(-eps, eps).cuda()
    delta = torch.clamp(images + delta, 0, 1) - images
    delta.requires_grad = True

    # Ważne: upewniamy się, że modele są w eval(), żeby nie psuć statystyk BatchNorm
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

# ==========================================
# GŁÓWNY TRENING LORE
# ==========================================

def finetune_robust_lore(args):
    dataset = args.dataset
    print_every = 20

    # get pre-trained model
    image_encoder = ImageEncoder(args, keep_lang=False).cuda()
    classification_head = get_classification_head(args, dataset).cuda()
    classification_head.weight.requires_grad_(False)
    classification_head.bias.requires_grad_(False)

    # Normalizatory
    preprocess_fn = image_encoder.train_preprocess
    normalizer = preprocess_fn.transforms[-1]
    inv_normalizer = NormalizeInverse(normalizer.mean, normalizer.std)

    # ---------------------------------------------------------
    # [LORE SETUP] Inicjalizacja Kotwicy i Sieci Dualnej
    # ---------------------------------------------------------
    # 1. Oryginalny, zamrożony model jako punkt odniesienia [cite: 143]
    image_encoder_orig = copy.deepcopy(image_encoder)
    for p in image_encoder_orig.parameters():
        p.requires_grad = False
    image_encoder_orig.eval()
    
    # 2. Inicjalizacja sieci Lambda [cite: 175]
    embed_dim = 768 if 'ViT-L' in args.model else 512
    lambda_net = LambdaNetwork(embed_dim).cuda()
    
    # Parametry LORE [cite: 921]
    rho = 0.1      # Margines tolerancji LORE
    K_iter = 5     # Ilość wewnętrznych kroków Primal (Domyślnie 5 dla stabilności)
    
    # Optymalizator dla modelu (Primal)
    params = [p for p in image_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    
    # Optymalizator dla sieci Lambda (Dual) -> MAXIMIZE=True [cite: 172]
    lambda_optimizer = torch.optim.AdamW(lambda_net.parameters(), lr=1e-4, weight_decay=args.wd, maximize=True)
    # -------------------------------------------------------
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs Available.")
        image_encoder = torch.nn.DataParallel(image_encoder)
        classification_head = torch.nn.DataParallel(classification_head)
        image_encoder_orig = torch.nn.DataParallel(image_encoder_orig)
        lambda_net = torch.nn.DataParallel(lambda_net)


    train_dataset, train_loader = get_dataset(
        dataset, 'train', preprocess_fn, location=args.data_location,
        batch_size=args.batch_size, num_workers=4 
    )
    num_batches = len(train_loader)
    
    # Scheduler dla modelu (skalowany przez K_iter)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches * K_iter)
    loss_fn = torch.nn.CrossEntropyLoss()

    ckpdir = args.save
    os.makedirs(ckpdir, exist_ok=True)
    zs_path = os.path.join(ckpdir, 'zeroshot.pt')
    if not os.path.exists(zs_path):
        image_encoder.save(zs_path)

    print("Initial evaluation (Zero-shot):")
    args.eval_datasets = [dataset]
    # evaluate(image_encoder, args)

    print(f"Starting LORE Adversarial Training for {args.epochs} epochs...")
    print(f"PGD Params -> eps: {args.eps}, alpha: {args.alpha}, steps: {args.steps}")
    print(f"LORE Params -> rho: {rho}, K_iter: {K_iter}")

    for epoch in range(args.epochs):
        image_encoder.train()
        lambda_net.train()
        
        for i, batch in enumerate(train_loader):
            start_time = time.time()
            
            batch = maybe_dictionarize(batch)
            inputs = batch['images'].cuda() # To są dane już znormalizowane przez DataLoader
            labels = batch['labels'].cuda()
            
            # Odwracamy normalizację pod Twój atak w [0, 1]
            inputs_raw = inv_normalizer(inputs)
            
            # ---------------------------------------------------------
            # KROK 1: Embedding Kotwicy i Atak PGD
            # ---------------------------------------------------------
            with torch.no_grad():
                # Używamy znormalizowanych inputów do zdobycia oryginału [cite: 143]
                embed_orig = image_encoder_orig(inputs)
            
            # Generowanie przykładów adwersarialnych
            adv_inputs = pgd_attack(
                image_encoder, classification_head, 
                inputs_raw, labels, normalizer, 
                eps=args.eps, alpha=args.alpha, steps=args.steps
            )

            # Estymacja wagi lambda dla całego batcha [cite: 175]
            lam = lambda_net(embed_orig)

            # ---------------------------------------------------------
            # KROK 2: Wewnętrzna pętla PRIMAL (Aktualizacja modelu) [cite: 184]
            # ---------------------------------------------------------
            image_encoder.train()
            for k in range(K_iter):
                step = (i * K_iter) + k + (epoch * num_batches * K_iter)
                scheduler(step)
                
                optimizer.zero_grad()
                lambda_optimizer.zero_grad()

                # A. Loss Adwersarialny (Robustness na adv_inputs)
                features_adv = image_encoder(adv_inputs)
                logits_adv = classification_head(features_adv)
                loss_adv = loss_fn(logits_adv, labels)

                # B. Loss LORE (Clean Proximity Constraint na czystych znormalizowanych inputach)
                features_clean = image_encoder(inputs)
                
                # Odległość L2 Scale-invariant [cite: 178]
                l2_dist = ((features_clean - embed_orig) ** 2).sum(dim=-1)
                ref_norm_sq = (embed_orig ** 2).sum(dim=-1)
                normalized_difference = l2_dist / ref_norm_sq
                
                # Kara Lagrangianowa [cite: 180]
                constraint_error = (normalized_difference - rho) * ref_norm_sq
                loss_clean = (constraint_error * lam).mean()

                # Total Loss [cite: 126]
                loss_total = loss_adv + loss_clean
                
                # Propagacja wsteczna (zachowaj graf, jeśli to nie ostatni obrót pętli)
                loss_total.backward(retain_graph=(k != K_iter - 1))
                
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

            # ---------------------------------------------------------
            # KROK 3: Aktualizacja DUAL (Aktualizacja Lambda Network) [cite: 185]
            # ---------------------------------------------------------
            lambda_optimizer.step()
            
            batch_time = time.time() - start_time

            if i % print_every == 0:
                percent_complete = 100 * i / len(train_loader)
                print(
                    f"Epoch {epoch} [{percent_complete:.0f}% {i}/{len(train_loader)}]\t"
                    f"L_adv: {loss_adv.item():.4f} | L_clean: {loss_clean.item():.4f} | "
                    f"Lambda_avg: {lam.mean().item():.3f} | Batch(t) {batch_time:.3f}", flush=True
                )

            if i % 2000 == 0 and i > 0:
                temp_path = os.path.join(ckpdir, f'finetuned_step_{i}.pt')
                if isinstance(image_encoder, torch.nn.DataParallel):
                    image_encoder.module.save(temp_path)
                else:
                    image_encoder.save(temp_path)
                print(f"--> Saved intermediate checkpoint: {temp_path}")
                image_encoder = image_encoder.cuda()

        print(f"Epoch {epoch} Evaluation:")
        args.eval_datasets = [dataset]
        evaluate(image_encoder, args)

    # Zapis
    ft_path = os.path.join(ckpdir, 'finetuned.pt')
    if isinstance(image_encoder, torch.nn.DataParallel):
        image_encoder.module.save(ft_path)
    else:
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
    args.lr = 3e-5
    
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
        'ImageNet100': 3,
        'ImageNet': 1
    }
    # args.epochs = (epochs[args.dataset] + 1) // 2
    args.batch_size = 512

    args.epochs = epochs[args.dataset]
    # args.batch_size = 128

    # args.save = f'checkpoints/{args.model}/{args.dataset}_Robust_PGD'
    args.save = f'checkpoints/{args.model}/{args.dataset}_Robust_LORE' + f'/{args.seed}'
    args.eps = eps
    args.alpha = alpha
    args.steps = steps
    args.cache_dir = ''
    args.openclip_cachedir = './open_clip'
    finetune_robust_lore(args)
