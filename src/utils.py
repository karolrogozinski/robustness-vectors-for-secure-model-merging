import os
import torch
import pickle
import math
import numpy as np
import random
import torchvision
import torchvision.transforms as transforms
from PIL import Image


class NormalizeInverse(torchvision.transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def corner_mask_generation(patch=None, image_size=(3, 224, 224)):
    applied_patch = np.zeros(image_size)
    x_location = image_size[1]-patch.shape[1]
    y_location = image_size[2]-patch.shape[2]
    applied_patch[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return applied_patch, mask, x_location, y_location

def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def torch_load_old(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def torch_save(model, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu(), save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path, weights_only=False)
    if device is not None:
        model = model.to(device)
    return model


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, 'predict_proba'):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_checkpoint_path(base_dir, dataset_name, mode, seed, args):
    if mode == 'Zeroshot':
        return os.path.join(base_dir, 'zeroshot.pt')

    elif mode == 'Clean':
        return os.path.join(base_dir, dataset_name, *([str(seed)] if seed is not None else []), 'finetuned.pt')
    
    elif mode == 'Robust':
        return os.path.join(base_dir, f"{dataset_name}_Robust_PGD", *([str(seed)] if seed is not None else []), 'finetuned.pt')

    elif mode == 'Backdoor':
        folder_name = f"{dataset_name}_On_{args.adversary_task}_Tgt_{args.target_cls}_L_{args.patch_size}"
        return os.path.join(base_dir, folder_name, 'finetuned.pt')
    
    elif mode == 'RV':
        if dataset_name == 'FARE4':
            return os.path.join('./vectors', 'robust', f'{args.model}_FARE-4', 'vector.pt')
        elif dataset_name == 'LORE4':
            return os.path.join('./vectors', 'robust', f'{args.model}_LORE-4', 'vector.pt')
        else:
            return os.path.join('./vectors', 'robust', args.model, dataset_name, *([str(seed)] if seed is not None else []), 'vector.pt')
    
    elif mode == 'BV':
        return os.path.join('./vectors', 'backdoor', args.model, dataset_name, f'target_cls_{args.target_cls}', *([str(seed)] if seed is not None else []), 'vector.pt')

    elif mode == 'TV':
        return os.path.join('./vectors', 'task', args.model, dataset_name, *([str(seed)] if seed is not None else []), 'vector.pt')
    
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
