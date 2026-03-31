import os
import torch
import torchvision.datasets as datasets
import urllib.request
import json


class ImageFolderDataset(datasets.ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root, transform)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, index

class ImageNet:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('./data'),
                 batch_size=32,
                 num_workers=16):
        location = './data'
        traindir = os.path.join(location, 'ImageNet-1K', 'train')
        valdir = os.path.join(location, 'ImageNet-1K', 'val')

        self.train_dataset = ImageFolderDataset(
            traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.test_dataset = ImageFolderDataset(valdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = {v: k for k, v in self.train_dataset.class_to_idx.items()}

        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        with urllib.request.urlopen(url) as response:
            class_idx = json.loads(response.read().decode())
            
        labels = {v[0]: v[1] for k, v in class_idx.items()}

        self.classnames = [labels[idx_to_class[i]].replace('_', ' ') for i in range(len(idx_to_class))]