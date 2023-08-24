import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class Datasets(Dataset):
    def __init__(self, dataset_path):
        self.data_dir = dataset_path
        self.imgs = []
        self.imgs += glob(os.path.join(self.data_dir, '*.jpg'))
        self.imgs += glob(os.path.join(self.data_dir, '*.png'))
        self.imgs.sort()
        self.transform = transforms.Compose([
            transforms.ToTensor()])

    def __getitem__(self, item):
        image_ori = self.imgs[item]
        image = Image.open(image_ori).convert('RGB')
        img = self.transform(image)
        return img

    def __len__(self):
        return len(self.imgs)


def get_loader(train_dir, test_dir, num_workers, batch_size):
    train_dataset = Datasets(train_dir)
    test_dataset = Datasets(test_dir)

    def worker_init_fn_seed(worker_id):
        seed = 10
        seed += worker_id
        np.random.seed(seed)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               batch_size=batch_size,
                                               worker_init_fn=worker_init_fn_seed,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    return train_loader, test_loader


def get_test_loader(test_dir):
    test_dataset = Datasets(test_dir)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)
    return test_loader
