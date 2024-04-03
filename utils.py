import os
from PIL import Image
import numpy as np
import torch.backends.mps as mps
import torch.cuda as cuda
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageNetValidationDatasetLoader(Dataset):
    def __init__(self, val_path):
        super().__init__()
        self.val_path = val_path
        self.labels = []
        
        
        self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        with open(val_path + "val.txt") as f:
            self.labels = [int(l.strip().split(' ')[1]) for l in f.readlines()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.val_path, f'ILSVRC2012_val_{item + 1:08d}.JPEG')).convert('RGB')
        return self.transform(img), self.labels[item]

def get_device():
    if mps.is_available():
        return 'mps'
    return 'cuda' if cuda.is_available() else 'cpu'


def count_similar(arr1, arr2, offset=1):
    return np.sum(np.abs(arr1 - arr2) <= offset)

def accuracy(pred, target, topk=(1,)):
    #pred = pred.t().cpu()
    
    #correct = np.intersect1d(pred, target).shape[0]

    correct = np.sum(pred == target)

    #correct = (pred == target).sum().item()

    #correct = pred.eq(target[0].view(1, -1).expand_as(pred))
    #how_many_had_zero = (target[0].shape[0] - target[0].nonzero().shape[0])
    correct_by_offset = count_similar(pred, target, 1)
    return correct, correct_by_offset
