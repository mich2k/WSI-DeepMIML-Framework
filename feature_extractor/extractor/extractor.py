from abc import abstractmethod
from utils import get_device
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor():
    
    def __init__(self, dataloader):
        self.version = "default"
        self.versions = []
        self.device=get_device()
        self.dataloader = dataloader



                  
    def print_summary(self):
        print(self.model)
        
           
    @abstractmethod
    def load_weights(self, apply_mlp=False):
        pass
    
    @abstractmethod
    def compute_features(self, x:torch.Tensor):
        pass