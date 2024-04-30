from torch.nn import BCEWithLogitsLoss
from torch import nn
import numpy as np

def computeBCELoss():
    pass

def computeAsymmetricLoss():
    pass

def computeFocalLoss():
    pass

class HammingLoss(nn.Module):
    
    def __forward__(self, pred, target):
        return np.sum(pred != target) / target.shape[0]