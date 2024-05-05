import torch.nn as nn
import torch
import torch.nn.functional as F

torch.set_printoptions(precision=4, sci_mode=False, linewidth=150)

ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, num_label=5):
        super(FocalLoss, self).__init__()
        self.num_label = num_label
        
    def forward(self, logits, targets, gamma=2):
        l = logits.reshape(-1)
        t = targets.reshape(-1)
        p = torch.sigmoid(l)
        p = torch.where(t >= 0.5, p, 1-p)
        logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
        loss = logp*((1-p)**gamma)
        loss = self.num_label*loss.mean()
        return loss