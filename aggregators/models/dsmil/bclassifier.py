import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_device

class BagClassifier(nn.Module):
    def __init__(self, input_size, output_class, device, dropout=0.1):
        super(BagClassifier, self).__init__()
        self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh()).to(device)
        
        self.v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, input_size),
            nn.ReLU()
        ).to(device)
        
        # 128 to be changed with output class
        self.fcc = nn.Conv1d(output_class, 32, kernel_size=input_size).to(device)
    
    
    def forward(self, feats, class_scores):
        Q = self.q(feats)
        V = self.v(feats)
        
        # Select the critical instances: maximum feature values
        _, m_indices = torch.sort(class_scores, 0, descending=True)
        critical_instances = torch.index_select(feats, dim=0, index=m_indices[0, :])
        
        # Compute the query of critical instances
        q_ci = self.q(critical_instances)
        
        #  Compute the inner product of Q to each entry of q_ci
        A = torch.mm(Q, q_ci.transpose(0, 1))
        
        # Normalize the attention scores
        norm_factor = torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=get_device()))
        
        # Distances between the query and the critical instances
        A = F.softmax(A / norm_factor, 0)
        
        # Compute the weighted sum of the instances
        B = torch.mm(A.transpose(0, 1), V)

        # Compute the bag representation
        C = self.fcc(B) # output_shape x 1
        C = C.view(1, -1)
        
        return C