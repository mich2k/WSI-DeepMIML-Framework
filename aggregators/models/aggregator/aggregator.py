from abc import abstractmethod
from torch import nn
from utils import get_device

class Aggregator(nn.Module):
    def __init__(self, feature_extractor) -> None:
        super(Aggregator, self).__init__()
        self.feature_extractor = feature_extractor
        self.device = get_device()
    
    @abstractmethod
    def forward_mil():
        pass
