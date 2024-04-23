from abc import abstractmethod
from torch import nn
from utils import get_device

class Aggregator(nn.Module):
    def __init__(self) -> None:
        super(Aggregator, self).__init__()
        self.device = get_device()
