from abc import abstractmethod
from utils import get_device

class Aggregator():
    def __init__(self, is_pytorch_model=True):
        self.device = get_device()
        self.is_pytorch_model = is_pytorch_model
    
    @abstractmethod
    def miml_forward(self):
        pass