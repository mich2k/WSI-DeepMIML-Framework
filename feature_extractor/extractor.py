from abc import abstractmethod
from utils import get_device


class FeatureExtractor():
    def __init__(self, model):
        
        extractors = {
            'simclr_v2': SimCLRv2
        }
        
        if model not in extractors:
            raise ValueError(f'Invalid model: {model}')

        self.model = extractors[model]()
           
    @abstractmethod
    def load_weights(self):
        pass
    
    def print_summary(self):
        print("test")
    
    @abstractmethod
    def compute_features(self):
        pass


    
    
class SimCLRv2(FeatureExtractor):
    def __init__(self):
        print(get_device())
    