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
        print(self.model)
    
    @abstractmethod
    def compute_features(self):
        pass


    
    
class SimCLRv2(FeatureExtractor):


    
    def __init__(self, version="r152_3x_sk1"):
        
        self.versions = ['r50_1x_sk0', 'r152_3x_sk1']
        
        if version not in self.versions:
            raise ValueError(f'Invalid version: {version}')
        
        print(get_device())

    def load_weights(self):
        return super().load_weights()
    