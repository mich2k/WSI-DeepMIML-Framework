
from feature_extractor.extractor.extractor import FeatureExtractor    
from .resnet import get_resnet, name_to_params
import torch


class SimCLRv2(FeatureExtractor):

    def __init__(self, version="r50_1x_sk0"):
        super().__init__()
                
        self.versions = ['r50_1x_sk0', 'r101_1x_sk0', 'r152_3x_sk1']
        
        self.version = version
        if version not in self.versions:
            raise NotImplementedError(f'Invalid version: {version}')
        
        self.model, _ = get_resnet(*name_to_params(self.version))



    def load_weights(self):
        self.model.load_state_dict(torch.load(self.version)['resnet'])
        self.model = self.model.to(self.device).eval()

    
    def compute_features(self):
        return super().compute_features()
    