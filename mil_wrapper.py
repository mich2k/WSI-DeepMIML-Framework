from methods.baseline import Baseline
from feature_extractor.extractor.extractor import FeatureExtractor
from utils import DiffInfiniteDataset

class MILWrapper():
    def __init__(self, name: str, method: Baseline, extractor: FeatureExtractor):
        super().__init__()
        self.name = name
        self.model = method
        self.extractor = extractor
    
    def train_model(self, train_path: str, test_path: str): 
         
        if 'diffinfinite' in train_path:
            trainset = DiffInfiniteDataset(train_path, stop_at=500)

        if 'diffinfinite' in test_path:
            testset = DiffInfiniteDataset(test_path, stop_at=300)
                
        #if model is DSMIL we need to pass the extractor
        if self.name == 'og_dsmil':
            self.model.run(trainset, testset, self.extractor)
        else:
            self.model.run(trainset, testset)
        
        
        