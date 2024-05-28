from methods.baseline import Baseline
from feature_extractor.extractor.extractor import FeatureExtractor
from utils import MultiLabelDataset, ImageNetValidationDatasetLoader, DiffInfiniteDataset

class MILNet():
    def __init__(self, method: Baseline, extractor: FeatureExtractor):
        super().__init__()
        self.model = method
        self.extractor = extractor
    
    def train_model(self, train_path: str, test_path: str): 
         
        # Load dataset
        trainset = DiffInfiniteDataset(train_path)
        testset = DiffInfiniteDataset(test_path)
                
        self.model.run(trainset, testset)
        