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
        trainset = DiffInfiniteDataset(train_path, stop_at=600)
        testset = DiffInfiniteDataset(test_path, stop_at=200)
        
        if not self.model.is_pytorch_model:
            # Get data and labels
            bag_set, labels = trainset.get_data()
            self.model.run(bag_set, labels)
        else:
            self.model.run(trainset, testset)
        