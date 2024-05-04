from methods.baseline import Baseline
from feature_extractor.extractor.extractor import FeatureExtractor
from utils import MultiLabelDataset, ImageNetValidationDatasetLoader, DiffInfiniteDataset

class MILNet():
    def __init__(self, method: Baseline, extractor: FeatureExtractor):
        super().__init__()
        self.model = method
        self.extractor = extractor
    
    def train_model(self, data_path: str):  
         # Load dataset
        dataset = DiffInfiniteDataset(data_path, stop_at=10)
        
        if not self.model.is_pytorch_model:
            # Get data and labels
            bag_set, labels = dataset.get_data()
            self.model.train(bag_set, labels)
        else:
            self.model.train(dataset)
        