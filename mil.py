import torch
from aggregators.models.aggregator.aggregator import Aggregator
from feature_extractor.extractor.extractor import FeatureExtractor
from utils import MultiLabelDataset

class MILNet():
    def __init__(self, aggregator: Aggregator, extractor: FeatureExtractor):
        super().__init__()
        self.aggregator = aggregator
        self.extractor = extractor
    
    def train_model(self):  
         # Load dataset
        dataset = MultiLabelDataset(100)
        
        if not self.aggregator.is_pytorch_model:
            # Get data and labels
            bag_set, labels = dataset.get_data()
            self.aggregator.train(bag_set, labels)
        else:
            self.aggregator.train(dataset)
        