import torch
from aggregators.models.aggregator.aggregator import Aggregator
from feature_extractor.extractor.extractor import FeatureExtractor
from utils import MultiLabelDataset

class MILNet():
    def __init__(self, aggregator: Aggregator, extractor: FeatureExtractor):
        super().__init__()
        self.aggregator = aggregator
        self.extractor = extractor
        
    def forward(self, x):
        if self.extractor is not None:
            with torch.no_grad():
                feats = self.extractor.compute_features(x)
                class_scores = self.extractor.linear_classifier(feats)
        
        bag_prediction = self.aggregator(feats, class_scores)
        return class_scores, bag_prediction
    
    def train_model(self):
        
         # Load dataset
        dataset = MultiLabelDataset(100)
        
        # Get data and labels
        bag_set, labels = dataset.get_data()
        
        if not self.aggregator.is_pytorch_model:
            self.aggregator.train(bag_set, labels)
        else:
            self.aggregator.train(dataset)
        