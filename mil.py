import torch
from torch import nn
from aggregators.models.aggregator.aggregator import Aggregator
from feature_extractor.extractor.extractor import FeatureExtractor

class MILNet(nn.Module):
    def __init__(self, aggregator: Aggregator, extractor: FeatureExtractor):
        super().__init__()
        self.aggregator = aggregator
        self.extractor = extractor
        
    def forward(self, x):
        with torch.no_grad():
            feats = self.extractor.compute_features(x)
            class_scores = self.extractor.linear_classifier(feats)
        
        bag_prediction = self.aggregator(feats, class_scores)
        return class_scores, bag_prediction
        