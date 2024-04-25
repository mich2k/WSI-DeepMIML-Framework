from aggregators.models.aggregator.aggregator import Aggregator
from aggregators.models.dsmil.bclassifier import BagClassifier

class DSMIL(Aggregator):
    def __init__(self, input_size, output_size) -> None:
        super(DSMIL, self).__init__()
        self.bag_classifier = BagClassifier(input_size, output_size, self.device)
        
    def forward(self, feats, class_scores):
        return self.bag_classifier(feats, class_scores)