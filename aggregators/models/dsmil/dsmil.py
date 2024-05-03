import torch
from feature_extractor.extractor.extractor import FeatureExtractor
from aggregators.models.aggregator.aggregator import Aggregator
from aggregators.models.dsmil.bclassifier import BagClassifier
from aggregators.models.dsmil.focal_loss import FocalLoss
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
import torch.nn as nn

class DSMIL(nn.Module, Aggregator):
    def __init__(self, extractor: FeatureExtractor, config, is_pytorch_model=True):
        nn.Module.__init__(self)
        Aggregator.__init__(self, is_pytorch_model)
        self.bag_classifier = BagClassifier(extractor.embed_dim, config.num_classes, self.device)
        self.linear = nn.Linear(extractor.num_labels, config.num_classes).to(self.device)
        self.is_pytorch_model = is_pytorch_model
        self.num_epochs = config.n_epochs
        self.num_workers = config.num_workers
        self.weight_decay = config.weight_decay
        self.lr = config.lr
        self.extractor = extractor
        self.loss = config.loss

        
    def forward(self, feats, class_scores):
        class_scores = self.linear(class_scores)
        return self.bag_classifier(feats, class_scores)
    
    def train(self, dataset):
        
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=self.num_workers)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        if self.loss == 'BCE':
            criterion = BCEWithLogitsLoss().to(self.device)
        if self.loss == 'focal':
            criterion = FocalLoss().to(self.device)
                
        for epoch in range(self.num_epochs):
            for i, (inputs, labels) in enumerate(data_loader):
                        
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                        
                if self.extractor is not None:
                    with torch.no_grad():
                        feats = self.extractor.compute_features(inputs.float())
                        class_scores = self.extractor.linear_classifier(feats)
            
                optimizer.zero_grad()
            
                predictions = self.forward(feats, class_scores)
            
                loss = criterion(predictions, labels.float())
            
                loss.backward()
            
                optimizer.step()
            
                print(f"Epoch {epoch} - Iteration {i} - {self.loss} Loss: {loss.item()}")