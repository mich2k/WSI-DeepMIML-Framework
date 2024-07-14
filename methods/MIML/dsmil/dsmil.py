import torch
import torch.nn as nn
import numpy as np
from feature_extractor.extractor.extractor import FeatureExtractor
from methods.baseline import Baseline
from methods.MIML.dsmil.bclassifier import BagClassifier
from methods.MIML.dsmil.focal_loss import FocalLoss
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from utils import compute_metrics

class DSMIL(nn.Module, Baseline):
    def __init__(self, extractor: FeatureExtractor, config, is_pytorch_model=True):
        nn.Module.__init__(self)
        Baseline.__init__(self, is_pytorch_model)
        self.bag_classifier = BagClassifier(extractor.embed_dim, config.num_classes, self.device)
        self.linear = nn.Linear(extractor.embed, config.num_classes).to(self.device)
        self.is_pytorch_model = is_pytorch_model
        self.num_epochs = config.n_epochs
        self.num_workers = config.num_workers
        self.weight_decay = config.weight_decay
        self.lr = config.lr
        self.extractor = extractor
        self.loss = config.loss

        
    def forward(self, feats, class_scores):
        self.linear(feats)
        return self.bag_classifier(feats, class_scores)
    
    
    def train(self):
        
        for i, (inputs, labels) in enumerate(self.trainloader):
                
                inputs = inputs.squeeze(0)
                labels = labels.squeeze(0)
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()

                if self.extractor is not None:
                    with torch.no_grad():
                        feats = self.extractor.compute_features(inputs.float())
                                        
                predictions = self.forward(feats)
                
                loss = self.criterion(predictions, labels.float())
                            
                loss.backward()
            
                self.optimizer.step()
                
                accuracy, precision, recall, f1 = compute_metrics(predictions, labels)
                
                print(f"Iteration {i} - Accurarcy: {accuracy} - {self.loss} Loss: {loss.item()} - Precision: {precision} - Recall: {recall} - F1: {f1}")
            
    def test(self):
                
        for i, (inputs, labels) in enumerate(self.testloader):
                
            inputs = inputs.squeeze(0)
            labels = labels.squeeze(0)
                
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
                        
            if self.extractor is not None:
                with torch.no_grad():
                    feats, class_scores = self.extractor.compute_features(inputs.float())
            
            predictions = self.forward(feats, class_scores)
            
            loss = self.criterion(predictions, labels.float())
            
            accuracy, precision, recall, f1 = compute_metrics(predictions, labels)
                
            print(f"Iteration {i} - Accurarcy: {accuracy} - {self.loss} Loss: {loss.item()} - Precision: {precision} - Recall: {recall} - F1: {f1}")
            
    def run(self, trainset, testset):
        
        self.trainloader = DataLoader(trainset, batch_size=1, shuffle=True, pin_memory=True, num_workers=self.num_workers)
        self.testloader = DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True, num_workers=self.num_workers)
        
        #metrics = compute_metric()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        if self.loss == 'BCE':
            self.criterion = BCEWithLogitsLoss().to(self.device)
        if self.loss == 'focal':
            self.criterion = FocalLoss().to(self.device)
                
        for epoch in range(self.num_epochs):
            print(f"Train Epoch: {epoch}")
            self.train()
            
        for epoch in range(self.num_epochs):
            print(f"Test Epoch: {epoch}")
            self.test()
    