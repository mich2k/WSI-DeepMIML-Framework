import torch
import sys
import feature_extractor.models.dino.utility as utility
from torch import nn
from torchvision import models as torchvision_models
from feature_extractor.extractor.extractor import FeatureExtractor, Version
from feature_extractor.models.dino.linear_classifier import LinearClassifier
from feature_extractor.models.dino import vision_transformer as vits

class DINO(FeatureExtractor):
    # architectures: vit or resnet50
    def __init__(self, data_path, checkpoint_path, versions, config):
        super().__init__(data_path, checkpoint_path, versions, config.version_id, versions[config.version_id].dimensionality)
        
        self.version_id = config.version_id
        self.patch_size = config.patch_size
        self.n_last_blocks = versions[self.version_id].n_last_blocks
        self.avgpool_patchtokens = versions[self.version_id].avgpool_patchtokens
        self.num_labels = config.num_labels
        self.custom_weights = config.custom_weights
        self.model = None        
        
        if self.version_id in vits.__dict__.keys():
                self.model = vits.__dict__[self.version_id](patch_size=self.patch_size, num_classes=0)
                self.embed_dim = self.model.embed_dim * (self.n_last_blocks + int(self.avgpool_patchtokens))
            
        # otherwise, we check if the version_id is in torchvision models
        elif self.version_id in torchvision_models.__dict__.keys():
            self.model = torchvision_models.__dict__[self.version_id]()
            self.embed_dim = self.model.fc.weight.shape[1]
            self.model.fc = nn.Identity()
        else:
            print(f"Unknow architecture: {self.version_id}")
            sys.exit(1)
        
        
        self.model = self.model.to(self.device)
        self.linear_classifier = LinearClassifier(self.embed_dim, num_labels=self.num_labels)
        self.linear_classifier = self.linear_classifier.to(self.device)
        
        self._load_weights(self.checkpoint_path, apply_fc=True, custom_weights=self.custom_weights)
        
    
    def _load_weights(self, checkpoint_path, apply_fc, pretrained_weights='resnet50', checkpoint_key='teacher', custom_weights=False, dataset_name="lung", scale_level=20):
        if custom_weights:
            utility.load_custom_weights(self.model, checkpoint_path, checkpoint_key, dataset_name, scale_level)   
        
        else:
            utility.load_pretrained_weights(self.model, pretrained_weights, checkpoint_key, self.version_id, self.patch_size)
            
        if apply_fc:
            utility.load_pretrained_linear_weights(self.linear_classifier, self.version_id, self.patch_size)
        
        print(f"Model {self.version_id} built.")
    
    def compute_features(self, x:torch.Tensor, eval=False):

        with torch.no_grad():
            if "vit" in self.version_id:
                intermediate_output = self.model.get_intermediate_layers(x, self.n_last_blocks)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if self.avgpool_patchtokens:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = self.model(x)
            
            
        return output
