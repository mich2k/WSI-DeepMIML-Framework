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
    def __init__(self, dataloader, checkpoint_path, versions, version_id="resnet50", out_dimensionality=100, patch_size=16, avgpool_patchtokens=False, n_last_blocks=4, num_labels=1000):
        super().__init__(dataloader, checkpoint_path, versions, version_id, out_dimensionality)
        
        self.version_id = version_id
        self.patch_size = patch_size
        self.n_last_blocks = n_last_blocks
        self.avgpool_patchtokens = avgpool_patchtokens
        self.dataloader = dataloader
        
        self.model = None        
        
        if self.version_id in vits.__dict__.keys():
                self.model = vits.__dict__[self.architecure](patch_size=patch_size, num_classes=0)
                self.embed_dim = self.model.embed_dim * (n_last_blocks + int(avgpool_patchtokens))
            
        # otherwise, we check if the version_id is in torchvision models
        elif version_id in torchvision_models.__dict__.keys():
            self.model = torchvision_models.__dict__[self.version_id]()
            self.embed_dim = self.model.fc.weight.shape[1]
            self.model.fc = nn.Identity()

        else:
            print(f"Unknow architecture: {self.version_id}")
            sys.exit(1)
            
        self.linear_classifier = LinearClassifier(self.embed_dim, num_labels=num_labels)
        self.linear_classifier = self.linear_classifier.to(self.device)
        
        self._load_weights(apply_fc=True)
        
    
    def _load_weights(self, apply_fc, pretrained_weights='resnet50', checkpoint_key='teacher'):
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
            if eval:
                output = self.linear_classifier(output)
        return output
