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
    def __init__(self, dataloader, checkpoint_path, architecture="resnet50", patch_size=16, avgpool_patchtokens=False, n_last_blocks=4, num_labels=1000):
        super().__init__(dataloader, checkpoint_path)
        
        self.architecture = architecture
        self.patch_size = patch_size
        self.n_last_blocks = n_last_blocks
        self.avgpool_patchtokens = avgpool_patchtokens
        self.dataloader = dataloader
        
        self.model = None
        
        self.set_versions([Version('resnet50', 2048), Version('vit_base', 768), Version('vit_small', 384)])
        self.is_version_id_supported(architecture)
        self.version = Version(architecture, 100)
        
        
        
        if architecture in vits.__dict__.keys():
                self.model = vits.__dict__[architecture](patch_size=patch_size, num_classes=0)
                self.embed_dim = self.model.embed_dim * (n_last_blocks + int(avgpool_patchtokens))
            
        # otherwise, we check if the architecture is in torchvision models
        elif architecture in torchvision_models.__dict__.keys():
            self.model = torchvision_models.__dict__[architecture]()
            self.embed_dim = self.model.fc.weight.shape[1]
            self.model.fc = nn.Identity()

        else:
            print(f"Unknow architecture: {architecture}")
            sys.exit(1)
            
        self.linear_classifier = LinearClassifier(self.embed_dim, num_labels=num_labels)
        self.linear_classifier = self.linear_classifier.to(self.device)
    
    def load_weights(self, apply_mlp=True, pretrained_weights='resnet50', checkpoint_key='teacher'):
        utility.load_pretrained_weights(self.model, pretrained_weights, checkpoint_key, self.architecture, self.patch_size)
        if apply_mlp:
            utility.load_pretrained_linear_weights(self.linear_classifier, self.architecture, self.patch_size)
        print(f"Model {self.architecture} built.")
    
    def compute_features(self, x:torch.Tensor, eval=False):
        with torch.no_grad():
            if "vit" in self.architecture:
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
