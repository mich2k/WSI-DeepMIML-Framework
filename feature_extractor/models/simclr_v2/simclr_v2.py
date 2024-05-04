
from feature_extractor.extractor.extractor import FeatureExtractor, Version    
from .resnet import get_resnet, name_to_params
import torch
from torch.nn import Identity


class SimCLRv2(FeatureExtractor):

    def __init__(self, data_path, checkpoint_path, versions, config):
        super().__init__(data_path, checkpoint_path, versions, config.version_id, versions[config.version_id].dimensionality)
                           
        self.model, _ = get_resnet(*name_to_params(self.version.get_version_id()))
        self.linear_classifier = self.model.fc
        self._load_weights(self.version)
        self.embed_dim = 2048
        self.num_labels = config.num_labels



    def _load_weights(self, current_version:Version):
        self.model.load_state_dict(torch.load(self.checkpoint_path + current_version.get_version_id() + '.pth')['resnet'])
        self.model = self.model.to(self.device).eval()
    
    def compute_features(self, x:torch.Tensor, eval=False):
        if eval:
            return self.model(x, apply_fc=True)
        else:
            return self.model(x)

    def bypass_backbone_fc(self):
        super().bypass_backbone_fc()
        self.model.fc = Identity()
