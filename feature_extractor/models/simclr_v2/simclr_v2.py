
from feature_extractor.extractor.extractor import FeatureExtractor, Version    
from .resnet import get_resnet, name_to_params
import torch
from utils import get_device
from torch.nn import Identity


class SimCLRv2(FeatureExtractor):

    def __init__(self, dataloader, checkpoint_path, version_id="r50_1x_sk0"):
        super().__init__(dataloader, checkpoint_path)
        self.device = get_device()
        

        self.set_versions(
            [Version('r50_1x_sk0', 100), Version('r101_1x_sk0', 100), Version('r152_3x_sk1', 100)]
        )

        self.is_version_id_supported(version_id)
                   
        self.version = self.get_version_by_id(version_id)

        self.model, _ = get_resnet(*name_to_params(self.version.get_version_id()))



    def load_weights(self):
        self.model.load_state_dict(torch.load(self.checkpoint_path + self.version.get_version_id() + '.pth')['resnet'])
        self.model = self.model.to(self.device).eval()
    
    def compute_features(self, x:torch.Tensor, eval=False):
        if eval:
            return self.model(x, apply_fc=True)
        else:
            return self.model(x)

    def bypass_backbone_fc(self):
        super().bypass_backbone_fc()
        #self.model.fc = Identity()
