import torch
import utils
import torchvision.models as models

class SimCLRv1:
    def __init__(self):
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Identity()
        self.model.load_state_dict(torch.load('feature_extractor/models/simclrv1.pth'))
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        self.model.to(self.device)