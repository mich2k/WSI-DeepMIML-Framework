import torch
from torchvision.models import resnet50, get_model_weights
from torchvision.models.resnet import ResNet, Bottleneck
import torchvision
from pprint import pprint

modellino = ResNet(block=Bottleneck, layers=[3, 4, 6, 3])


res = torch.load('ext/DINO/dino_resnet50_pretrain.pth')
res1 = torch.load('ext/DINO/dino_resnet50_pretrain_full_checkpoint.pth')
res2 = torch.load('ext/DINO/dino_resnet50_linearweights.pth')
print(len(res1['student']))
print(len(res))
print(res2.keys())
print(len(res2['state_dict']))


model = resnet50(weights=None)
model.load_state_dict(res1['student'], strict=False)

#print(torchvision.models.__dict__['resnet50']())

#with open('out_dino.json', 'w') as f:
#    pprint(res1, stream=f)