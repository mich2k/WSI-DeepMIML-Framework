import torch
import sys
import utils
from torch import nn
from torchvision import models as torchvision_models
from feature_extractor.extractor.extractor import FeatureExtractor
from feature_extractor.models.dino.linear_classifier import LinearClassifier
from feature_extractor.models.dino import vision_transformer as vits


class DINO(FeatureExtractor):
    # architectures: vit, resnet50 or xcit
    def __init__(self, dataloader, architecture="resnet50", patch_size=16, avgpool_patchtokens=False, n_last_blocks=4, num_labels=1000, gpu=0):
        super().__init__(dataloader)
        self.architecture = architecture
        self.patch_size = patch_size
        self.n_last_blocks = n_last_blocks
        self.avgpool_patchtokens = avgpool_patchtokens
        self.model = None
        self.dataloader = dataloader
        
        if architecture in vits.__dict__.keys():
                self.model = vits.__dict__[architecture](patch_size=patch_size, num_classes=0)
                self.embed_dim = self.model.embed_dim * (n_last_blocks + int(avgpool_patchtokens))
        
        # if the network is a XCiT
        elif "xcit" in architecture:
            self.model = torch.hub.load('facebookresearch/xcit:main', architecture, num_classes=0)
            self.embed_dim = self.model.embed_dim
            
        # otherwise, we check if the architecture is in torchvision models
        elif architecture in torchvision_models.__dict__.keys():
            self.model = torchvision_models.__dict__[architecture]()
            self.embed_dim = self.model.fc.weight.shape[1]
            self.model.fc = nn.Identity()
        else:
            print(f"Unknow architecture: {architecture}")
            sys.exit(1)
            
        self.linear_classifier = LinearClassifier(self.embed_dim, num_labels=num_labels)
        self.linear_classifier = self.linear_classifier.cuda()
        self.linear_classifier = nn.parallel.DistributedDataParallel(self.linear_classifier, device_ids=[gpu])

    
    def load_weights(self, apply_mlp=True, pretrained_weights='resnet50', checkpoint_key='teacher'):
        utils.load_pretrained_weights(self.model, pretrained_weights, checkpoint_key, self.architecture, self.patch_size)
        if apply_mlp:
            utils.load_pretrained_linear_weights(self.linear_classifier, self.architecture, self.patch_size)
        print(f"Model {self.architecture} built.")
    
    def compute_features(self):
        return super().compute_features()
    
    @torch.no_grad()
    def validate_network(self, val_loader):
        self.model.cuda()
        self.model.eval()
        self.linear_classifier.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'
        for inp, target in metric_logger.log_every(val_loader, 20, header):
            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # forward
            with torch.no_grad():
                if "vit" in self.architecture:
                    intermediate_output = self.model.get_intermediate_layers(inp, self.n_last_blocks)
                    output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                    if self.avgpool_patchtokens:
                        output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                        output = output.reshape(output.shape[0], -1)
                else:
                    output = self.model(inp)
            output = self.linear_classifier(output)
            loss = nn.CrossEntropyLoss()(output, target)

            if self.linear_classifier.module.num_labels >= 5:
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            else:
                acc1, = utils.accuracy(output, target, topk=(1,))

            batch_size = inp.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            if self.linear_classifier.module.num_labels >= 5:
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        if self.linear_classifier.module.num_labels >= 5:
            print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
        else:
            print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, losses=metric_logger.loss))
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
