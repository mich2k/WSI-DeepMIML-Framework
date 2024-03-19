import os
import argparse
from collections import Counter

import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from resnet import get_resnet, name_to_params
torch.set_autocast_enabled(True)

class ImagenetValidationDataset(Dataset):
    def __init__(self, val_path):
        super().__init__()
        self.val_path = val_path
        self.labels = []
        self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        with open('/home/mich/Scrivania/WSI-DeepMIML-Framework/ext/SimCLRv2-Pytorch/ILSVRC2012_img_val/val.txt') as f:
            self.labels = [int(l.strip().split(' ')[1]) - 1 for l in f.readlines()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.val_path, f'ILSVRC2012_val_{item + 1:08d}.JPEG')).convert('RGB')
        return self.transform(img), self.labels[item]


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().cpu()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum().item()
        res.append(correct_k)
    return res


@torch.no_grad()
def run(pth_path, val_path, n_samples, batch=380): # default for debug
    device = 'cuda'
    dataset = ImagenetValidationDataset(val_path)
    data_loader = DataLoader(dataset, batch_size=batch, shuffle=False, pin_memory=True, num_workers=12)
    model, _ = get_resnet(*name_to_params(pth_path))
    model.load_state_dict(torch.load(pth_path)['resnet'])
    model = model.to(device).eval()
    preds = []
    target = []
    
    zero=0
    counter=0
    for images, labels in tqdm(data_loader):
        #print(images)
        if counter == n_samples and n_samples != 0: # since dataloader loads/samples batch_size at time we can early stop (easiest approach)
            pass
        _, pred = model(images.to(device), apply_fc=True).topk(1, dim=1)    # pred shape: (380, 1), (batch size, 1)
        
        preds.append(pred.squeeze(1).cpu())
        #print(pred)
        #predcpu= pred.squeeze(1).cpu().numpy()
        if(pred.sum() == 0):    # dim=None reduces all dimensions
            print(f"pred is zero @{zero}")
            zero+=1
        
        target.append(labels)
        counter+=1
    
    #print(p, t)    # throws error! UnboundLocalError: cannot access local variable 'p' where it is not associated with a value
    p = torch.cat(preds).numpy()
    t = torch.cat(target).numpy()
    all_counters = [Counter() for i in range(1000)]
    for i in range(n_samples*batch):    # now respecting the n_samples
        all_counters[t[i]][p[i]] += 1
    total_correct = 0
    for i in range(1000):
        try:
            total_correct += all_counters[i].most_common(1)[0][1] # most_common returns a list, if counter is empty then we get index out of range with []
        except IndexError:
            print(f"Empty counter at index {i}")
            pass
    print(f'ACC: {total_correct / (n_samples*batch) * 100}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimCLR verifier')
    parser.add_argument('-pth_path', default='ext/SimCLRv2-Pytorch/r50_1x_sk0.pth', type=str, help='path of the input checkpoint file')
    parser.add_argument('-val_path', type=str, help='path of the validation dataset', default="ext/SimCLRv2-Pytorch/ILSVRC2012_img_val/")
    
    # n_samples = 0 takes the whole dataloader, as batch_size*n_sample,
    #   for instance being our dataset 50k it will try with 50000/380=131.579
    #   so it will try with 132 batches, thus reaching 132*380=50160 samples
     
    parser.add_argument('-n_samples', default=3, type=int, help='number of batch_size samples load')
    parser.add_argument('-batch_size', default=380, type=int, help='batch size')

    args = parser.parse_args()
    
    # fix:
    # https://stackoverflow.com/questions/66857471/cuda-initialization-cuda-unknown-error-this-may-be-due-to-an-incorrectly-set
    
    if not torch.cuda.is_available():
        print('No CUDA device found')
    elif not torch.backends.mps.is_available():
        print('No MPS device found')  
    else:
        exit(1)  
        
    print(f"using batch size of: {args.batch_size}")
    print(f"sample size of: {args.n_samples}")

    run(args.pth_path, args.val_path, args.n_samples, args.batch_size)
