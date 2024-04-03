import os
import argparse
from collections import Counter
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from resnet import get_resnet, name_to_params

def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class ImagenetValidationDataset(Dataset):
    def __init__(self, val_path):
        super().__init__()
        self.val_path = val_path
        self.labels = []
        self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        with open(val_path + "val.txt") as f:
            self.labels = [int(l.strip().split(' ')[1]) for l in f.readlines()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.val_path, f'ILSVRC2012_val_{item + 1:08d}.JPEG')).convert('RGB')
        return self.transform(img), self.labels[item]


def count_similar(arr1, arr2, offset=1):
    return np.sum(np.abs(arr1 - arr2) <= offset)

def accuracy(pred, target, topk=(1,)):
    #pred = pred.t().cpu()
    
    #correct = np.intersect1d(pred, target).shape[0]

    correct = np.sum(pred == target)

    #correct = (pred == target).sum().item()

    #correct = pred.eq(target[0].view(1, -1).expand_as(pred))
    #how_many_had_zero = (target[0].shape[0] - target[0].nonzero().shape[0])
    correct_by_offset = count_similar(pred, target, 1)
    return correct, correct_by_offset


@torch.no_grad()
def run(pth_path, val_path, n_samples, batch=380): # default for debug
    device = get_device()
    dataset = ImagenetValidationDataset(val_path)
    
    data_loader = DataLoader(dataset, batch_size=batch, shuffle=False, pin_memory=True, num_workers=11)
    
    model, _ = get_resnet(*name_to_params(pth_path))
    model.load_state_dict(torch.load(pth_path)['resnet'])

    
    model = model.to(device).eval()
    preds = []
    target = []
    
    counter = 0
    
    for images, labels in tqdm(data_loader):

        if counter == n_samples and n_samples != 0: # since dataloader loads/samples batch_size at time we can early stop (easiest approach)
            break
        
        if batch == 1:
            pillow_image = transforms.ToPILImage()(images[0])
        
        p = model(images.to(device), apply_fc=True)
        _, pred = p.topk(1, dim=1)
        #_, pred = model(images.to(device), apply_fc=True).topk(1, dim=1)    # pred shape: (380, 1), (batch size, 1)
        
        preds.append(pred.squeeze(1).cpu())
        #print(pred)
        
        target.append(labels)
        counter+=1
    
    #print(p, t)    # throws error! UnboundLocalError: cannot access local variable 'p' where it is not associated with a value
    p = torch.cat(preds).numpy()
    t = torch.cat(target).numpy()
    
    
    # I make 1000 counters, one for each class
    # I index at the t[i] counter, i-th is the image, meanwhile t[i] is its class
    # all_counters[t[i]] hence is the Counter for that specific class
    
    # I then increment the counter at the p[i] index, which is the prediction for the i-th image
    # if [t[i]] is the correct class, lets say 65, I want that p[i], the prediction for the i-th image, to be 65 too
    # all_counters[65][65] -> we want this ideally to match for all the images which are in the class 65
    
    
    all_counters = [Counter() for i in range(1000)]
    for i in range(n_samples*batch):    # now respecting the n_samples
        all_counters[t[i]][p[i]] += 1
    total_correct = 0
    for i in range(1000):
        try:
            # I pick the i-th counter with all_counters[i]
            # then i retrieve the top-k common predicted classes for that class, in this case the top 1
            # this returns me a list of the top-k tuples, we have k=1 hence will be a list with one tuple inside, with [0] i retrieve the only tuple in the list, [(0,3)]
            # now we get our tuple, in the following fashion (0,3), this means for the class i, i predicted 0 three times (hope that i=0)
            # with the [1] i get the second element of that tuple, hence 3 which is the number of predictions for the most common class for the i-th class
            
            total_correct += all_counters[i].most_common(1)[0][1] 
        except IndexError:
            #print(f"Empty counter at index {i}") # most_common returns a list, if counter is empty then we get index out of range with []
            pass    
    
    correct, correct_by_offset = accuracy(p, t)
    
    print('\n----------------------------------------')
    print(f"Accuracy {correct}/{n_samples*batch} -> {str(correct/(n_samples*batch)*100)[:5]}%")
    print(f"Accuracy by offset {correct_by_offset}/{n_samples*batch} -> {str(correct_by_offset/(n_samples*batch)*100)[:5]}%")
    
    print(f'ACC: {total_correct}, {total_correct / (n_samples*batch) * 100}')
    print('----------------------------------------\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimCLR verifier')
    parser.add_argument('-pth_path', default='checkpoints/r101_1x_sk0.pth', type=str, help='path of the input checkpoint file')
    parser.add_argument('-val_path',  default="datasets/ILSVRC2012_img_val/", type=str, help='path of the validation dataset')
    
    # n_samples = 0 takes the whole dataloader, as batch_size*n_sample,
    #   for instance being our dataset 50k it will try with 50000/380=131.579
    #   so it will try with 132 batches, thus reaching 132*380=50160 samples
     
    parser.add_argument('-n_samples', default=5, type=int, help='number of batch_size samples load')
    parser.add_argument('-batch_size', default=128, type=int, help='batch size')

    args = parser.parse_args()
    
    print('\n----------------------------------------')
    
    # fix:
    # https://stackoverflow.com/questions/66857471/cuda-initialization-cuda-unknown-error-this-may-be-due-to-an-incorrectly-set
    
    if not torch.cuda.is_available():
        print('No CUDA device found')
    elif not torch.backends.mps.is_available():
        print('No MPS device found')  
    else:
        exit(1)  
        
    print(f"using batch size of: {args.batch_size}")
    
    if(args.n_samples == 0):
        print(f"using whole dataset")
    else:
        print(f"sample size of: {args.n_samples}")
    
    print('----------------------------------------\n')


    run(args.pth_path, args.val_path, args.n_samples, args.batch_size)