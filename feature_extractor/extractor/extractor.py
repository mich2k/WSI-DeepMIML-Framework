from abc import abstractmethod
from utils import get_device, ImageNetValidationDatasetLoader, accuracy
import torch
from tqdm import tqdm
from collections import Counter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class FeatureExtractor():
    
    def __init__(self, dataloader):
        self.model = None
        self.version = "default"
        self.versions = []
        self.device=get_device()
        self.dataloader = dataloader
                  
    def print_summary(self):
        print(self.model)
        
           
    @abstractmethod
    def load_weights(self, apply_mlp=False):
        pass
    
    @abstractmethod
    def compute_features(self, x:torch.Tensor, eval):
        pass
    
    @torch.no_grad()
    def benchmark(self, val_path, n_samples=3, batch=380):
        device = get_device()
        dataset = ImageNetValidationDatasetLoader(val_path)

        data_loader = DataLoader(dataset, batch_size=batch, shuffle=False, pin_memory=True, num_workers=11)

        self.model = self.model.to(device).eval()
        
        preds = []
        target = []

        counter = 0

        for images, labels in tqdm(data_loader):

            if counter == n_samples and n_samples != 0: # since dataloader loads/samples batch_size at time we can early stop (easiest approach)
                break
            
            if batch == 1:
                pillow_image = transforms.ToPILImage()(images[0])
            
            p = self.compute_features(images.to(device), eval=True)
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
