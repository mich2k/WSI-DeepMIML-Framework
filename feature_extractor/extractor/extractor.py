from abc import abstractmethod
from utils import get_device, ImageNetValidationDatasetLoader, accuracy
import torch
from tqdm import tqdm
from collections import Counter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import Linear, Module, Sequential, Identity


class FCLayer(Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = Sequential(Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class Version():
    def __init__(self, version_id, backbone_dimensionality, parent_type = None):
        self.version_id = version_id
        self.backbone_dimensionality = backbone_dimensionality
        self.parent_type = parent_type
    
    def get_dimensionality(self):
        return self.backbone_dimensionality
    
    def get_version_id(self) -> str:
        return self.version_id
    


class FeatureExtractor():

    def __init__(self, dataloader, checkpoint_path, versions_dict:dict, version_id, out_fc_dimensionality):
        self.model = None
        self.device = get_device()

        self.checkpoint_path = checkpoint_path

        self.versions:Version = []
        self.dataloader = dataloader
        self.has_custom_fc = False

        # we might have aggregators which dont need an extra FC layer, hence we can leave the out_fc_dimensionality to -1 (<0)
        # usually this is needed when we bypass the backbone fc by backbone.fc = nn.Identity() and we want to exit with a custom dimensionality

        self._build_versions(versions_dict)

        self._is_version_id_supported(version_id)

        self.version:Version = self._get_version_by_id(version_id)
            
        self.fc = FCLayer(self.version.backbone_dimensionality, out_fc_dimensionality)

        # if not out_fc_dimensionality < 0:
        #     self.has_custom_fc = True
        # else:
        #     self.bypass_backbone_fc()
                  
    def print_summary(self):
        print(self.model)
        
           
    @abstractmethod
    def _load_weights(self, current_version:Version):
        pass
    

    @abstractmethod
    def _build_versions(self):
        pass

    @abstractmethod
    def compute_features(self, x:torch.Tensor, eval):
        pass

    @abstractmethod
    def bypass_backbone_fc(self):
        pass

    
    def _build_versions(self, versions:dict):
        for k,v in versions.items():
            self.versions.append(Version(k, v))
        self._set_parent_in_versions()
    
    def _is_version_id_supported(self, version_id:str):
        flag = False
        for version in self.versions:
            if version_id == version.get_version_id():
                flag = True
        if not flag:
            raise NotImplementedError(f'Invalid version: {version_id} for {self.__class__.__name__} extractor')


    def _get_version_by_id(self, version_id:str) -> Version:
        for version in self.versions:
            if version_id == version.get_version_id():
                return version
        raise NotImplementedError(f'Invalid version: {version_id} for {self.__class__.__name__}')

    def _set_parent_in_versions(self):
        for version in self.versions:
            version.parent_type = self.__class__.__name__

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
