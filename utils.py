import os
from PIL import Image
import numpy as np
import torch.backends.mps as mps
import torch.cuda as cuda
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from munch import Munch, munchify
import yaml
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset

class DiffInfiniteDataset(Dataset):
    def __init__(self, data_path, use_strategy=True, stop_at=0):
        self.labels = []
        self.images = []
        
        self.labels_csv = pd.read_csv(data_path + "annotator.csv")
        data_path = data_path + "patched_presplit_out/patches/"
        
        # Load Images
        for folder in tqdm(os.listdir(data_path)):
            if len(self.images) == stop_at and stop_at != 0:
                break
            
            bag = []
            
            for image_name in os.listdir(os.path.join(data_path, folder)):
                if len(self.images) == stop_at and stop_at != 0:
                    break
                
                image_path = os.path.join(data_path, folder, image_name)
                
                image = Image.open(image_path)
                image = np.array(image)
                image = image.reshape(image.shape[2], image.shape[0], image.shape[1])
                
                bag.append(image)
                
            if bag:  # Ensure the bag is not empty
                self.images.append(np.array(bag))
            
        # Load labels from csv file
        labels_column = ['Unknown', 'Carcinoma', 'Necrosis', 'Tumor_Stroma', 'Others'] if not use_strategy else ['ABS_Unknown', 'ABS_Carcinoma', 'ABS_Necrosis', 'ABS_Tumor_Stroma', 'ABS_Others']
        
        for index, row in self.labels_csv.iterrows():
            if len(self.labels) == stop_at and stop_at != 0:
                break
            
            l = list(row[labels_column])
            self.labels.append(np.array(l))
        
        # Convert lists to numpy arrays
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
    def normalize(self):
        self.images = np.array([bag / 255.0 for bag in self.images])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
    def get_data(self):
        return self.images, self.labels
   
        
class MultiLabelDataset(Dataset): 
    def __init__(self, stop_at = 0):
        self.labels = []
        self.images = []
        labels_path= "datasets/multilabel_modified/labels.csv" 
        images_path= "datasets/multilabel_modified/images"
        df = pd.read_csv(labels_path)
        
        column_names = df.columns.tolist()

        df.rename(columns={column_names[1]: 'Classes'}, inplace=True)
        
        for index, row in df.iterrows():
            l = list(row.iloc[-10:])
            self.labels.append(l)
            
            if len(self.labels) == stop_at and stop_at != 0:
                break
            
        self.labels= np.array(self.labels)
        
        image_names=df['Image_Name'].tolist()
           
        for img_name in tqdm(image_names):
            image_path = os.path.join(images_path, img_name) 
            image = Image.open(image_path)
            image = np.array(image)
            if len(image.shape) == 2:
                copied_images = [image.copy() for _ in range(3)] 
                image = np.stack(copied_images, axis=-1)
            
            image = cv2.resize(image, (32,32))
            self.images.append(image.reshape((3,32,32)))
            
            if len(self.images) == stop_at and stop_at != 0:
                break
            
            
        self.images = np.array(self.images)
            
        self.normalize()
        
    def normalize(self):
        self.images = self.images/255.0
        
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
    def get_data(self):
        return self.images, self.labels

class ImageNetValidationDatasetLoader(Dataset):
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

def get_device():
    if mps.is_available():
        return 'mps'
    return 'cuda' if cuda.is_available() else 'cpu'


def load_config(config_path) -> Munch:
    try:
        with open(config_path, 'r') as configuration_fstream:
            yaml_dict = yaml.safe_load(configuration_fstream)
            return munchify(yaml_dict)
    except FileNotFoundError:
        print("Error with config file")
        raise FileNotFoundError


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

def get_sample_batch(n, channels=3):
    inputs = []
    for i in range(n):
        inputs.append(torch.randn((channels,255,255)))
    return torch.stack(inputs).to(get_device())

def binary_relevance_transformation(images, labels, nested_array=True, get_bags = False, stop_at=0):
        
        # Apply binary relevance transformation
        mi_features = []
        mi_labels = []
        bags = []
        
        
        batch_size = images.shape[0]
        labels = labels.reshape(batch_size, -1)
        
        # For each element in the batch
        for i in range(batch_size):
            
            if len(mi_features) == stop_at and stop_at != 0:
                break
            
            bag = []
            
            # For each label related to a single bag
            for j in range(labels.shape[1]):
                
                mi_features.append(images[i])
                if nested_array:
                    mi_labels.append([labels[i][j]])
                else:
                    mi_labels.append(labels[i][j])
                    
                if get_bags:
                    bag.append(i + j)
                
                bags.append(np.array(bag))
        
        if get_bags:
            return np.array(mi_features), np.array(mi_labels), bags
        
        return np.array(mi_features), np.array(mi_labels)
    

def compute_metrics(pred, target):
    
    
    pred = torch.sigmoid(pred)

    
    # Check if it is a numpy array
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach().numpy()
        
    if isinstance(target, torch.Tensor):
        target = target.cpu().detach().numpy()
        
        
    
    threshold = 0.6
    
    # check if target has continous values or integers
    if 'float' not in target.dtype.name:
        pred = (pred > threshold).astype(int)
        
            
    # Compute precision, recall, f1-score
    accuracy = accuracy_score(target, pred)
    precision = precision_score(target, pred, average='weighted', zero_division=0)
    recall = recall_score(target, pred, average='weighted',  zero_division=0)
    f1_scr = f1_score(target, pred, average='weighted',  zero_division=0)
    
    return accuracy, precision, recall, f1_scr