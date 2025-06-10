# from constants import *
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms 
from PIL import Image
from os.path import join as jn
import os
import numpy as np
# DD


# DD. UNETDATASET
# unetDataset = UNetDataset()
# interp. an object that contains a representation of the images and masks to enter the dataset
class UNetDataset(Dataset):
    def __init__(self,path_images,path_masks):
        # self.root = path 
        self.imagesPath = path_images
        self.masksPath = path_masks
        self.images = sorted([jn(self.imagesPath,i) for i in os.listdir(self.imagesPath)])
        self.masks = sorted([jn(self.masksPath,i) for i in os.listdir(self.masksPath)])
        # I've noticed the DataLoader class transforms the data into tensors, making the similar operation below optional (I think)
        self.transform = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()
        ])
    
    def __getitem__(self,index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")
        
        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.images)
    
        
        

