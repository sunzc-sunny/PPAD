import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
from matplotlib import pyplot as plt
import random


from .my_generate_anomaly import generate_anomaly



class Zhang(torch.utils.data.Dataset):
    def __init__(self, root, train=True, img_size=(256, 256), normalize=False, enable_transform=True, full=True, pseudo=False, shot=None):

        self.data = []
        self.train = train
        self.root = root
        self.normalize = normalize
        self.img_size = img_size
        self.mean = 0.1307
        self.std = 0.3081
        self.full = full
        self.pseudo = pseudo
        self.shot = shot   
 
        self.load_data()
        transform = transforms.Compose([
            transforms.Resize(size=224, interpolation=Image.BICUBIC),  
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.39799, 0.39799, 0.39799], std=[0.32721349, 0.32721349, 0.32721349])
        ])
        self.transforms = transform



    def load_data(self):
        if self.train:
            if self.shot is None:
                items = os.listdir(os.path.join(self.root, 'normal_256'))
            else:
                items = os.listdir(os.path.join(self.root, 'normal_256'))[:self.shot]

            for item in items:
                img = Image.open(os.path.join(self.root, 'normal_256', item)).convert('L').resize(self.img_size)
                img = np.array(img)
                img = np.stack((img,)*3, axis=-1)
                img = Image.fromarray(img)

                self.data.append((img, torch.Tensor([1,0])))




        if not self.train:
            if self.shot is None:
                items = os.listdir(os.path.join(self.root, 'normal_256'))
            else:
                items = os.listdir(os.path.join(self.root, 'normal_256'))[:self.shot]
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                img = Image.open(os.path.join(self.root, 'normal_256', item)).convert('L').resize(self.img_size)
                img = np.array(img)
                img = np.stack((img,)*3, axis=-1)
                img = Image.fromarray(img)

                self.data.append((img, torch.Tensor([1,0])))

            if self.shot is None:
                items = os.listdir(os.path.join(self.root, 'pneumonia_256'))
            else:
                items = os.listdir(os.path.join(self.root, 'pneumonia_256'))[:self.shot]
            
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                img = Image.open(os.path.join(self.root, 'pneumonia_256', item)).convert('L').resize(self.img_size)
                img = np.array(img)
                img = np.stack((img,)*3, axis=-1)
                img = Image.fromarray(img)
                self.data.append((img, torch.Tensor([0,1])))
        

        print('%d data loaded from: %s' % (len(self.data), self.root))
    

    def __getitem__(self, index):
        img, label = self.data[index]

        
        img = self.transforms(img)
        if self.normalize:
            img -= self.mean
            img /= self.std
        return img, (torch.zeros((1,)) + label).long()

    def __len__(self):
        return len(self.data)




