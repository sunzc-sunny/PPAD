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
import json

class VinCXR(torch.utils.data.Dataset):
    def __init__(self, root="/data/sunzc/Med-AD_v1_D/VinCXR", train=True, img_size=(256, 256), normalize=False, enable_transform=True, full=True):

        self.data = []
        self.train = train
        self.root = root
        self.normalize = normalize
        self.img_size = img_size
        self.mean = 0.1307
        self.std = 0.3081
        self.full = full
        self.train_list_normal = []
        self.train_list_abnormal = []
        self.test_list_normal = []
        self.test_list_abnormal = []


        transform = transforms.Compose([
            transforms.Resize(size=224, interpolation=Image.BICUBIC), 
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.39799, 0.39799, 0.39799], std=[0.32721349, 0.32721349, 0.32721349])
        ])

        self.transforms = transform

        self.load_json()

        self.load_data()

    def load_json(self):
        json_files = self.root + '/data.json'
        with open(json_files, 'r') as f:
            data = json.load(f)
            train_list = data['train']
            test_list = data['test']

            self.train_list_normal = train_list["0"]
            self.train_list_abnormal = train_list["unlabeled"]
            self.test_list_normal = test_list["0"]
            self.test_list_abnormal = test_list["1"]




    def load_data(self):
        if self.train:
            items = os.listdir(os.path.join(self.root, 'images'))
            for item in self.train_list_normal:
                img = Image.open(os.path.join(self.root, 'images', item)).convert('L').resize(self.img_size)
                img = np.array(img)
                img = np.stack((img,)*3, axis=-1)
                img = Image.fromarray(img)

                self.data.append((img, 0))

            for item in self.train_list_normal:
                img = Image.open(os.path.join(self.root, 'images', item)).convert('L').resize(self.img_size)
                img = np.array(img)
                img = np.stack((img,)*3, axis=-1)
                img = Image.fromarray(img)

                self.data.append((img, 1))

        if not self.train:
            items = os.listdir(os.path.join(self.root, 'images'))
            for idx, item in enumerate(self.test_list_normal):
                if not self.full and idx > 9:
                    break
                img = Image.open(os.path.join(self.root, 'images', item)).convert('L').resize(self.img_size)
                img = np.array(img)
                img = np.stack((img,)*3, axis=-1)
                img = Image.fromarray(img)

                self.data.append((img, 0))
            items = os.listdir(os.path.join(self.root, 'images'))
            for idx, item in enumerate(self.test_list_abnormal):
                if not self.full and idx > 9:
                    break
                img = Image.open(os.path.join(self.root, 'images', item)).convert('L').resize(self.img_size)
                img = np.array(img)
                img = np.stack((img,)*3, axis=-1)
                img = Image.fromarray(img)
                self.data.append((img, 1))
        
        

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

