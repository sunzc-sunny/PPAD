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

from .mask_generate_anomaly import generate_anomaly_mask


class anomaly_transform(object):
    def __init__(self, mypseudo=True, anomaly_source_images=None):
        self.anomaly_weight = [-0.999, -0.99, 2, 3]
        self.anomaly_source_images = anomaly_source_images
        self.mypseudo = mypseudo

    def __call__(self, image, mask):
        image = np.array(image)
        if self.mypseudo:
            random_choice = random.choice(self.anomaly_weight)

            new_img = generate_anomaly_mask(image, random_choice, mask)
            new_img = Image.fromarray(new_img)

            return new_img, torch.Tensor([0,1])
        else:
            
            anomaly_source_image = random.choice(self.anomaly_source_images)
            new_img = fpi_mask_generate_anomaly(image, anomaly_source_image, mask)
            new_img = Image.fromarray(new_img)

            return new_img, torch.Tensor([0,1])


class MaskVinCXR(torch.utils.data.Dataset):
    def __init__(self, root="/data/sunzc/Med-AD_v1_D/VinCXR", train=True, img_size=(256, 256), normalize=False, transforms=None, full=True, shot=None, prob=0.5):

        self.data = []
        self.train = train
        self.root = root
        self.normalize = normalize
        self.img_size = img_size
        self.mean = 0.1307
        self.std = 0.3081
        self.full = full
        self.shot = shot
        self.prob = prob
        self.train_list_normal = []
        self.train_list_abnormal = []
        self.test_list_normal = []
        self.test_list_abnormal = []


        self.transforms = transforms
        self.anomaly_transform = anomaly_transform()

        self.load_json()
        self.load_data()

        mask_1 = torch.zeros((224, 224, 1))
        mask_1[:, 0:112, :] = 1
        # mask_1 = mask_1.permute(2,0,1).unsqueeze(0)

        mask_2 = torch.zeros((224, 224, 1))
        mask_2[:, 112:, :] = 1
        # mask_2 = mask_2.permute(2,0,1).unsqueeze(0)

        mask_3 = torch.zeros((224, 224, 1))
        mask_3[0:120, :, :] = 1
        # mask_3 = mask_3.permute(2,0,1).unsqueeze(0)

        mask_4 = torch.zeros((224, 224, 1))
        mask_4[120:, :, :] = 1
        # mask_4 = mask_4.permute(2,0,1).unsqueeze(0)

        mask_5 = torch.zeros((224, 224, 1))
        mask_5[:, :, :] = 1
        # mask_5 = mask_5.permute(2,0,1).unsqueeze(0)

        self.masks = [mask_1, mask_2, mask_3, mask_4, mask_5]
        self.position_names = ["left lung", "right lung", "upper lung", "lower lung", ""]
        # print(len(self.train_list_normal), len(self.train_list_abnormal), len(self.test_list_normal), len(self.test_list_abnormal))
        # self.masks = [mask_5, mask_5, mask_5, mask_5, mask_5]
        # self.position_names = ["", "", "", "", ""]

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
            # items = os.listdir(os.path.join(self.root, 'images'))
            items = random.sample(self.train_list_normal, int(self.shot))

            for item in items:
                img = Image.open(os.path.join(self.root, 'images', item)).convert('L').resize(self.img_size)
                # 将图像拓展为3通道
                img = np.array(img)
                img = np.stack((img,)*3, axis=-1)
                img = Image.fromarray(img)

                self.data.append((img, torch.Tensor([1,0])))


        if not self.train:
            for idx, item in enumerate(self.test_list_normal):

                img = Image.open(os.path.join(self.root, 'images', item)).convert('L').resize(self.img_size)
                img = np.array(img)
                img = np.stack((img,)*3, axis=-1)
                img = Image.fromarray(img)

                self.data.append((img, torch.Tensor([1,0])))
                # self.data.append((Image.open(os.path.join(self.root, 'normal_256', item)).resize(self.img_size), 0))

            for idx, item in enumerate(self.test_list_abnormal):

                img = Image.open(os.path.join(self.root, 'images', item)).convert('L').resize(self.img_size)
                img = np.array(img)
                img = np.stack((img,)*3, axis=-1)
                img = Image.fromarray(img)
                self.data.append((img, torch.Tensor([0,1])))
                # self.data.append((Image.open(os.path.join(self.root, 'pneumonia_256', item)).resize(self.img_size), 1))
        
        

        print('%d data loaded from: %s' % (len(self.data), self.root))
    

    def __getitem__(self, index):
        img, label = self.data[index]
        # print(img.size)
        if self.train == True:
            ran = random.random()
            if ran >= self.prob:
                random_index = random.randint(0, 4)
                mask = self.masks[random_index]
                position_name = self.position_names[random_index]
                img, label = self.anomaly_transform(img, mask)
            else:
                random_index = random.randint(0, 4)
                position_name = self.position_names[random_index]
                mask = self.masks[random_index]

                label = torch.Tensor([1,0])

            mask = mask.permute(2,0,1)
            img= self.transforms(img)

            # print(img.shape, label.shape, mask, position_name)
            return img,  label.long(), mask, position_name

        else:
            position_name = self.position_names
            mask = self.masks
            masks = []
            for m in mask:
                m = m.permute(2,0,1)
                masks.append(m)
            img= self.transforms(img)



            # print(img.shape, label.shape, mask, position_name)
            return img,  label.long(), masks, position_name


    def __len__(self):
        return len(self.data)
