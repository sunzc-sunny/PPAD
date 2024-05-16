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


from .mask_generate_anomaly import generate_anomaly_mask

from .generate_anomaly import fpi_mask_generate_anomaly

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



class MaskZhangTrain(torch.utils.data.Dataset):
    def __init__(self, root, mypseudo=True, train=True, img_size=(224, 224), transforms=None, enable_transform=True, full=True, shot=None, prob=0.5):

        self.data = []
        self.train = train
        self.root = root
        self.img_size = img_size

        self.full = full
        self.shot = shot   
        self.transforms = transforms
        self.load_data()

        if mypseudo == False:

            self.anomaly_source_images = self.load_anomaly_source_images()
            self.anomaly_transform = anomaly_transform(mypseudo=False, anomaly_source_images=self.anomaly_source_images)
        
        else:
            self.anomaly_transform = anomaly_transform()

        mask_1 = torch.zeros((224, 224, 1))
        mask_1[:, 0:112, :] = 1

        mask_2 = torch.zeros((224, 224, 1))
        mask_2[:, 112:, :] = 1

        mask_3 = torch.zeros((224, 224, 1))
        mask_3[0:120, :, :] = 1

        mask_4 = torch.zeros((224, 224, 1))
        mask_4[120:, :, :] = 1

        mask_5 = torch.zeros((224, 224, 1))
        mask_5[:, :, :] = 1

        self.masks = [mask_1, mask_2, mask_3, mask_4, mask_5]
        self.position_names = ["left lung", "right lung", "upper lung", "lower lung", ""]

        
        self.prob = prob


    def load_data(self):
        if self.train:
            if self.shot is None:
                items = os.listdir(os.path.join(self.root, 'normal_256'))
            else:
                items = random.sample(os.listdir(os.path.join(self.root, 'normal_256')), self.shot)


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
    

    def load_anomaly_source_images(self):

        items = os.listdir(os.path.join(self.root, 'normal_256'))

        anomaly_source_images = []
        for item in items:
            img = Image.open(os.path.join(self.root, 'normal_256', item)).convert('L').resize(self.img_size)
            img = np.array(img)
            img = np.stack((img,)*3, axis=-1)
            img = Image.fromarray(img)

            anomaly_source_images.append(img)

        return anomaly_source_images


    def __getitem__(self, index):
        img, label = self.data[index]
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

            return img,  label.long(), mask, position_name

        else:
            position_name = self.position_names
            mask = self.masks
            masks = []
            for m in mask:
                m = m.permute(2,0,1)
                masks.append(m)
            img= self.transforms(img)



            return img,  label.long(), masks, position_name

    def __len__(self):
        return len(self.data)


