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

class NihCxr(torch.utils.data.Dataset):
    def __init__(self, root="/data/sunzc/NihCxr", train=True, img_size=(256, 256), normalize=False, enable_transform=True, full=True):

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
            transforms.Resize(size=224, interpolation=Image.BICUBIC),  # 注意：需要导入 Image 模块
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

        self.transforms = transform
        self.load_data()


    def load_data(self):
        if self.train:
            items = os.listdir(os.path.join(self.root, 'norm_MaleAdultPA_train_curated_list'))
            for item in items:
                img = Image.open(os.path.join(self.root, 'norm_MaleAdultPA_train_curated_list', item)).convert('L').resize(self.img_size)
                # 将图像拓展为3通道
                img = np.array(img)
                img = np.stack((img,)*3, axis=-1)
                img = Image.fromarray(img)

                self.data.append((img, 0))

            items = os.listdir(os.path.join(self.root, 'norm_FemaleAdultPA_train_curated_list'))
            for item in items:
                img = Image.open(os.path.join(self.root, 'norm_FemaleAdultPA_train_curated_list', item)).convert('L').resize(self.img_size)
                # 将图像拓展为3通道
                img = np.array(img)
                img = np.stack((img,)*3, axis=-1)
                img = Image.fromarray(img)

                self.data.append((img, 0))

        if not self.train:
            items = os.listdir(os.path.join(self.root, 'norm_FemaleAdultPA_test_curated_list'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                img = Image.open(os.path.join(self.root, 'norm_FemaleAdultPA_test_curated_list', item)).convert('L').resize(self.img_size)
                img = np.array(img)
                img = np.stack((img,)*3, axis=-1)
                img = Image.fromarray(img)

                self.data.append((img, 0))


            items = os.listdir(os.path.join(self.root, 'norm_MaleAdultPA_test_curated_list'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                img = Image.open(os.path.join(self.root, 'norm_MaleAdultPA_test_curated_list', item)).convert('L').resize(self.img_size)
                img = np.array(img)
                img = np.stack((img,)*3, axis=-1)
                img = Image.fromarray(img)

                self.data.append((img, 0))


            items = os.listdir(os.path.join(self.root, 'anomaly_FemaleAdultPA_test_curated_list'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                img = Image.open(os.path.join(self.root, 'anomaly_FemaleAdultPA_test_curated_list', item)).convert('L').resize(self.img_size)
                img = np.array(img)
                img = np.stack((img,)*3, axis=-1)
                img = Image.fromarray(img)

                self.data.append((img, 1))


            items = os.listdir(os.path.join(self.root, 'anomaly_MaleAdultPA_test_curated_list'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                img = Image.open(os.path.join(self.root, 'anomaly_MaleAdultPA_test_curated_list', item)).convert('L').resize(self.img_size)
                img = np.array(img)
                img = np.stack((img,)*3, axis=-1)
                img = Image.fromarray(img)

                self.data.append((img, 1))
        

        print('%d data loaded from: %s' % (len(self.data), self.root))
    

    def __getitem__(self, index):
        img, label = self.data[index]
        # print(img.size)
        img = self.transforms(img)
        if self.normalize:
            img -= self.mean
            img /= self.std
        return img, (torch.zeros((1,)) + label).long()

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    dataset = NihCxr('/data/sunzc/NihCxr', train=False)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for i, (img, label) in enumerate(trainloader):
        print(img.shape)

        show_image = img.squeeze().numpy()
        show_image = show_image.transpose(1,2,0)
        print(label)
        plt.imshow(show_image, cmap='gray')
        plt.show()
        # break
        if i > 10:
            break