import numpy as np
import torch
import random
from torchvision import transforms


def generate_anomaly(anomaly_source_image, image, core_percent=0.8):

    random_brightness = transforms.ColorJitter(brightness=0.9)
    anomaly_source_image = random_brightness(anomaly_source_image)
    anomaly_source_image = torch.from_numpy(np.array(anomaly_source_image))
    image = torch.from_numpy(np.array(image))
    dims = np.array(np.shape(image)[:-1]) 
    core = core_percent * dims 
    offset = (1 - core_percent) * dims / 2  
    min_width = np.round(0.05 * dims[1])
    max_width = np.round(0.2 * dims[1])  
    center_dim1 = np.random.randint(offset[0], offset[0] + core[0])
    center_dim2 = np.random.randint(offset[1], offset[1] + core[1])
    patch_center = np.array([center_dim1, center_dim2])
    patch_width = np.random.randint(min_width, max_width)

    coor_min = patch_center - patch_width
    coor_max = patch_center + patch_width

    coor_min = np.clip(coor_min, 0, dims)
    coor_max = np.clip(coor_max, 0, dims)

    alpha = torch.rand(1)  
    alpha = alpha * 0.5 + 0.5
    mask = torch.zeros_like(image)[:,:,0].squeeze()
    mask = mask.float()

    mask[coor_min[0]:coor_max[0], coor_min[1]:coor_max[1]] = alpha
    mask_inv = 1 - mask
    



    anomaly_source = anomaly_source_image
    image_synthesis = mask_inv * image[:,:,0] + mask * anomaly_source[:,:,0]
    image_synthesis = image_synthesis.numpy()
    image_synthesis = np.stack((image_synthesis,)*3, axis=-1)
    image_synthesis = image_synthesis.astype(np.uint8)
    return image_synthesis


def fpi_mask_generate_anomaly(image, anomaly_source_image, image_mask):

    image_mask = image_mask.squeeze()

    left = np.where(image_mask==1)[1].min()
    right = np.where(image_mask==1)[1].max()
    top = np.where(image_mask==1)[0].min()
    bottom = np.where(image_mask==1)[0].max()

    center_dim1 = np.random.randint(top, bottom)  
    center_dim2 = np.random.randint(left, right) 

    anomaly_source_image = torch.from_numpy(np.array(anomaly_source_image))
    image = torch.from_numpy(np.array(image))
    dims = np.array(np.shape(image)[:-1]) 


    min_width = np.round(0.05 * dims[1])
    max_width = np.round(0.2 * dims[1])  
    patch_center = np.array([center_dim1, center_dim2])
    patch_width = np.random.randint(min_width, max_width)

    coor_min = patch_center - patch_width
    coor_max = patch_center + patch_width

    coor_min = np.clip(coor_min, 0, dims)
    coor_max = np.clip(coor_max, 0, dims)

    alpha = torch.rand(1)  
    alpha = alpha * 0.5 + 0.5
    mask = torch.zeros_like(image)[:,:,0].squeeze()
    mask = mask.float()

    mask[coor_min[0]:coor_max[0], coor_min[1]:coor_max[1]] = alpha
    mask_inv = 1 - mask
    


    anomaly_source = anomaly_source_image
    image_synthesis = mask_inv * image[:,:,0] + mask * anomaly_source[:,:,0]
    image_synthesis = image_synthesis.numpy()
    image_synthesis = np.stack((image_synthesis,)*3, axis=-1)
    image_synthesis = image_synthesis.astype(np.uint8)
    return image_synthesis

def generate_random_excluding(low, high, exclude_number):
    while True:
        random_number = random.randint(low, high)
        if random_number != exclude_number:
            return random_number


