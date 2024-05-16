import noise
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import cv2
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import skimage.segmentation as skimage_seg
from scipy.ndimage import distance_transform_edt as distance
import torch
from torchvision import transforms

def generate_perlin_noise(size, scale, octaves, persistence, lacunarity):
    world = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            world[i][j] = noise.pnoise2(i/scale,
                                        j/scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=1024,
                                        repeaty=1024,
                                        base=42)
    return world


def generate_random_points_in_perlin_noise(size, num_points, scale, octaves, persistence, lacunarity):
    world = generate_perlin_noise(size, scale, octaves, persistence, lacunarity)
    points = []
    for _ in range(num_points):
        x = np.random.randint(0, size[0])
        y = np.random.randint(0, size[1])
        points.append((x, y, world[x][y]))
    return points


def generate_random_bezier(p0, p1, control_point_factor=10, num_points=100):
    t = np.linspace(0, 1, num_points)
    
    control_point = (p0 + p1) / 2 + np.random.normal(0, control_point_factor, size=2)
    
    bezier_curve = np.outer((1 - t) ** 2, p0) + np.outer(2 * (1 - t) * t, control_point) + np.outer(t ** 2, p1)
    
    return bezier_curve


def generate_random_bezier_cubic(p0, p1, control_point_factor=8, num_points=100):
    t = np.linspace(0, 1, num_points)
    
    control_point1 = (p0 + p1) / 2 + np.random.normal(0, control_point_factor, size=2)
    control_point2 = (p0 + p1) / 2 + np.random.normal(0, control_point_factor, size=2)
    
    bezier_curve = (
        np.outer((1 - t) ** 3, p0) +
        np.outer(3 * (1 - t) ** 2 * t, control_point1) +
        np.outer(3 * (1 - t) * t ** 2, control_point2) +
        np.outer(t ** 3, p1)
    )
    
    return bezier_curve


def compute_sdf(img_gt,class_num, weight=-0.99):

    w_num,h_num = img_gt.shape
    img_gt_split = np.zeros((w_num,h_num,class_num))

    for num in range(1,class_num+1):
        img_gt_split[...,num-1][img_gt==num] =1


    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros((class_num,w_num,h_num))
    if np.max(img_gt) == 0:
        return normalized_sdf
    for c in range(class_num):
        img_gt = img_gt_split[...,c]
        posmask = img_gt.astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = 1 + posdis / np.max(posdis) * weight
            sdf[boundary==1] = 1


    return sdf


def mask_generate(size, num_points):
    size = size
    num_points = num_points
    size_max = max(size)
    shape = (size_max, size_max)

    scale = 20.0  
    octaves = 6 
    persistence = 0.5  
    lacunarity = 2.0  
    
    points = generate_random_points_in_perlin_noise(size, num_points, scale, octaves, persistence, lacunarity)

    point_coordinates = np.array([(point[0], point[1]) for point in points])

    hull = ConvexHull(point_coordinates)


    path = Path(hull.points[hull.vertices])


    mask1 = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            if path.contains_point((j, i)):
                mask1[i, j] = 1



    coordinates = []
    lines = []

    for simplex in hull.simplices:

        line = generate_random_bezier_cubic(point_coordinates[simplex[0]], point_coordinates[simplex[1]])
        lines.append(line)

    lines = np.array(lines)


    mask2 = np.zeros(shape)

    for line in lines:
        path = Path(line)

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1])) 
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T 

        grid = path.contains_points(points)
        mask = grid.reshape(shape[0], shape[1]) 


        mask2[mask] = 1


    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    union_minus_intersection = np.logical_xor(union, intersection)
    union_minus_intersection = cv2.dilate(union_minus_intersection.astype(np.uint8), np.ones((5,5),np.uint8), iterations=1)

    return union_minus_intersection


def gamma_correction(image, gamma_map):

    gamma_map = torch.tensor(gamma_map)


    gamma_corrected = torch.pow(image/255, gamma_map) * 255

    gamma_corrected = torch.clamp(gamma_corrected, 0, 255)

    gamma_corrected = gamma_corrected.numpy()

    return gamma_corrected


def generate_anomaly_mask(image, weight, image_mask):

    image = torch.from_numpy(np.array(image))
    dims = np.array(np.shape(image)[:-1])  
    image_mask = image_mask.squeeze()
    top_offset = 50
    left_offset = 30
    left = np.where(image_mask==1)[1].min()
    right = np.where(image_mask==1)[1].max()
    top = np.where(image_mask==1)[0].min()
    bottom = np.where(image_mask==1)[0].max()

    center_dim1 = np.random.randint(top, bottom) 
    center_dim2 = np.random.randint(left, right)  

    min_width = np.round(0.2 * dims[1])
    max_width = np.round(0.8 * dims[1]) 

    patch_center = np.array([center_dim1, center_dim2])
    patch_width = np.random.randint(min_width, max_width)
    patch_height = np.random.randint(min_width, max_width)

    mymask = mask_generate((patch_width, patch_height), 10)
    mymask = torch.from_numpy(mymask)


    mask = torch.zeros_like(image)[:,:,0].squeeze()
    mask = mask.float()
    mymask_size = mymask.shape

    mysize_h, mysize_w = mymask_size
    the_x1 = patch_center[0]-mysize_w//2
    the_x2 = patch_center[0]+mysize_w-mysize_w//2

    the_y1 = patch_center[1]-mysize_h//2
    the_y2 = patch_center[1]+mysize_h-mysize_h//2

    if the_x1 < 0:
        the_x2 = the_x2 - the_x1
        the_x1 = 0

    if the_y1 < 0:
        the_y2 = the_y2 - the_y1
        the_y1 = 0

    if the_x2 > dims[0]:
        the_x1 = the_x1 + (dims[0]-1-the_x2)
        the_x2 = dims[0]-1

    if the_y2 > dims[1]:
        the_y1 = the_y1 + (dims[1]-1-the_y2)
        the_y2 = dims[1]-1


    mask[the_x1:the_x2, the_y1:the_y2] = mymask


    sdf = compute_sdf(img_gt=mask.numpy(), class_num=2, weight=weight)


    anomaly_source_image = gamma_correction(image[:,:,0], sdf)
    anomaly_source_image = np.stack((anomaly_source_image,)*3, axis=-1)

    

    anomaly_source_image = anomaly_source_image.astype(np.uint8)

    return anomaly_source_image


