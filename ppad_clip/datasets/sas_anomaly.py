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
    
    # 生成两个随机的控制点
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
    """

    class_num为前景类别，不包含背景
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size,c, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """
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
            # sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf = 1 + posdis / np.max(posdis) * weight
            sdf[boundary==1] = 1
            # normalized_sdf[c,...] = sdf


    return sdf


def mask_generate(size, num_points):
    size = size
    num_points = num_points
    size_max = max(size)
    shape = (size_max, size_max)

    scale = 20.0  # 控制噪声的规模
    octaves = 6  # 八度数
    persistence = 0.5  # 持续度
    lacunarity = 2.0  # 衰减率
    
    points = generate_random_points_in_perlin_noise(size, num_points, scale, octaves, persistence, lacunarity)

    # 提取点的坐标
    point_coordinates = np.array([(point[0], point[1]) for point in points])

    # 计算凸包
    hull = ConvexHull(point_coordinates)


    path = Path(hull.points[hull.vertices])

    # 创建掩码

    mask1 = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            if path.contains_point((j, i)):
                mask1[i, j] = 1



    # 绘制原始点和凸包
    coordinates = []
    lines = []

    for simplex in hull.simplices:
        # plt.plot(point_coordinates[simplex, 0], point_coordinates[simplex, 1], 'b-')

        line = generate_random_bezier_cubic(point_coordinates[simplex[0]], point_coordinates[simplex[1]])
        lines.append(line)

    lines = np.array(lines)


    mask2 = np.zeros(shape)

    # 对每个曲线段创建一个路径对象，并生成一个mask
    for line in lines:
        # 创建一个路径对象
        path = Path(line)

        # 创建一个网格
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1])) # make a canvas with coordinates
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T 

        # 获取路径内的点
        grid = path.contains_points(points)
        mask = grid.reshape(shape[0], shape[1]) # now you have a mask with points inside a polygon

        # 使用mask填充图像

        mask2[mask] = 1

    # 显示图像

    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    union_minus_intersection = np.logical_xor(union, intersection)
    # 对union_minus_intersection进行膨胀
    union_minus_intersection = cv2.dilate(union_minus_intersection.astype(np.uint8), np.ones((5,5),np.uint8), iterations=1)

    return union_minus_intersection


def gamma_correction(image, gamma_map):
    """
    对图像进行Gamma变换，每个像素点使用不同的gamma值

    Parameters:
    - image: 输入图像 (PyTorch Tensor)
    - gamma_map: 与输入图像大小相同的二维数组，每个元素是对应位置的gamma值 (NumPy array)

    Returns:
    - gamma_corrected: Gamma变换后的图像 (PyTorch Tensor)
    """
    # Convert the gamma_map NumPy array to a PyTorch Tensor
    gamma_map = torch.tensor(gamma_map)

    # Apply Gamma correction using PyTorch power function
    # gamma_map = (gamma_map - gamma_map.min()) / (gamma_map.max() - gamma_map.min())
    # gamma_map = 1.5
    gamma_corrected = torch.pow(image/255, gamma_map) * 255

    # Clip the transformed pixel values to ensure they are in the valid range
    gamma_corrected = torch.clamp(gamma_corrected, 0, 255)

    # Convert the result to a NumPy array for further processing or visualization
    gamma_corrected = gamma_corrected.numpy()

    return gamma_corrected


def generate_anomaly(image, weight):

    # anomaly_source_image = torch.from_numpy(np.array(anomaly_source_image))
    image = torch.from_numpy(np.array(image))
    dims = np.array(np.shape(image)[:-1])  # H x W

    top_offset = 50
    left_offset = 30
    center_dim1 = np.random.randint(top_offset, dims[0]-top_offset)  # H
    center_dim2 = np.random.randint(left_offset, dims[1]-left_offset)  # W

    min_width = np.round(0.2 * dims[1])
    max_width = np.round(0.8 * dims[1])  # w

    patch_center = np.array([center_dim1, center_dim2])
    patch_width = np.random.randint(min_width, max_width)
    patch_height = np.random.randint(min_width, max_width)

    mymask = mask_generate((patch_width, patch_height), 10)
    mymask = torch.from_numpy(mymask)

    #创建一个width为patch_width，height为patch_height的mask
    # mymask = torch.ones(])

    mask = torch.zeros_like(image)[:,:,0].squeeze()
    mask = mask.float()
    mymask_size = mymask.shape

    mysize_h, mysize_w = mymask_size
    the_x1 = patch_center[0]-mysize_w//2
    the_x2 = patch_center[0]+mysize_w-mysize_w//2

    the_y1 = patch_center[1]-mysize_h//2
    the_y2 = patch_center[1]+mysize_h-mysize_h//2

    # 超出边界的分布移除掉
    if the_x1 < 0:
        # mymask = mymask[:, -the_x1:]
        the_x2 = the_x2 - the_x1
        the_x1 = 0

    if the_y1 < 0:
        # mymask = mymask[-the_y1:, :]
        the_y2 = the_y2 - the_y1
        the_y1 = 0

    if the_x2 > dims[0]:
        # mymask = mymask[:, :dims[0]-the_x2]
        the_x1 = the_x1 + (dims[0]-1-the_x2)
        the_x2 = dims[0]-1

    if the_y2 > dims[1]:
        # mymask = mymask[:dims[1]-the_y2, :]
        the_y1 = the_y1 + (dims[1]-1-the_y2)
        the_y2 = dims[1]-1


    mask[the_x1:the_x2, the_y1:the_y2] = mymask
    # new_mask = torch.zeros_like(mask)
    # new_mask[the_x1:the_x2, the_y1:the_y2] = 255
    # mask[the_x1:the_y1, the_x2:the_y2] = mymask

    sdf = compute_sdf(img_gt=mask.numpy(), class_num=2, weight=weight)


    anomaly_source_image = gamma_correction(image[:,:,0], sdf)
    anomaly_source_image = np.stack((anomaly_source_image,)*3, axis=-1)

    

    # image_synthesis = (1 - mask) * image[:,:,0] + mask * anomaly_source_image[:,:,0]
    # # image_synthesis = mask_inv * image[:,:,0] + mask * 255

    # image_synthesis = image_synthesis.numpy()
    # image_synthesis = np.stack((image_synthesis,)*3, axis=-1)
    # image_synthesis = image_synthesis.astype(np.uint8)
    anomaly_source_image = anomaly_source_image.astype(np.uint8)
    # return image_synthesis, anomaly_source_image, mask

    # return anomaly_source_image, mask, new_mask
    return anomaly_source_image



if __name__ == "__main__":

    img = Image.open('/data/sunzc/zhanglab/train/normal_256/NORMAL-996167-0001.jpeg')
    img = np.array(img)
    img = np.stack((img,)*3, axis=-1)
    img = Image.fromarray(img)
    img = img.resize((224, 224))
    anomaly_source_image = Image.open('/data/sunzc/zhanglab/train/normal_256/NORMAL-9990348-0001.jpeg')
    anomaly_source_image = np.array(anomaly_source_image)
    anomaly_source_image = np.stack((anomaly_source_image,)*3, axis=-1)
    anomaly_source_image = Image.fromarray(anomaly_source_image)
    anomaly_source_image = anomaly_source_image.resize((224, 224))

    # anomaly_source = generate_anomaly(img, -0.999)
    anomaly_source, mask, new_mask = generate_anomaly(img, -0.99)

    # anomaly_source, mask, new_mask = generate_anomaly(img, 2)


    fig, axes = plt.subplots(1, 4, figsize=(8, 12))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('original image')

    axes[1].imshow(anomaly_source, cmap='gray')
    axes[1].set_title('anomaly source image')

    axes[2].imshow(mask, cmap='jet')
    axes[2].set_title('new image')

    axes[3].imshow(new_mask, cmap='jet')
    axes[3].set_title('new image2')

    plt.show()