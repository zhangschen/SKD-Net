import cv2
import torch
import torch.nn.functional as F
import numpy as np
from numpy import sin, cos, tan
from torch import linalg



def check_margins(img, axis=-1):
    if axis == -1:
        img[img > 255.0] = 255.0
        img[img < 0.0] = 0.0
    else:
        img[:, :, axis][img[:, :, axis] > 255.0] = 255.0
        img[:, :, axis][img[:, :, axis] < 0.0] = 0.0
    return img


def to_black_and_white(img):
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return im.reshape(im.shape[0], im.shape[1], 1)


def swap_channels(image, swaps):
    image = image[:, :, swaps]
    return image

def generate_composed_homography(path_image_idx,max_angle=45, min_scaling=0.5, max_scaling=2.0, max_shearing=0.8):
    # random sample
    # scale = np.random.uniform(min_scaling, max_scaling)
    if path_image_idx % 10==0:
        angle=0
    else:
        angle = np.random.uniform(-max_angle, max_angle)  # 随机生成角度

    scale = np.random.uniform(1.0, 1.0)
    # shear = np.random.uniform(-max_shearing, max_shearing)
    shear = np.random.uniform(0, 0)
    # scale transform
    scale_mat = np.eye(3)
    scale_mat[0, 0] = scale
    scale_mat[1, 1] = scale
    # rotation transform
    angle = np.deg2rad(angle)
    rotation_mat = np.eye(3)
    rotation_mat[0, 0] = np.cos(angle)
    rotation_mat[0, 1] = -np.sin(angle)
    rotation_mat[1, 0] = np.sin(angle)
    rotation_mat[1, 1] = np.cos(angle)
    # shear transform
    shear_mat = np.eye(3)
    shear_mat[0, 1] = shear

    h = np.matmul(scale_mat, rotation_mat)
    return h, scale, np.rad2deg(angle), shear







