import torch
import numpy as np
import torch.nn.functional as F
import numpy
from e2cnn import gspaces
from torch import nn
from torchvision.models import resnet
from .kernels import gaussian_multiple_channels
from TZDJC import BASE_Transformer


class SKD(torch.nn.Module):
    def __init__(self, args, device):
        super(SKD, self).__init__()
        self.net1=SKD_Net()
    def forward(self, input_image):
        features1 = self.net1(input_image)
        return features1

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        # self.gate = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()
        self.conv1 = resnet.conv3x3(in_channel, out_channel)
        self.bn1 =nn.BatchNorm2d(out_channel)
        self.conv2 = resnet.conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))  # B x in_channels x H x W
        x = self.gate(self.bn2(self.conv2(x)))  # B x out_channels x H x W
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, strides, padding):
        super(ResBlock, self).__init__()
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=strides, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid(),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size
                               , stride=strides, padding=padding, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.block(x)
        x = self.bn3(self.conv3(x))

        return F.sigmoid(out + x)


class SKD_Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.gate = nn.Sigmoid()

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.block1 = ConvBlock(1, 32)                # (1,32,256,256)
        self.block2 = ResBlock(32, 64, 3 , 1, 1)      # (1,64,128,128)

        self.block3 = ResBlock(64,128,  3 , 1, 1)     # (1,128,64,64)

        self.Conv2= nn.Conv2d(128, 32, kernel_size=1, padding=0)

        self.Transformer=BASE_Transformer()
        #转换维度
        self.conv1 = resnet.conv1x1(32, 32)
        self.conv2 = resnet.conv1x1(64, 32)
        self.conv3 = resnet.conv1x1(128, 32)
        self.conv4 = resnet.conv1x1(32, 32)

    def forward(self, image):
        # ================================== feature encoder
        x1 = self.block1(image)  # B x 32 x H x W

        x2 = self.pool2(x1)
        x2 = self.block2(x2)  # B x 64 x H/2 x W/2

        x3 = self.pool2(x2)
        x3 = self.block3(x3)  # B x 128 x H/8 x W/8
        x33=self.Conv2(x3)

        x4=self.Transformer(x33)

        #卷积降成相同维度
        x111=self.conv1(x1)
        x222=self.conv2(x2)
        x333=self.conv3(x3)
        x444=self.conv4(x4)

        #对特征图进行进行上采样
        [_,_,H,W]=image.shape
        x222_up = F.interpolate(x222, size=(H, W), align_corners=True, mode='bilinear')
        x333_up = F.interpolate(x333, size=(H, W), align_corners=True, mode='bilinear')
        x444_up = F.interpolate(x444, size=(H, W), align_corners=True, mode='bilinear')

        x_up=0.6*x111+0.3*x222_up+0.1*x444_up
        #提取峰值图
        peak_x = peakiness_score(x_up)

        return peak_x



