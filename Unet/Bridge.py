# Inspired by https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py

import torch.nn as nn
from Unet.ConvBlock import ConvBlock


# Middle layer of U-net
class Bridge(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = ConvBlock(channels_in, channels_out)
        self.conv2 = ConvBlock(channels_out, channels_out)

    def forward(self, x):
        x = self.max_pool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
