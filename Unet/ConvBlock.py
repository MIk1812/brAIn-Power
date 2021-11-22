# Inspired by https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py

import torch.nn as nn

# Conv -> BN -> ReLU
class ConvBlock(nn.Module):

    def __init__(self, channels_in, channels_out, padding=1, kernel_size=3, stride=1, act_fct=nn.ReLU()):
        super().__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(channels_out)
        self.act_fct = act_fct

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_fct(x)

        return x
