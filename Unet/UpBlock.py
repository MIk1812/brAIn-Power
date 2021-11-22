# Inspired by https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py

import torch
import torch.nn as nn
from Unet.ConvBlock import ConvBlock

# Upsample -> Concatenate -> ConvBlock -> ConvBlock
class UpBlock(nn.Module):

    def __init__(self, channels_in, channels_out, up_channels_in=None, up_channels_out=None):
        super().__init__()

        if up_channels_in is None:
            up_channels_in = channels_in
        if up_channels_out is None:
            up_channels_out = channels_out

        # Double the resolution
        self.upsample = nn.ConvTranspose2d(up_channels_in, up_channels_out, kernel_size=2, stride=2)

        self.conv_block_1 = ConvBlock(channels_in, channels_out)
        self.conv_block_2 = ConvBlock(channels_out, channels_out)

    def forward(self, x_up, x_down):
        x = self.upsample(x_up)
        x = torch.cat([x, x_down], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x