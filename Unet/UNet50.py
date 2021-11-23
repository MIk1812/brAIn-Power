# Inspired by https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py

import torch
import torch.nn as nn
import torchvision
from Unet.UpBlock import UpBlock
from Unet.Bridge import Bridge
import torch.nn.functional as F

class UNet50(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes,rgb=True):
        super().__init__()
        self.rgb = rgb
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        blocks_up = []
        blocks_down = []
        self.conv0 = torch.nn.Conv2d(1,3,1,1)
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                blocks_up.append(bottleneck)
        self.down_blocks = nn.ModuleList(blocks_up)

        self.bridge = Bridge(2048, 4096)

        blocks_down.append(UpBlock(4096, 2048))
        blocks_down.append(UpBlock(2048, 1024))
        blocks_down.append(UpBlock(1024, 512))
        blocks_down.append(UpBlock(512, 256))
        blocks_down.append(UpBlock(channels_in=128 + 64, channels_out=128, up_channels_in=256, up_channels_out=128))
        blocks_down.append(UpBlock(channels_in=64 + 3, channels_out=64, up_channels_in=128, up_channels_out=64))
        self.up_blocks = nn.ModuleList(blocks_down)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

        self.softmax = nn.Softmax2d()

    def forward(self, x):
        if not self.rgb:
            x = self.conv0(x)
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            key = f"layer_{i}"
            pre_pools[key] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNet50.DEPTH - i}"
            x = block(x, pre_pools[key])

        x = self.out(x)
        x = self.softmax(x)
        del pre_pools
        return x


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


ALPHA = 0.5  # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5  # weighted contribution of modified CE loss compared to Dice loss


class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, ALPHA=ALPHA, CE_RATIO=CE_RATIO, eps=1e-9):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        inputs = torch.clamp(inputs, eps, 1.0 - eps)
        out = - (ALPHA * ((targets * torch.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)

        return combo


if __name__ == '__main__':
    model = UNet50(n_classes=9) # .cuda()
    inp = torch.rand((2, 3, 512, 512)) # .cuda()
    out = model(inp)
    print("finished...")





