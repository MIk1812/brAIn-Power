import torchvision
import torch.nn as nn
from torchvision.models import ResNet


# Explore the architecture of different ResNets
def printResNet(net: ResNet):
    children = list(net.children())
    print(len(children))

    print("\n-- INPUT BLOCK --")
    print(nn.Sequential(*list(net.children()))[:3])

    print("\n-- INPUT POOL --")
    print(list(net.children())[3])

    print("\n-- CHILDREN --")
    for i in range(len(children)):
        print("\nCHILD:")
        print(children[i])

printResNet(torchvision.models.resnet.resnet50(pretrained=True))