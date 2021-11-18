from torchvision import *
import torch.nn as nn

net = models.resnet50(pretrained=True)
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

