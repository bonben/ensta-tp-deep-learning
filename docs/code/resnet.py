"""ResNet pour CIFAR-10 (images 32x32).

Adapté de https://github.com/kuangliu/pytorch-cifar (licence MIT).
Différence avec le ResNet d'ImageNet : la première couche est une convolution
3x3 sans max-pooling, car les images ne font que 32x32 pixels.

ResNet18 : 11,2 millions de paramètres, environ 95 % sur CIFAR-10.
C'est la baseline que vous compresserez dans la suite du module.
"""

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Deux convolutions 3x3 et un raccourci (la connexion résiduelle)."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Si la taille change (stride ou nombre de canaux), le raccourci doit
        # lui aussi changer de taille : convolution 1x1.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)  # la connexion résiduelle
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, width=64):
        super().__init__()
        self.in_planes = width

        self.conv1 = nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self._make_layer(block, width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * width, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * width, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * width, num_blocks[3], stride=2)
        self.linear = nn.Linear(8 * width * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 32x32
        out = self.layer1(out)                 # 32x32
        out = self.layer2(out)                 # 16x16
        out = self.layer3(out)                 # 8x8
        out = self.layer4(out)                 # 4x4
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


def ResNet18(num_classes=10, width=64):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, width)


def ResNet34(num_classes=10, width=64):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, width)
