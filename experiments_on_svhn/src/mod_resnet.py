import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride, alpha):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.a = -math.log(alpha-1, math.e)
        self.alpha = alpha
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = torch.sigmoid(self.bn2(self.conv2(out)) + self.a) * self.alpha
        if self.stride != 1:
            out = torch.sigmoid(self.shortcut(x) + self.a) * self.alpha * out
        else:
            out = self.shortcut(x) * out
        return out


class QuoNet(nn.Module):
    def __init__(self, block, num_blocks, alpha, num_classes=10):
        super(QuoNet, self).__init__()
        self.in_planes = 16
        self.alpha = alpha
        self.a = -math.log(alpha - 1, math.e)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.alpha))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.sigmoid(self.bn1(self.conv1(x)) + self.a) * self.alpha
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def q44():
    return QuoNet(BasicBlock, [7, 7, 7], alpha=1.8)


def q56():
    return QuoNet(BasicBlock, [9, 9, 9], alpha=1.7)


def q110():
    return QuoNet(BasicBlock, [18, 18, 18], alpha=1.5)
