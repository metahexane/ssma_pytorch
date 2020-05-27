import torch.nn as nn
import torch


class BottleneckSSMA(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, r1, r2, d3, stride=1, downsample=None, copy_from=None):
        super(BottleneckSSMA, self).__init__()
        half_d3 = int(d3 / 2)

        self.conv2a = nn.Conv2d(planes, half_d3, kernel_size=3, stride=1, dilation=r1,
                                padding=r1, bias=False)
        self.bn2a = nn.BatchNorm2d(half_d3)
        self.conv2b = nn.Conv2d(planes, half_d3, kernel_size=3, stride=1, dilation=r2,
                                padding=r2, bias=False)
        self.bn2b = nn.BatchNorm2d(half_d3)
        self.conv3 = nn.Conv2d(d3, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)

        # nn.init.xavier_uniform(self.conv2a, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform(self.conv2b, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform(self.conv3, gain=nn.init.calculate_gain('relu'))

        if copy_from is None:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
        else:
            self.conv1 = copy_from.conv1
            self.bn1 = copy_from.bn1

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out_a = self.conv2a(out)
        out_a = self.bn2a(out_a)
        out_a = self.relu(out_a)

        out_b = self.conv2b(out)
        out_b = self.bn2b(out_b)
        out_b = self.relu(out_b)

        out = torch.cat((out_a, out_b), dim=1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
