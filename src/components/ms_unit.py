import torch
import torch.nn as nn


class BottleneckSSMA(nn.Module):
    """PyTorch Module for multi-scale units (modified residual units) for Resnet50 stages"""

    def __init__(self, inplanes, planes, r1, r2, d3, stride=1, downsample=None, copy_from=None, drop_out=False):
        """Constructur

        :param inplanes: input dimension
        :param planes: output dimension
        :param r1: dilation rate and padding 1
        :param r2: dilation rate and padding 2
        :param d3: split factor
        :param stride: stride
        :param downsample: down sample rate
        :param copy_from: copy of residual unit from second/third stage resnet50
        :param drop_out: boolean for inclusion of dropout layer
        """
        super(BottleneckSSMA, self).__init__()
        self.dropout = drop_out

        half_d3 = int(d3 / 2)

        self.conv2a = nn.Conv2d(planes, half_d3, kernel_size=3, stride=1, dilation=r1,
                                padding=r1, bias=False)
        self.bn2a = nn.BatchNorm2d(half_d3)
        self.conv2b = nn.Conv2d(planes, half_d3, kernel_size=3, stride=1, dilation=r2,
                                padding=r2, bias=False)
        self.bn2b = nn.BatchNorm2d(half_d3)
        self.conv3 = nn.Conv2d(d3, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)

        nn.init.kaiming_uniform_(self.conv2a.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2b.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity="relu")

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
        """Forward Pass

        :param x: input feature maps
        :return: output feature maps
        """
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

        if self.dropout:
            m = nn.Dropout(p=0.5)
            out = m(out)

        return out
