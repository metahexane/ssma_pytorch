import torch
from torch import nn as nn
import torch.nn.functional as F

class SSMA(nn.Module):
    def __init__(self):
        # encoder layers
        self.enc_conv_1 = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.enc_conv_1_bn = nn.BatchNorm2d(64)
        self.max_pool_2x2 = nn.MaxPool2d(2)

        self.u2_sizes_short = [(64, 256), (64, 256), (128, 512), (128, 512), (256, 1024)]
        self.u2_sizes_block = [(128, 512), (64, 256), (128, 512), (128, 512), (256, 1024)]
        self.enc_u2_short = []
        self.enc_u2_block = []

        self._init_u1()
        self._init_u2(self.enc_u2_short, self.u2_sizes_short, s=1)

    def _init_u1(self):
        self.enc_conv_u1_1 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.enc_conv_u1_1_bn = nn.BatchNorm2d(64)
        self.enc_conv_u1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.enc_conv_u1_2_bn = nn.BatchNorm2d(64)
        self.enc_conv_u1_3 = nn.Conv2d(64, 256, kernel_size=1, stride=1)
        self.enc_conv_u1_4 = nn.Conv2d(64, 256, kernel_size=1, stride=1)

    def _init_u2(self, arry, sizes, s=1):
        for i, x in enumerate(sizes):
            u2_comps = [
                nn.BatchNorm2d(x[1]),
                nn.Conv2d(x[1], x[0], kernel_size=1, stride=1),
                nn.BatchNorm2d(x[0]),
                nn.Conv2d(x[0], x[0], kernel_size=3, stride=s),
                nn.BatchNorm2d(x[0]),
                nn.Conv2d(x[0], x[1], kernel_size=1, stride=1),

            ]
            arry.append(u2_comps)


    def forward(self, x):
        pass

    def encode(self, x):
        x = F.relu(self.enc_conv_1_bn(self.enc_conv_1(x)))
        x = self.max_pool_2x2(x)

        pass

    def decode(self, x):
        pass



    def enc_unit_1(self, x):
        x = F.relu(self.enc_conv_u1_1_bn(self.enc_conv_u1_1(x)))
        o1 = self.enc_conv_u1_4(x)
        x = F.relu(self.enc_conv_u1_2_bn(self.enc_conv_u1_2(x)))
        x = self.enc_conv_u1_3(x)
        return x + o1
