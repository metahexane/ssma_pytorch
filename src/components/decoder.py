import torch.nn as nn
import torch


class Decoder(nn.Module):
    """PyTorch Module for decoder"""

    def __init__(self, C, fusion=False):
        """Constructor

        :param C: Number of categories
        :param fusion: boolean for fused skip connections (False for stage 1, True for stages 2 and 3)
        """
        super(Decoder, self).__init__()

        # variables
        self.num_categories = C
        self.fusion = fusion

        # layers stage 1
        self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        nn.init.kaiming_uniform_(self.deconv1.weight, nonlinearity="relu")
        self.deconv1_bn = nn.BatchNorm2d(256)

        # layers stage 2
        self.stage2 = nn.Sequential(
            nn.Conv2d(280, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256)
        )
        for i, layer in enumerate(self.stage2):
            if str(type(layer)) == "<class 'torch.nn.modules.conv.Conv2d'>" or \
                                   str(type(layer)) == "<class 'torch.nn.modules.conv.ConvTranspose2d'>":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")

        # layers stage 3
        self.stage3 = nn.Sequential(
            nn.Conv2d(280, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.num_categories, 1),
            nn.BatchNorm2d(self.num_categories),
            nn.ConvTranspose2d(self.num_categories, self.num_categories, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(self.num_categories)
        )
        for i, layer in enumerate(self.stage2):
            if str(type(layer)) == "<class 'torch.nn.modules.conv.Conv2d'>" or \
                                   str(type(layer)) == "<class 'torch.nn.modules.conv.ConvTranspose2d'>":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")

        # decoder auxiliary layers
        self.aux_conv1 = nn.Conv2d(256, self.num_categories, 1)
        nn.init.kaiming_uniform_(self.aux_conv1.weight, nonlinearity="relu")
        self.aux_conv1_bn = nn.BatchNorm2d(self.num_categories)
        self.aux_conv2 = nn.Conv2d(256, self.num_categories, 1)
        nn.init.kaiming_uniform_(self.aux_conv2.weight, nonlinearity="relu")
        self.aux_conv2_bn = nn.BatchNorm2d(self.num_categories)

        # decoder fuse skip layers
        self.fuse_conv1 = nn.Conv2d(256, 24, 1)
        nn.init.kaiming_uniform_(self.fuse_conv1.weight, nonlinearity="relu")
        self.fuse_conv1_bn = nn.BatchNorm2d(24)
        self.fuse_conv2 = nn.Conv2d(256, 24, 1)
        nn.init.kaiming_uniform_(self.fuse_conv2.weight, nonlinearity="relu")
        self.fuse_conv2_bn = nn.BatchNorm2d(24)

    def forward(self, x, skip1, skip2):
        """Forward pass

        :param x: input feature maps from eASPP
        :param skip1: skip connection 1
        :param skip2: skip connection 2
        :return: final output and auxiliary output 1 and 2
        """
        # stage 1
        x = torch.relu(self.deconv1_bn(self.deconv1(x)))
        y1 = self.aux(x, self.aux_conv1, self.aux_conv1_bn, 8)
        if self.fusion:
            # integrate fusion skip
            int_fuse_skip = self.integrate_fuse_skip(x, skip1, self.fuse_conv1, self.fuse_conv1_bn)
            x = torch.cat((x, int_fuse_skip), 1)
        else:
            x = torch.cat((x, skip1), 1)

        # stage 2
        x = self.stage2(x)
        y2 = self.aux(x, self.aux_conv2, self.aux_conv2_bn, 4)
        if self.fusion:
            # integrate fusion skip
            int_fuse_skip = self.integrate_fuse_skip(x, skip2, self.fuse_conv2, self.fuse_conv2_bn)
            x = torch.cat((x, int_fuse_skip), 1)
        else:
            x = torch.cat((x, skip2), 1)

        # stage 3
        y3 = self.stage3(x)

        return y1, y2, y3

    def aux(self, x, conv, bn, scale):
        """Compute auxiliary output"""
        x = bn(conv(x))

        return nn.UpsamplingBilinear2d(scale_factor=scale)(x)

    def integrate_fuse_skip(self, x, fuse_skip, conv, bn):
        """Integrate fuse skip connection with decoder"""
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = torch.relu(bn(conv(x)))

        return torch.mul(x, fuse_skip)
