import torch.nn as nn
import torch


class eASPP(nn.Module):
    """PyTorch Module for eASPP"""

    def __init__(self):
        """Constructor

        Initializes the 5 branches of the eASPP network.
        """

        super(eASPP, self).__init__()

        # branch 1
        self.branch1_conv = nn.Conv2d(2048, 256, kernel_size=1)
        self.branch1_bn = nn.BatchNorm2d(256)

        self.branch234 = nn.ModuleList([])
        self.branch_rates = [3, 6, 12]
        for rate in self.branch_rates:
            # branch 2
            branch = nn.Sequential(
                nn.Conv2d(2048, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, dilation=rate, padding=rate),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, dilation=rate, padding=rate),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
            self.branch234.append(branch)
        for i, sequence in enumerate(self.branch234):
            for ii, layer in enumerate(sequence):
                if str(type(layer)) == "<class 'torch.nn.modules.conv.Conv2d'>":
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")

        # branch 5
        self.branch5_conv = nn.Conv2d(2048, 256, 1)
        nn.init.kaiming_uniform_(self.branch5_conv.weight, nonlinearity="relu")
        self.branch5_bn = nn.BatchNorm2d(256)

        # final layer
        self.eASPP_fin_conv = nn.Conv2d(1280, 256, kernel_size=1)
        nn.init.kaiming_uniform_(self.eASPP_fin_conv.weight, nonlinearity="relu")
        self.eASPP_fin_bn = nn.BatchNorm2d(256)

    def forward(self, x):
        """Forward pass

        :param x: input from encoder (in stage 1) or from fused encoders (in stage 2 and 3)
        :return: feature maps to be forwarded to decoder
        """
        # branch 1: 1x1 convolution
        out = torch.relu(self.branch1_bn(self.branch1_conv(x)))

        # branch 2-4: atrous pooling
        y = self.branch234[0](x)
        out = torch.cat((out, y), 1)
        y = self.branch234[1](x)
        out = torch.cat((out, y), 1)
        y = self.branch234[2](x)
        out = torch.cat((out, y), 1)

        # branch 5: image pooling
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = torch.relu(self.branch5_bn(self.branch5_conv(x)))
        x = nn.Upsample((24, 48), mode="bilinear")(x)
        out = torch.cat((out, x), 1)

        return torch.relu(self.eASPP_fin_bn(self.eASPP_fin_conv(out)))
