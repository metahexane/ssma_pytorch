import torch
import torch.nn as nn


class SSMAUnit(nn.Module):

    def __init__(self, features, bottleneck):
        reduce_size = int(features / bottleneck)
        double_features = int(2 * features)
        self.link = nn.Sequential(
            nn.Conv2d(double_features, reduce_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(reduce_size, double_features, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(double_features, features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features)
        )

        nn.init.kaiming_uniform_(self.link[0].weight)
        nn.init.kaiming_uniform_(self.link[2].weight)
        nn.init.kaiming_uniform_(self.final_conv[0].weight)

    def forward(self, x1, x2):
        x_12 = torch.cat((x1, x2), dim=1)

        x_12_est = self.link(x_12)
        x_12 = x_12 * x_12_est
        x_12 = self.final_conv(x_12)

        return x_12
