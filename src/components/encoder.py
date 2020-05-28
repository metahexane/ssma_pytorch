from torchvision.models import resnet50
import torch.nn as nn
from components.rep_unit import BottleneckSSMA

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.enc_skip2_conv = nn.Conv2d(256, 24, kernel_size=1, stride=1)
        self.enc_skip2_conv_bn = nn.BatchNorm2d(24)
        self.enc_skip1_conv = nn.Conv2d(512, 24, kernel_size=1, stride=1)
        self.enc_skip1_conv_bn = nn.BatchNorm2d(24)

        nn.init.kaiming_uniform_(self.enc_skip2_conv.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.enc_skip1_conv.weight, nonlinearity="relu")

        self.res_n50_enc = resnet50(pretrained=True)
        self.res_n50_enc.layer2[-1] = BottleneckSSMA(512, 128, 1, 2, 64, copy_from=self.res_n50_enc.layer2[-1])

        u3_sizes_short = [(256, 1, 256, 2, 1024), (256, 1, 256, 16, 1024), (256, 1, 256, 8, 1024),
                          (256, 1, 256, 4, 1024)]
        for i, x in enumerate(u3_sizes_short):
            self.res_n50_enc.layer3[2 + i] = BottleneckSSMA(x[-1], x[0], x[1], x[3], x[2],
                                                       copy_from=self.res_n50_enc.layer3[2 + i])

        u3_sizes_block = [(512, 2, 512, 4, 2048), (512, 2, 512, 8, 2048), (512, 2, 512, 16, 2048)]
        for i, res in enumerate(u3_sizes_block):
            downsample = None
            if i == 0:
                downsample = self.res_n50_enc.layer4[0].downsample
                downsample[0].stride = (1, 1)

            self.res_n50_enc.layer4[i] = BottleneckSSMA(res[-1], res[0], res[1], res[3], res[2], downsample=downsample,
                                                   copy_from=self.res_n50_enc.layer4[i])

    def forward(self, x):
        x = self.res_n50_enc.conv1(x)
        x = self.res_n50_enc.bn1(x)
        x = self.res_n50_enc.relu(x)
        x = self.res_n50_enc.maxpool(x)

        x = self.res_n50_enc.layer1(x)  # this connection goes to conv and then decoder/skip 2
        s2 = self.enc_skip2_conv_bn(self.enc_skip2_conv(x))

        x = self.res_n50_enc.layer2(x)
        s1 = self.enc_skip1_conv_bn(self.enc_skip1_conv(x))

        x = self.res_n50_enc.layer3(x)

        x = self.res_n50_enc.layer4(x)

        return x, s2, s1