import torch
from torch import nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from rep_unit import BottleneckSSMA
from decoder import Decoder
from easpp import eASPP

class SSMA(nn.Module):
    def __init__(self, C):
        super(SSMA, self).__init__()
        # variables
        self.num_categories = C

        self.encoder_mod1 = self._create_encoder()
        self.encoder_mod2 = self._create_encoder()
        self.eASPP = eASPP()
        self.decoder = Decoder(self.num_categories)


        self.ssma_sizes = [(24, 6), (24, 6), (2048, 16)]
        self.ssma_blocks = nn.ModuleList([])
        self._init_ssma(self.ssma_blocks, self.ssma_sizes)
        for i, block in enumerate(self.ssma_blocks):
            if str(type(block)) == "<class 'torch.nn.modules.conv.Conv2d'>":
                nn.init.kaiming_uniform_(block.weight, nonlinearity="relu")

        self.enc_skip2_conv_m1 = nn.Conv2d(256, 24, kernel_size=1, stride=1)
        self.enc_skip2_conv_bn_m1 = nn.BatchNorm2d(24)
        self.enc_skip1_conv_m1 = nn.Conv2d(512, 24, kernel_size=1, stride=1)
        self.enc_skip1_conv_bn_m1 = nn.BatchNorm2d(24)

        self.enc_skip2_conv_m2 = nn.Conv2d(256, 24, kernel_size=1, stride=1)
        self.enc_skip2_conv_bn_m2 = nn.BatchNorm2d(24)
        self.enc_skip1_conv_m2 = nn.Conv2d(512, 24, kernel_size=1, stride=1)
        self.enc_skip1_conv_bn_m2 = nn.BatchNorm2d(24)

        nn.init.kaiming_uniform_(self.enc_skip2_conv_m1.weight)
        nn.init.kaiming_uniform_(self.enc_skip1_conv_m1.weight)
        nn.init.kaiming_uniform_(self.enc_skip2_conv_m2.weight)
        nn.init.kaiming_uniform_(self.enc_skip1_conv_m2.weight)

    def _create_encoder(self):
        res_n50_enc = resnet50(pretrained=True)
        res_n50_enc.layer2[-1] = BottleneckSSMA(512, 128, 1, 2, 64, copy_from=res_n50_enc.layer2[-1])

        u3_sizes_short = [(256, 1, 256, 2, 1024), (256, 1, 256, 16, 1024), (256, 1, 256, 8, 1024),
                          (256, 1, 256, 4, 1024)]
        for i, x in enumerate(u3_sizes_short):
            res_n50_enc.layer3[2 + i] = BottleneckSSMA(x[-1], x[0], x[1], x[3], x[2],
                                                        copy_from=res_n50_enc.layer3[2 + i])

        u3_sizes_block = [(512, 2, 512, 4, 2048), (512, 2, 512, 8, 2048), (512, 2, 512, 16, 2048)]
        for i, res in enumerate(u3_sizes_block):
            downsample = None
            if i == 0:
                downsample = res_n50_enc.layer4[0].downsample
                downsample[0].stride = (1, 1)

            res_n50_enc.layer4[i] = BottleneckSSMA(res[-1], res[0], res[1], res[3], res[2], downsample=downsample,
                                                    copy_from=res_n50_enc.layer4[i])

        return res_n50_enc

    def _init_ssma(self, blocks, sizes):
        for x in sizes:
            siz2 = int(2 * x[0])
            sizRe = int(x[0] / x[1])
            cur_ssma = nn.ModuleList([
                nn.Conv2d(siz2, sizRe, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(sizRe, siz2, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(siz2, x[0], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(x[0])
            ])
            blocks.append(cur_ssma)



    def forward(self, mod1, mod2):
        # output x for eASPP, skip connection 1 and skip connection 2
        m1_x, m1_s1, m1_s2 = self.encode(mod1, self.encoder_mod1,
                                         self.enc_skip2_conv_bn_m1,
                                         self.enc_skip2_conv_m1,
                                         self.enc_skip1_conv_bn_m1,
                                         self.enc_skip1_conv_m1)
        m2_x, m2_s1, m2_s2 = self.encode(mod2, self.encoder_mod2,
                                         self.enc_skip2_conv_bn_m2,
                                         self.enc_skip2_conv_m2,
                                         self.enc_skip1_conv_bn_m2,
                                         self.enc_skip1_conv_m2)

        ssma_s1 = self.fusion_ssma(m1_s1, m2_s1, self.ssma_blocks[0])
        ssma_s2 = self.fusion_ssma(m1_s2, m2_s2, self.ssma_blocks[1])
        ssma_x = self.fusion_ssma(m1_x, m2_x, self.ssma_blocks[2])

        ssma_x = self.eASPP(ssma_x)

        aux1, aux2, res = self.Decoder(ssma_x, ssma_s1, ssma_s2)

        return aux1, aux2, res

    def fusion_ssma(self, x1, x2, ssma_block):
        x_12 = torch.cat((x1, x2), dim=1)
        xc_12 = F.relu(ssma_block[0](x_12))
        xc_12 = torch.sigmoid(ssma_block[1](xc_12))
        x_12 = x_12 * xc_12
        x_12 = ssma_block[3](ssma_block[2](x_12))
        return x_12


    def encode(self, x, encoder, s2_bn, s2_c, s1_bn, s1_c):
        x = encoder.conv1(x)
        x = encoder.bn1(x)
        x = encoder.relu(x)
        x = encoder.maxpool(x)

        x = encoder.layer1(x)  # this connection goes to conv and then decoder/skip 2
        s2 = s2_bn(s2_c(x))

        x = encoder.layer2(x)
        s1 = s1_bn(s1_c(x))

        x = encoder.layer3(x)

        x = encoder.layer4(x)

        return x, s1, s2

