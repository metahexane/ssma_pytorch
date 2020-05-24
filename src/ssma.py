import torch
from torch import nn as nn
import torch.nn.functional as F


class SSMA(nn.Module):
    def __init__(self, C):
        # variables
        self.num_categories = C

        # encoder layers
        self.enc_conv_1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.enc_conv_1_bn = nn.BatchNorm2d(64)
        self.max_pool_2x2 = nn.MaxPool2d(2)

        self.u2_sizes_short = [(64, 256), (64, 256), (128, 512), (128, 512), (256, 1024)]
        self.u2_sizes_block = [(128, 512), (256, 1024)]
        self.enc_u2_short = []
        self.enc_u2_block = []

        self.u3_sizes_short = [(128, 1, 64, 2, 512), (256, 1, 256, 2, 1024), (256, 1, 256, 16, 1024),
                               (256, 1, 256, 8, 1024),
                               (256, 1, 256, 4, 1024), (512, 2, 512, 8, 2048), (512, 2, 512, 16, 2048)]
        self.u3_sizes_block = [(512, 2, 512, 4, 2048)]
        self.enc_u3_short = []
        self.enc_u3_block = []

        self.ssma_sizes = [(24, 6), (24, 6), (2048, 16)]
        self.ssma_blocks = []
        self._init_ssma(self.ssma_blocks, self.ssma_sizes)

        self.integrate_fuse_skip_sizes = [(256, 24), (256, 24)]
        self.integrate_fuse_skip_blocks = []
        self._init_integrate_fuse_skip(self.integrate_fuse_skip_blocks, self.integrate_fuse_skip_sizes)

        self.eASPP_atrous_branches_r = [3, 6, 12]
        self.eASPP_branches = []
        self._init_eASPP_branches(self.eASPP_branches, self.eASPP_atrous_branches_r)
        self.eASPP_fin_conv = nn.Conv2d(1280, 256, kernel_size=1)
        self.eASPP_fin_conv_bn = nn.BatchNorm2d(256)

        self._init_u1()
        self._init_u2(self.enc_u2_short, self.u2_sizes_short, s=1)
        self._init_u2(self.enc_u2_block, self.u2_sizes_block, s=2)

        self._init_u3(self.u3_sizes_short, self.u3_sizes_short)
        self._init_u3(self.u3_sizes_block, self.u3_sizes_block)

        self.enc_skip1_conv = nn.Conv2d(256, 24, kernel_size=1, stride=1)
        self.enc_skip1_conv_bn = nn.BatchNorm2d(24)
        self.enc_skip2_conv = nn.Conv2d(512, 24, kernel_size=1, stride=1)
        self.enc_skip2_conv_bn = nn.BatchNorm2d(24)

        # decoder layers
        self.dec_deconv_1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2)  # kernel-size as defined in og-code
        self.dec_deconv_1_bn = nn.BatchNorm2d(256)
        self.dec_deconv_2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2)
        self.dec_deconv_2_bn = nn.BatchNorm2d(256)
        self.dec_deconv_3 = nn.ConvTranspose2d(self.num_categories, self.num_categories, kernel_size=8, stride=4)
        self.dec_deconv_3_bn = nn.BatchNorm2d(self.num_categories)
        self.dec_conv_1 = nn.Conv2d(280, 256, 3, padding=1)
        self.dec_conv_1_bn = nn.BatchNorm2d(280)
        self.dec_conv_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.dec_conv_2_bn = nn.BatchNorm2d(256)
        self.dec_conv_3 = nn.Conv2d(280, 256, 3, padding=1)
        self.dec_conv_3_bn = nn.BatchNorm2d(280)
        self.dec_conv_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.dec_conv_4_bn = nn.BatchNorm2d(256)
        self.dec_conv_5 = nn.Conv2d(256, self.num_categories, 1)
        self.dec_conv_5_bn = nn.BatchNorm2d(self.num_categories)

        # decoder auxiliary layers
        self.aux_conv1 = nn.Conv2d(256, self.num_categories, 1)
        self.aux_conv1_bn = nn.BatchNorm2d(256)
        self.aux_conv2 = nn.Conv2d(256, self.num_categories, 1)
        self.aux_conv2_bn = nn.BatchNorm2d(256)

    def _init_u1(self):
        self.enc_conv_u1_1 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.enc_conv_u1_1_bn = nn.BatchNorm2d(64)
        self.enc_conv_u1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.enc_conv_u1_2_bn = nn.BatchNorm2d(64)
        self.enc_conv_u1_3 = nn.Conv2d(64, 256, kernel_size=1, stride=1)
        self.enc_conv_u1_4 = nn.Conv2d(64, 256, kernel_size=1, stride=1)

    def _init_u2(self, arry, sizes, s=1):
        for i, x in enumerate(sizes):
            u2_comps = [
                nn.BatchNorm2d(x[1]),
                nn.Conv2d(x[1], x[0], kernel_size=1, stride=1),
                nn.BatchNorm2d(x[0]),
                nn.Conv2d(x[0], x[0], kernel_size=3, stride=s, padding=1),
                nn.BatchNorm2d(x[0]),
                nn.Conv2d(x[0], x[1], kernel_size=1, stride=1),
            ]
            if s == 2:
                # convolution in link
                u2_comps.append(nn.Conv2d(x[1], x[1], kernel_size=1, stride=s))
            arry.append(u2_comps)

    def _init_u3(self, arry, sizes, block=False):
        for i, x in enumerate(sizes):
            u3_comps = [
                nn.BatchNorm2d(x[-1]),
                nn.Conv2d(x[-1], x[0], kernel_size=1, stride=1),
                nn.BatchNorm2d(x[0]),
                nn.Conv2d(x[0], x[2] / 2, dilation=x[1], kernel_size=3, stride=1, padding=1),
                # if dilation: more padding
                nn.BatchNorm2d(x[2] / 2),
                nn.Conv2d(x[0], x[2] / 2, dilation=x[3], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(x[2] / 2),
                nn.Conv2d(x[2], x[-1], kernel_size=1, stride=1)
            ]
            if block:
                u3_comps.append(nn.Conv2d(x[0], x[-1], kernel_size=1, stride=1))
            arry.append(u3_comps)

    def _init_ssma(self, blocks, sizes):
        for x in sizes:
            cur_ssma = [
                nn.Conv2d(2 * x[0], x[0] / x[1], kernel_size=3, stride=1, padding=1),
                nn.Conv2d(x[0] / x[1], 2 * x[0], kernel_size=3, stride=1, padding=1),
                nn.Conv2d(2 * x[0], x[0], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(x[0])
            ]
            blocks.append(cur_ssma)

    def _init_eASPP_branches(self, blocks, rates):
        # branch 1: 1x1 convolution
        blocks.append([
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.BatchNorm2d(256)
        ])
        # branch 2-4: atrous pooling branches
        for rate in rates:
            atrous_branch = [
                nn.Conv2d(2048, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, dilation=rate, padding=rate),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, dilation=rate, padding=rate),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 256, kernel_size=1),
                nn.BatchNorm2d(64),
            ]
            blocks.append(atrous_branch)
        # branch 5: image pooling, image-level feature (ParseNet)
        blocks.append([
            nn.Conv2d(2048, 256, 1),
            nn.BatchNorm2d(256)
        ])

    def _init_integrate_fuse_skip(self, blocks, sizes):
        for x in sizes:
            cur_ifs = [
                nn.Conv2d(x[0], x[1], 1),
                nn.BatchNorm2d(x[1])
            ]
            blocks.append(cur_ifs)

    def forward(self, mod1, mod2):
        # output x for eASPP, skip connection 1 and skip connection 2
        m1_x, m1_s1, m1_s2 = self.encode(mod1)
        m2_x, m2_s1, m2_s2 = self.encode(mod2)

        ssma_s1 = self.fusion_ssma(m1_s1, m2_s1, self.ssma_blocks[0])
        ssma_s2 = self.fusion_ssma(m1_s2, m2_s2, self.ssma_blocks[1])
        ssma_x = self.fusion_ssma(m1_x, m2_x, self.ssma_blocks[2])

        ssma_x = self.eASPP(ssma_x)

        y1, y2, y3 = self.decode(ssma_x, ssma_s1, ssma_s2)

    def fusion_ssma(self, x1, x2, ssma_block):
        x_12 = torch.cat((x1, x2), 2)
        xc_12 = F.relu(ssma_block[0](x_12))
        xc_12 = F.sigmoid(ssma_block[1](xc_12))
        x_12 = x_12 * xc_12
        x_12 = ssma_block[3](ssma_block[2](x_12))
        return x_12

    def eASPP(self, x):
        # branch 1: 1x1 convolution
        branch = self.eASPP_branches[0]
        out = torch.relu(branch[1](branch[0](x)))  # with or without relu?

        # branch 2-4: atrous pooling
        for i in range(1, 4):
            branch = self.eASPP_branches[i]
            y = torch.relu(branch[1](branch[0](x)))
            y = torch.relu(branch[3](branch[2](y)))
            y = torch.relu(branch[5](branch[4](y)))  # with or without relu?
            out = torch.cat((out, y), 1)

        # branch 5: image pooling
        branch = self.eASPP_branches[4]
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = torch.relu(branch[1](branch[0](x)))  # with or without relu?
        x = nn.Upsample((24, 48), mode="bilinear")(x)

        out = torch.cat((out, x), 1)

        return torch.relu(self.eASPP_fin_conv_bn(self.eASPP_fin_conv(out)))  # with or without relu?

    def encode(self, x):
        x = F.relu(self.enc_conv_1_bn(self.enc_conv_1(x)))
        x = self.max_pool_2x2(x)
        x = self.enc_unit_1(x)
        x = self.enc_unit_2(x, self.enc_u2_short[0], block=False)

        x = self.enc_unit_2(x, self.enc_u2_short[1],
                            block=False)  # this connection goes to conv and then decoder/skip 2
        s2 = self.enc_skip2_conv_bn(self.enc_skip2_conv(x))

        x = self.enc_unit_2(x, self.enc_u2_block[0], block=True)
        x = self.enc_unit_2(x, self.enc_u2_short[2], block=False)
        x = self.enc_unit_2(x, self.enc_u2_short[3], block=False)
        x = self.enc_unit_3(x, self.enc_u3_short[0],
                            block=False)  # this connection goes to conv and then decoder/skip 1
        s1 = self.enc_skip1_conv_bn(self.enc_skip1_conv(x))

        x = self.enc_unit_2(x, self.enc_u3_block[1], block=True)
        x = self.enc_unit_2(x, self.enc_u3_short[4], block=False)
        x = self.enc_unit_3(x, self.enc_u3_short[1], block=False)
        x = self.enc_unit_3(x, self.enc_u3_short[2], block=False)
        x = self.enc_unit_3(x, self.enc_u3_short[3], block=False)
        x = self.enc_unit_3(x, self.enc_u3_short[4], block=False)
        x = self.enc_unit_3(x, self.enc_u3_block[0], block=True)
        x = self.enc_unit_3(x, self.enc_u3_short[5], block=False)
        x = self.enc_unit_3(x, self.enc_u3_short[6], block=False)

        return x, s1, s2

    def decode(self, x, fuse_skip1, fuse_skip2):
        # stage 1
        x = self.dec_deconv_1_bn(self.dec_deconv_1(x))
        y1 = self.aux1(x)
        int_fuse_skip = self.integrate_fuse_skip(x, fuse_skip1, self.integrate_fuse_skip_blocks[0])
        x = torch.cat((x, int_fuse_skip), 1)

        # stage 2
        x = torch.relu(self.dec_conv_1_bn(self.dec_conv_1(x)))
        x = torch.relu(self.dec_conv_2_bn(self.dec_conv_2(x)))
        x = self.dec_conv_2_bn(self.dec_deconv_2(x))
        y2 = self.aux2(x)
        int_fuse_skip = self.integrate_fuse_skip(x, fuse_skip2, self.integrate_fuse_skip_blocks[1])
        x = torch.cat((x, int_fuse_skip), 1)

        # stage 3
        x = torch.relu(self.dec_conv_3_bn(self.dec_conv_3(x)))
        x = torch.relu(self.dec_conv_4_bn(self.dec_conv_4(x)))
        x = torch.relu(self.dec_conv_5_bn(self.dec_conv_5(x)))
        y3 = self.dec_deconv_3_bn(self.dec_deconv_3(x))

        return y1, y2, y3

    def enc_unit_1(self, x):
        x = F.relu(self.enc_conv_u1_1_bn(self.enc_conv_u1_1(x)))
        o1 = self.enc_conv_u1_4(x)
        x = F.relu(self.enc_conv_u1_2_bn(self.enc_conv_u1_2(x)))
        x = self.enc_conv_u1_3(x)
        return x + o1

    def enc_unit_2(self, x, unit, block=False):
        o1 = x.copy()
        if block:
            o1 = unit[-1](o1)
        x = F.relu(unit[0](x))
        x = F.relu(unit[2](unit[1](x)))
        x = F.relu(unit[4](unit[3](x)))
        x = unit[5](x)
        return x + o1

    def enc_unit_3(self, x, unit, block=True):
        o1 = x.copy()
        if block:
            o1 = unit[-1](o1)

        x = F.relu(unit[0](x))
        x = F.relu(unit[2](unit[1](x)))
        a1 = F.relu(unit[4](unit[3](x)))
        a2 = F.relu(unit[6](unit[5](x)))
        a = torch.cat((a1, a2), dim=2)
        x = unit[7](a)
        return x + o1

    def integrate_fuse_skip(self, x, fuse_skip, unit):
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = torch.relu(unit[1](unit[0](x)))

        return torch.mul(x, fuse_skip)

    def aux1(self, x):
        x = self.aux_conv1_bn(self.aux_conv1(x))

        return nn.UpsamplingBilinear2d(scale_factor=8)(x)

    def aux2(self, x):
        x = self.aux_conv1_bn(self.aux_conv1(x))

        return nn.UpsamplingBilinear2d(scale_factor=4)(x)
