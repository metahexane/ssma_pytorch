from torch import nn as nn
from components.decoder import Decoder
from components.easpp import eASPP
from components.encoder import Encoder
from components.ssma import SSMA

class AdapNet(nn.Module):
    def __init__(self, C, encoders=[]):
        super(AdapNet, self).__init__()

        self.num_categories = C
        self.fusion = False

        if len(encoders) > 0:
            self.encoder_mod1 = encoders[0]
            self.encoder_mod2 = encoders[1]
            self.ssma_s1 = SSMA(24, 6)
            self.ssma_s2 = SSMA(24, 6)
            self.ssma_res = SSMA(2048, 16)
            self.fusion = True
        else:
            self.encoder_mod1 = Encoder()

        self.eASPP = eASPP()
        self.decoder = Decoder(self.num_categories)


    def forward(self, mod1, mod2=None):
        m1_x, skip2, skip1 = self.encoder_mod1(mod1)

        if self.fusion:
            m2_x, m2_s2, m2_s1 = self.encoder_mod2(mod2)
            skip2 = self.ssma_s2(skip2, m2_s2)
            skip1 = self.ssma_s1(skip1, m2_s1)
            m1_x = self.ssma_res(m1_x, m2_x)

        m1_x = self.eASPP(m1_x)

        aux1, aux2, res = self.decoder(m1_x, skip1, skip2)

        return aux1, aux2, res