from torch import nn as nn
from components.decoder import Decoder
from components.easpp import eASPP
from components.encoder import Encoder
from components.ssma import SSMA

class AdapNet(nn.Module):
    """PyTorch module for 'AdapNet++' and 'AdapNet++ with fusion architecture' """

    def __init__(self, C, encoders=[]):
        """Constructor

        :param C: number of categories
        :param encoders: array of zero or two encoders. If the array is empty, this class is effectively AdapNet++ and a
                         new encoder will be initialized (used in stage 1). If the array has two encoders, this class
                         is effectively AdapNet++ with fusion architecture and it will use the two pre-trained encoders
                         with SSMA fusion (used in stages 2 and 3).
        """
        super(AdapNet, self).__init__()

        self.num_categories = C
        self.fusion = False

        if len(encoders) > 0:
            self.encoder_mod1 = encoders[0]
            self.encode_mod1.layer3[2].dropout = False
            self.encoder_mod2 = encoders[1]
            self.encode_mod1.layer3[2].dropout = False
            self.ssma_s1 = SSMA(24, 6)
            self.ssma_s2 = SSMA(24, 6)
            self.ssma_res = SSMA(2048, 16)
            self.fusion = True
        else:
            self.encoder_mod1 = Encoder()

        self.eASPP = eASPP()
        self.decoder = Decoder(self.num_categories, self.fusion)

    def forward(self, mod1, mod2=None):
        """Forward pass

        In the case of AdapNet++, only 1 modality is used (either the RGB-image, or the Depth-image). With 'AdapNet++
        with fusion architechture' two modalities are used (both the RGB-image and the Depth-image).

        :param mod1: modality 1
        :param mod2: modality 2
        :return: final output and auxiliary output 1 and 2
        """
        m1_x, skip2, skip1 = self.encoder_mod1(mod1)

        if self.fusion:
            m2_x, m2_s2, m2_s1 = self.encoder_mod2(mod2)
            skip2 = self.ssma_s2(skip2, m2_s2)
            skip1 = self.ssma_s1(skip1, m2_s1)
            m1_x = self.ssma_res(m1_x, m2_x)

        m1_x = self.eASPP(m1_x)

        aux1, aux2, res = self.decoder(m1_x, skip1, skip2)

        return aux1, aux2, res
