import torch
import torch.nn as nn
import torch.nn.functional as F
from  transformer import TransformerDecoderLayer
from cswin_transformer import CSWin,CSWinDe

from MIRNet import *
class UNet_emb(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3,bias=False):
        """Initializes U-Net."""

        super(UNet_emb, self).__init__()
        self.cswin = CSWin(patch_size=2, embed_dim=48, depth=[4,4,4,4],
                           split_size=[1, 2, 4, 8], num_heads=[8, 16, 32,64], mlp_ratio=4., qkv_bias=True, qk_scale=None, #[8, 16, 32,64]
                           drop_rate=0., attn_drop_rate=0.,
                           drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False)

        self.cswinde = CSWinDe(patch_size=2, embed_dim=48*8, depth=[4, 4, 4],
                           split_size=[4, 2, 1], num_heads=[32, 16, 8], mlp_ratio=4., qkv_bias=True, qk_scale=None,
                           drop_rate=0., attn_drop_rate=0.,
                           drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False)

        #
        self.conv3 = nn.Conv2d(48, 32, 3, stride=1, padding=1)#48
        self.ReLU3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.ReLU4 = nn.ReLU(inplace=True)


        self._block5= nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2))#nn.UpsamplingBilinear2d(scale_factor=2)
        self.MSRB4 = MSRB(32, 3, 1, 2, bias)
        self._block7= nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1))

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def Encoder(self, x):  #Encoder  #forward
        """Through encoder, then decoder by adding U-skip connections. """
        swin_in = x  # 96,192,384,768

        swin_out_1 = self.cswin(swin_in)  #,swin_input_1
        swin_out_2 = self.cswinde(swin_out_1)  # ,swin_input_1

        # csde_out = self.cswinde(swin_out_1)
        decoder_1 = self.ReLU3(self.conv3 (swin_out_2[2]))  # 64
        upsample5 = self._block5(decoder_1)  # 32
        upsample5 = self.MSRB4(upsample5)
        decoder_0 = self.ReLU4(self.conv4(upsample5))  # 48

        result = self._block7(decoder_0)+x  # 23
        return result



           
class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                # nn.AdaptiveAvgPool2d(bin),
                # nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),  # (bin*2+1)
                nn.Conv2d(in_dim, reduction_dim, kernel_size=(bin), bias=False),#(bin*2+1)
                #nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)


    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))   #what is f(x)
        return torch.cat(out, 1)         
          
