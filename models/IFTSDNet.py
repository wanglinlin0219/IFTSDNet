"""
IFTSDNet
please Cited the paper:
L. Wang, J. Zhang, Q. Guo, and D. Chen.
"IFTSDNet: An Interact-Feature Transformer Network With Spatial Detail Enhancement Module for Change Detection," in IEEE Geoscience and Remote Sensing Letters, vol. 20, pp. 1-5, 2023
"""

import torch
import torch.nn as nn
from .Transformer import TransformerDecoder, Transformer
from einops import rearrange
from models.IFlayer import Conv_IF_Layer
import matplotlib.pyplot as plt


class BasicConv2d(nn.Module):
    
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class FEC(nn.Module):
    """feature extraction cell"""

    def __init__(self, in_ch, mid_ch, out_ch):
        super(FEC, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output

class token_encoder(nn.Module):
    def __init__(self, in_chan = 64, token_len = 4, heads = 8):
        super(token_encoder, self).__init__()
        self.token_len = token_len
        self.conv_a = nn.Conv2d(in_chan, token_len, kernel_size=1, padding=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, token_len, in_chan))
        self.transformer = Transformer(dim=in_chan, depth=1, heads=heads, dim_head=64, mlp_dim=64, dropout=0)

    def forward(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()

        tokens = torch.einsum('bln, bcn->blc', spatial_attention, x)

        tokens += self.pos_embedding
        x = self.transformer(tokens)
        return x

class token_decoder(nn.Module):
    def __init__(self, in_chan = 64, size = 16, heads = 8):
        super(token_decoder, self).__init__()
        self.pos_embedding_decoder = nn.Parameter(torch.randn(2, in_chan, size, size))  # Adjust the first value, ”2“, based on the configured batch size.
        self.transformer_decoder = TransformerDecoder(dim=in_chan, depth=1, heads=heads, dim_head=True, mlp_dim=in_chan*2, dropout=0, softmax=in_chan)

    def forward(self, x, m):
        b, c, h, w = x.shape

        x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

class context_aggregator(nn.Module):
    def __init__(self, in_chan=64, size=16):
        super(context_aggregator, self).__init__()
        self.token_encoder = token_encoder(in_chan=in_chan, token_len=4)
        self.token_decoder = token_decoder(in_chan = 64, size = size, heads = 8)

    def forward(self, feature):
        token = self.token_encoder(feature)
        feature.size()
        token.size()
        out = self.token_decoder(feature, token)
        return out

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(3*out_channel, out_channel, 3, padding=1)  # 原来是乘以4
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class FEBlock1(nn.Module):
    """MSPSNet"""
    def __init__(self, in_ch=3, ou_ch=2, patch_size=256):
        super(FEBlock1, self).__init__()
        torch.nn.Module.dump_patches = True
        self.patch_size = patch_size
        n1 = 40  # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]  # 40, 80, 160, 320, 640

        self.conv0_0 = nn.Conv2d(3, n1, kernel_size=5, padding=2, stride=1)  #先变成256✖256✖40
        self.conv0 = FEC(filters[0], filters[0], filters[0]) #40

        self.conv2 = FEC(filters[0], filters[1], filters[1])

        self.conv4 = FEC(filters[1], filters[2], filters[2])
        self.conv5 = FEC(filters[2], filters[3], filters[3])
        self.conv6 = nn.Conv2d(232, filters[1], kernel_size=1, stride=1)
        self.conv7 = nn.Conv2d(filters[1], ou_ch, kernel_size=3, padding=1, bias=False)

        self.conv6_4 = nn.Conv2d(64, filters[1], kernel_size=1, stride=1)
        self.conv7_4 = nn.Conv2d(filters[1], ou_ch, kernel_size=3, padding=1, bias=False)

        self.conv6_3 = nn.Conv2d(128, filters[1], kernel_size=1, stride=1)
        self.conv7_3 = nn.Conv2d(filters[1], ou_ch, kernel_size=3, padding=1, bias=False)

        self.conv6_2 = nn.Conv2d(192, filters[1], kernel_size=1, stride=1)
        self.conv7_2 = nn.Conv2d(filters[1], ou_ch, kernel_size=3, padding=1, bias=False)


        self.conv1_1 = nn.Conv2d(filters[0] * 2, filters[0], kernel_size=1, stride=1)


        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(filters[1], 32)
        self.rfb3_1 = RFB_modified(filters[2], 32)
        self.rfb4_1 = RFB_modified(filters[3], 32)

        self.pool1 = nn.AdaptiveAvgPool2d(128)
        self.pool2 = nn.AdaptiveAvgPool2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d(32)

        # self.pool1 = nn.AdaptiveAvgPool2d(256)
        # self.pool2 = nn.AdaptiveAvgPool2d(128)
        # self.pool3 = nn.AdaptiveAvgPool2d(64)

        self.Up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.Up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.Up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

        self.CA_s4 = context_aggregator(in_chan=64, size=128)
        self.CA_s8 = context_aggregator(in_chan=64, size=64)
        self.CA_s16 = context_aggregator(in_chan=64, size=32)

        # self.CA_s4 = context_aggregator(in_chan=64, size=256)
        # self.CA_s8 = context_aggregator(in_chan=64, size=128)
        # self.CA_s16 = context_aggregator(in_chan=64, size=64)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        x = x1
        x1 = self.conv0(self.conv0_0(x1))  # Output of the first scale
        x3 = self.conv2(self.pool1(x1))
        x4 = self.conv4(self.pool2(x3))
        A_F4 = self.conv5(self.pool3(x4))

        # The second branch

        x2 = self.conv0(self.conv0_0(x2))
        x5 = self.conv2(self.pool1(x2))
        x6 = self.conv4(self.pool2(x5))
        A_F8 = self.conv5(self.pool3(x6))

        # TEM
        x3_rfb = self.rfb2_1(x3)  # channel -> 32
        x5_rfb = self.rfb2_1(x5)  # channel -> 32
        x4_rfb = self.rfb3_1(x4)  # channel -> 32
        x6_rfb = self.rfb3_1(x6)  # channel -> 32
        A_F4_rfb = self.rfb4_1(A_F4)  # channel -> 32
        A_F8_rfb = self.rfb4_1(A_F8)  # channel -> 32


        x3_x5_s1 = self.CA_s4(torch.cat([x3_rfb, x5_rfb], 1)) #修改一下，后融合；
        x4_x6_s2 = self.CA_s8(torch.cat([x4_rfb, x6_rfb], 1))
        A_F4_A_F8_s4 = self.CA_s16(torch.cat([A_F4_rfb, A_F8_rfb], 1))



        c4 = A_F4_A_F8_s4
        c3 = torch.cat([x4_x6_s2, self.Up1(c4)], 1)
        c2 = torch.cat([x3_x5_s1, self.Up1(c3)], 1)

        c4 = self.conv6_4(self.Up3(c4))
        out4 = self.conv7_4(c4)
        c3 = self.conv6_3(self.Up2(c3))
        out3 = self.conv7_3(c3)
        c2 = self.conv6_2(self.Up1(c2))
        out2 = self.conv7_2(c2)

        return (out2,), (out3,), (out4,)  # 改损失的时候可能需要改形式
















