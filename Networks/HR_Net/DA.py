import torch
import torch.nn as nn
# import models.common as common

import torchvision.ops.deform_conv as dc

class RCA_Block(nn.Module):
    def __init__(self, features):
        super(RCA_Block, self).__init__()
        firstblock = []
        firstblock.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False))
        firstblock.append(nn.PReLU())
        firstblock.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False))
        self.firstblock = nn.Sequential(*firstblock)

        ch_attention = []
        ch_attention.append(nn.AdaptiveAvgPool2d(1))
        ch_attention.append(nn.Conv2d(in_channels=features, out_channels=features // 16, kernel_size=1, padding=0))
        ch_attention.append(nn.PReLU())
        ch_attention.append(nn.Conv2d(in_channels=features // 16, out_channels=features, kernel_size=1, padding=0))
        ch_attention.append(nn.Sigmoid())
        self.cab = nn.Sequential(*ch_attention)

    def forward(self, x):
        residual = x
        data = self.firstblock(x)
        ch_data = self.cab(data) * data
        out = ch_data + residual

        return out


class _down(nn.Module):
    def __init__(self, nchannel):
        super(_down, self).__init__()
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(in_channels=nchannel, out_channels=2*nchannel, kernel_size=1, stride=1, padding=0)
        self.conv_down = nn.Conv2d(in_channels=nchannel, out_channels=2 * nchannel, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.prelu(self.conv_down(x))

        return out

class _up(nn.Module):
    def __init__(self, nchannel):
        super(_up, self).__init__()
        self.prelu = nn.PReLU()
        self.subpixel = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(in_channels=nchannel, out_channels=nchannel, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=nchannel, out_channels=nchannel//2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.prelu(self.subpixel(x))

        return out

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class RCAB(nn.Module):
    def __init__(self, nchannel):
        super(RCAB, self).__init__()
        self.RCAB = RCA_Block(nchannel)
    def forward(self, x):
        out = self.RCAB(x)

        return out

class deform_conv(nn.Module): # Deformed Convolution Attention Block
    def __init__(self, features):
        super(deform_conv, self).__init__()
        groups = 8
        kernel_size = 3

        self.prelu = nn.PReLU()
        self.offset_conv1 = nn.Conv2d(features, 2*kernel_size*kernel_size, kernel_size=3, stride=1, padding=1, bias=True)
        self.deconv1 = dc.DeformConv2d(features, features, kernel_size=3, stride=1, padding=1, dilation=1,
                                   groups=groups)

    def forward(self, x):
        # deform conv
        offset1 = self.prelu(self.offset_conv1(x))
        out = self.deconv1(x, offset1)

        return out

class DAB(nn.Module): # Deformed Convolution Attention Block
    def __init__(self, features):
        super(DAB, self).__init__()
        groups = 8
        kernel_size = 3

        self.prelu = nn.PReLU()
        self.offset_conv1 = nn.Conv2d(features, 2*kernel_size*kernel_size, kernel_size=3, stride=1, padding=1, bias=True)
        self.deconv1 = dc.DeformConv2d(features, features, kernel_size=3, stride=1, padding=1, dilation=1,
                                   groups=groups)
        self.conv = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        residual = x
        # deform conv
        offset1 = self.prelu(self.offset_conv1(x))
        feat_deconv1 = self.deconv1(x, offset1)

        # attention
        atten_conv = self.conv(x)
        atten_feat = self.softmax(atten_conv)

        out = atten_feat * feat_deconv1
        out = out + residual

        return out


class DCCAB(nn.Module): # Deformed Convolution Attention Block
    def __init__(self, features):
        super(DCCAB, self).__init__()
        self.dab = DAB(features)
        self.cab = RCAB(features)

    def forward(self, x):
        residual = x
        # deform conv
        dabdata = self.dab(x)
        cabdata = self.cab(dabdata)

        out = cabdata + residual

        return out