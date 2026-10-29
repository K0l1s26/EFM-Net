import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from model.Modules.basic_modules import DSCResBlock, DownsampleBlock
from thop import profile


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class OutConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()

        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class Up(nn.Module):

    def __init__(self, in_ch, skip_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch+skip_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Baseline(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Baseline, self).__init__()
        features = [32, 64, 128, 256]


        self.en1 = DSCResBlock(in_channels, features[0])
        self.en2 = DSCResBlock(features[0], features[1])  #  32→64
        self.en3 = DSCResBlock(features[1], features[2])  # 64→128
        self.en4 = DSCResBlock(features[2], features[3])  # 128→256
        self.en5 = DSCResBlock(features[3], features[3])  # 256→256

        # 下采样
        self.down1 = DownsampleBlock(features[0], features[0])  # 128→64
        self.down2 = DownsampleBlock(features[1], features[1])  # 64→32
        self.down3 = DownsampleBlock(features[2], features[2])  # 32→16
        self.down4 = DownsampleBlock(features[3], features[3])  # 16→8
        self.down5 = DownsampleBlock(features[3], features[3]) # 8→4

        # 解码阶段（上采样）
        self.up1 = Up(features[3], features[3], features[2])  # 256+256→128
        self.up2 = Up(features[2], features[2], features[1])  # 128+128→64
        self.up3 = Up(features[1], features[1], features[0])  # 64+64→32
        self.up4 = Up(features[0], features[0], features[0])  # 32+32→32
        self.up5 = Up(features[0], features[0], features[0])  # 32+32→32
        self.outc = OutConv(features[0], num_classes)  # 32→4

    def forward(self, x):
        x0 = self.en1(x)
        x1 = self.down1(x0)
        x2 = self.down2(self.en2(x1))
        x3 = self.down3(self.en3(x2) )
        x4 = self.down4(self.en4(x3))
        x5 = self.down5(self.en5(x4))

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x, x0)
        x = self.outc(x)
        return x
