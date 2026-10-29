import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="mamba_ssm")

import torch
import torch.nn as nn
from EFMNet import DSCResBlock,TPDMStage,Freqenhance,DownsampleBlock,UpsampleBlock,EdgeGuidedFusion,AttentionGate
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from baseline import Baseline

class EAFMNet(nn.Module):
    def __init__(self, in_channels,
                 num_classes,
                 m_depths=(2,2,1),
                 features=(16,32,64,128,256),
                 act="PRELU",
                 norm="GROUP",
                 groups=4,
                 order='POST',
                 drop_path_rate=0.1,
                 drop_p=0.0,
                 scale_factor=2,
                 mlp_ratio=2):
        super(EAFMNet, self).__init__()

        depths = [1, 1, m_depths[0], m_depths[1], m_depths[2], m_depths[1], m_depths[0], 1, 1]
        total_depth = sum(depths)
        half_len = total_depth // 2 + 1
        dpr_half1 = torch.linspace(0, drop_path_rate, half_len)
        dpr_half2 = torch.linspace(drop_path_rate, 0, half_len)[1:]
        dpr = [x.item() for x in torch.cat([dpr_half1, dpr_half2])]

        # During the encoding stage, the number of channels increases while the dimension remains unchanged.
        self.en1 = DSCResBlock(in_channels, features[0],
                               act=act,
                               norm=norm,
                               groups=groups,
                               order=order,
                               drop_p=drop_p,
                               drop_path=dpr[0])
        self.en2 = DSCResBlock(features[0], features[1],  # 16→32
                               act=act,
                               norm=norm,
                               groups=groups,
                               order=order,
                               drop_p=drop_p,
                               drop_path=dpr[1])
        dpr_idx = 2
        self.en3 = TPDMStage(features[1], features[2],depth=m_depths[0],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[0]])   # 32→64
        dpr_idx += m_depths[0]
        self.en4 = TPDMStage(features[2], features[3],depth=m_depths[1],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[1]])   # 64→128
        dpr_idx += m_depths[1]

        self.Bottleneck = TPDMStage(features[3], features[4],depth=m_depths[2],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[2]])   # 128→256
        dpr_idx += m_depths[2]

        self.freq1 = Freqenhance(features[0], prior='high')  # 16
        self.freq2 = Freqenhance(features[1], prior='high')  # 32
        self.freq3 = Freqenhance(features[2], prior='low')  # 64
        self.freq4 = Freqenhance(features[3], prior='low')  # 128

        # Downsample the number of channels remains the same, size is halved
        self.down1 = DownsampleBlock(features[0], features[0])  # 128→64
        self.down2 = DownsampleBlock(features[1], features[1])  # 64→32
        self.down3 = DownsampleBlock(features[2], features[2])  # 32→16
        self.down4 = DownsampleBlock(features[3], features[3])  # 16→8

        self.edgeguidedfusion = EdgeGuidedFusion(ch_list=[features[0], features[1], features[2], features[3]])

        self.up1 = UpsampleBlock(features[4], features[3], scale_factor=scale_factor)  # 256→128
        self.up2 = UpsampleBlock(features[3], features[2], scale_factor=scale_factor)  # 128→64
        self.up3 = UpsampleBlock(features[2], features[1], scale_factor=scale_factor)  # 64→32
        self.up4 = UpsampleBlock(features[1], features[0], scale_factor=scale_factor)  # 32→16

        self.att_g1 = AttentionGate(F_g=features[3], F_l=features[3], F_int=features[2])
        self.att_g2 = AttentionGate(F_g=features[2], F_l=features[2], F_int=features[1])
        self.att_g3 = AttentionGate(F_g=features[1], F_l=features[1], F_int=features[0])
        self.att_g4 = AttentionGate(F_g=features[0], F_l=features[0], F_int=features[0]//2)

        # decoder
        self.de1 = TPDMStage(features[4], features[3],depth=m_depths[1],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[1]])   # 256+128→128
        dpr_idx += m_depths[1]
        self.de2 = TPDMStage(features[3], features[2],depth=m_depths[0],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[0]])   # 128+64→64
        dpr_idx += m_depths[0]
        self.de3 = DSCResBlock(features[2], features[1],
                               act=act,
                               norm=norm,
                               groups=groups,
                               order=order,
                               drop_p=drop_p,
                               drop_path=dpr[dpr_idx])
        dpr_idx += 1
        self.de4 = DSCResBlock(features[1], features[0],
                               act=act,
                               norm=norm,
                               groups=groups,
                               order=order,
                               drop_p=drop_p,
                               drop_path=dpr[dpr_idx])

        self.out = nn.Conv3d(features[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        # Encoder
        x1 = self.en1(x)  # size：(16,128,128,128)
        x1 = self.freq1(x1)

        x2 = self.down1(x1)  # (16,64,64,64)
        x2 = self.en2(x2)    # (32,64,64,64)
        x2 = self.freq2(x2)

        x3 = self.down2(x2)  # (32,32,32,32)
        x3 = self.en3(x3)    # (64,32,32,32)
        x3 = self.freq3(x3)

        x4 = self.down3(x3)  # (64,16,16,16)
        x4 = self.en4(x4)    # (128,16,16,16)
        x4 = self.freq4(x4)

        x5 = self.down4(x4)  # (128,8,8,8)
        x5 = self.Bottleneck(x5)    # (256,8,8,8)


        # skip connection
        y1,y2,y3,y4 = self.edgeguidedfusion(x1,x2,x3,x4)

        # Decoder
        x = self.up1(x5)  # (128,16,16,16)
        skip1 = self.att_g1(x,y4)   # (128,16,16,16)
        x = torch.cat([x, skip1], dim=1)  # (256,16,16,16)
        x = self.de1(x)  # (128,16,16,16)

        x = self.up2(x)  # (64,32,32,32)
        skip2 = self.att_g2(x,y3)  # (64,32,32,32)
        x = torch.cat([x, skip2], dim=1) # (128,32,32,32)
        x = self.de2(x) # (64,32,32,32)

        x = self.up3(x)  # (32,64,64,64)
        skip3 = self.att_g3(x,y2) # (32,64,64,64)
        x = torch.cat([x, skip3], dim=1) # (64,64,64,64)
        x = self.de3(x)  # (32,64,64,64)

        x = self.up4(x)  # (16,128,128,128)
        skip4 = self.att_g4(x,y1)  # (16,128,128,128)
        x = torch.cat([x, skip4], dim=1) # (32,128,128,128)
        x = self.de4(x)  # (16,128,128,128)

        out = self.out(x)  # (3,128,128,128)

        return out

class MNet(nn.Module):
    def __init__(self, in_channels,
                 num_classes,
                 m_depths=(2,2,1),
                 features=(16,32,64,128,256),
                 act="PRELU",
                 norm="GROUP",
                 groups=4,
                 order='POST',
                 drop_path_rate=0.1,
                 drop_p=0.0,
                 scale_factor=2,
                 mlp_ratio=2):
        super(MNet, self).__init__()

        depths = [1, 1, m_depths[0], m_depths[1], m_depths[2], m_depths[1], m_depths[0], 1, 1]
        total_depth = sum(depths)  # 13
        half_len = total_depth // 2 + 1  # 对于13层，前半段是7层
        dpr_half1 = torch.linspace(0, drop_path_rate, half_len)
        dpr_half2 = torch.linspace(drop_path_rate, 0, half_len)[1:]
        dpr = [x.item() for x in torch.cat([dpr_half1, dpr_half2])]

        # During the encoding stage, the number of channels increases while the dimension remains unchanged.
        self.en1 = DSCResBlock(in_channels, features[0],    # 4→16 通道数增大
                               act=act,
                               norm=norm,
                               groups=groups,
                               order=order,
                               drop_p=drop_p,
                               drop_path=dpr[0])
        self.en2 = DSCResBlock(features[0], features[1],  # 16→32
                               act=act,
                               norm=norm,
                               groups=groups,
                               order=order,
                               drop_p=drop_p,
                               drop_path=dpr[1])
        dpr_idx = 2
        self.en3 = TPDMStage(features[1], features[2],depth=m_depths[0],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[0]])   # 32→64
        dpr_idx += m_depths[0]
        self.en4 = TPDMStage(features[2], features[3],depth=m_depths[1],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[1]])   # 64→128
        dpr_idx += m_depths[1]

        self.Bottleneck = TPDMStage(features[3], features[4],depth=m_depths[2],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[2]])   # 128→256
        dpr_idx += m_depths[2]

        # Downsample the number of channels remains the same, size is halved
        self.down1 = DownsampleBlock(features[0], features[0])
        self.down2 = DownsampleBlock(features[1], features[1])
        self.down3 = DownsampleBlock(features[2], features[2])
        self.down4 = DownsampleBlock(features[3], features[3])

        self.up1 = UpsampleBlock(features[4], features[3], scale_factor=scale_factor)
        self.up2 = UpsampleBlock(features[3], features[2], scale_factor=scale_factor)
        self.up3 = UpsampleBlock(features[2], features[1], scale_factor=scale_factor)
        self.up4 = UpsampleBlock(features[1], features[0], scale_factor=scale_factor)

        # decoder
        self.de1 = TPDMStage(features[4], features[3],depth=m_depths[1],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[1]])
        dpr_idx += m_depths[1]
        self.de2 = TPDMStage(features[3], features[2],depth=m_depths[0],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[0]])
        dpr_idx += m_depths[0]
        self.de3 = DSCResBlock(features[2], features[1],
                               act=act,
                               norm=norm,
                               groups=groups,
                               order=order,
                               drop_p=drop_p,
                               drop_path=dpr[dpr_idx])
        dpr_idx += 1
        self.de4 = DSCResBlock(features[1], features[0],
                               act=act,
                               norm=norm,
                               groups=groups,
                               order=order,
                               drop_p=drop_p,
                               drop_path=dpr[dpr_idx])

        self.out = nn.Conv3d(features[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        # Encoder
        x1 = self.en1(x)  # size：(16,128,128,128)

        x2 = self.down1(x1)  # (16,64,64,64)
        x2 = self.en2(x2)    # (32,64,64,64)

        x3 = self.down2(x2)  # (32,32,32,32)
        x3 = self.en3(x3)    # (64,32,32,32)

        x4 = self.down3(x3)  # (64,16,16,16)
        x4 = self.en4(x4)    # (128,16,16,16)

        x5 = self.down4(x4)  # (128,8,8,8)
        x5 = self.Bottleneck(x5)    # (256,8,8,8)


        # Decoder
        x = self.up1(x5)  # (128,16,16,16)
        x = torch.cat([x, x4], dim=1)  # (256,16,16,16)
        x = self.de1(x)  # (128,16,16,16)

        x = self.up2(x)  # (64,32,32,32)

        x = torch.cat([x, x3], dim=1) # (128,32,32,32)
        x = self.de2(x) # (64,32,32,32)

        x = self.up3(x)  # (32,64,64,64)

        x = torch.cat([x, x2], dim=1) # (64,64,64,64)
        x = self.de3(x)  # (32,64,64,64)

        x = self.up4(x)  # (16,128,128,128)

        x = torch.cat([x, x1], dim=1) # (32,128,128,128)
        x = self.de4(x)  # (16,128,128,128)

        out = self.out(x)  # (3,128,128,128)

        return out

class FMNet(nn.Module):
    def __init__(self, in_channels,
                 num_classes,
                 m_depths=(2,2,1),
                 features=(16,32,64,128,256),
                 act="PRELU",
                 norm="GROUP",
                 groups=4,
                 order='POST',
                 drop_path_rate=0.1,
                 drop_p=0.0,
                 scale_factor=2,
                 mlp_ratio=2):
        super(FMNet, self).__init__()

        depths = [1, 1, m_depths[0], m_depths[1], m_depths[2], m_depths[1], m_depths[0], 1, 1]
        total_depth = sum(depths)  # 13
        half_len = total_depth // 2 + 1  # 对于13层，前半段是7层
        dpr_half1 = torch.linspace(0, drop_path_rate, half_len)
        dpr_half2 = torch.linspace(drop_path_rate, 0, half_len)[1:]
        dpr = [x.item() for x in torch.cat([dpr_half1, dpr_half2])]
        self.en1 = DSCResBlock(in_channels, features[0],
                               act=act,
                               norm=norm,
                               groups=groups,
                               order=order,
                               drop_p=drop_p,
                               drop_path=dpr[0])
        self.en2 = DSCResBlock(features[0], features[1],
                               act=act,
                               norm=norm,
                               groups=groups,
                               order=order,
                               drop_p=drop_p,
                               drop_path=dpr[1])
        dpr_idx = 2
        self.en3 = TPDMStage(features[1], features[2],depth=m_depths[0],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[0]])
        dpr_idx += m_depths[0]
        self.en4 = TPDMStage(features[2], features[3],depth=m_depths[1],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[1]])
        dpr_idx += m_depths[1]

        self.Bottleneck = TPDMStage(features[3], features[4],depth=m_depths[2],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[2]])
        dpr_idx += m_depths[2]

        self.freq1 = Freqenhance(features[0], prior='high')
        self.freq2 = Freqenhance(features[1], prior='high')
        self.freq3 = Freqenhance(features[2], prior='low')
        self.freq4 = Freqenhance(features[3], prior='low')

        # Downsample the number of channels remains the same, size is halved
        self.down1 = DownsampleBlock(features[0], features[0])
        self.down2 = DownsampleBlock(features[1], features[1])
        self.down3 = DownsampleBlock(features[2], features[2])
        self.down4 = DownsampleBlock(features[3], features[3])


        self.up1 = UpsampleBlock(features[4], features[3], scale_factor=scale_factor)
        self.up2 = UpsampleBlock(features[3], features[2], scale_factor=scale_factor)
        self.up3 = UpsampleBlock(features[2], features[1], scale_factor=scale_factor)
        self.up4 = UpsampleBlock(features[1], features[0], scale_factor=scale_factor)

        # decoder
        self.de1 = TPDMStage(features[4], features[3],depth=m_depths[1],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[1]])
        dpr_idx += m_depths[1]
        self.de2 = TPDMStage(features[3], features[2],depth=m_depths[0],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[0]])
        dpr_idx += m_depths[0]
        self.de3 = DSCResBlock(features[2], features[1],
                               act=act,
                               norm=norm,
                               groups=groups,
                               order=order,
                               drop_p=drop_p,
                               drop_path=dpr[dpr_idx])
        dpr_idx += 1
        self.de4 = DSCResBlock(features[1], features[0],
                               act=act,
                               norm=norm,
                               groups=groups,
                               order=order,
                               drop_p=drop_p,
                               drop_path=dpr[dpr_idx])

        self.out = nn.Conv3d(features[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        # Encoder
        x1 = self.en1(x)
        x1 = self.freq1(x1)

        x2 = self.down1(x1)
        x2 = self.en2(x2)
        x2 = self.freq2(x2)

        x3 = self.down2(x2)
        x3 = self.en3(x3)
        x3 = self.freq3(x3)

        x4 = self.down3(x3)
        x4 = self.en4(x4)
        x4 = self.freq4(x4)

        x5 = self.down4(x4)
        x5 = self.Bottleneck(x5)

        # Decoder
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.de1(x)

        x = self.up2(x)

        x = torch.cat([x, x3], dim=1)
        x = self.de2(x)

        x = self.up3(x)

        x = torch.cat([x, x2], dim=1)
        x = self.de3(x)

        x = self.up4(x)

        x = torch.cat([x, x1], dim=1)
        x = self.de4(x)  #

        out = self.out(x)

        return out


class AFMNet(nn.Module):
    def __init__(self, in_channels,
                 num_classes,
                 m_depths=(2,2,1),
                 features=(16,32,64,128,256),
                 act="PRELU",
                 norm="GROUP",
                 groups=4,
                 order='POST',
                 drop_path_rate=0.1,
                 drop_p=0.0,
                 scale_factor=2,
                 mlp_ratio=2):
        super(AFMNet, self).__init__()

        depths = [1, 1, m_depths[0], m_depths[1], m_depths[2], m_depths[1], m_depths[0], 1, 1]
        total_depth = sum(depths)
        half_len = total_depth // 2 + 1
        dpr_half1 = torch.linspace(0, drop_path_rate, half_len)
        dpr_half2 = torch.linspace(drop_path_rate, 0, half_len)[1:]
        dpr = [x.item() for x in torch.cat([dpr_half1, dpr_half2])]

        self.en1 = DSCResBlock(in_channels, features[0],
                               act=act,
                               norm=norm,
                               groups=groups,
                               order=order,
                               drop_p=drop_p,
                               drop_path=dpr[0])
        self.en2 = DSCResBlock(features[0], features[1],
                               act=act,
                               norm=norm,
                               groups=groups,
                               order=order,
                               drop_p=drop_p,
                               drop_path=dpr[1])
        dpr_idx = 2
        self.en3 = TPDMStage(features[1], features[2],depth=m_depths[0],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[0]])
        dpr_idx += m_depths[0]
        self.en4 = TPDMStage(features[2], features[3],depth=m_depths[1],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[1]])
        dpr_idx += m_depths[1]

        self.Bottleneck = TPDMStage(features[3], features[4],depth=m_depths[2],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[2]])
        dpr_idx += m_depths[2]

        self.freq1 = Freqenhance(features[0], prior='high')
        self.freq2 = Freqenhance(features[1], prior='high')
        self.freq3 = Freqenhance(features[2], prior='low')
        self.freq4 = Freqenhance(features[3], prior='low')

        self.down1 = DownsampleBlock(features[0], features[0])
        self.down2 = DownsampleBlock(features[1], features[1])
        self.down3 = DownsampleBlock(features[2], features[2])
        self.down4 = DownsampleBlock(features[3], features[3])

        self.up1 = UpsampleBlock(features[4], features[3], scale_factor=scale_factor)
        self.up2 = UpsampleBlock(features[3], features[2], scale_factor=scale_factor)
        self.up3 = UpsampleBlock(features[2], features[1], scale_factor=scale_factor)
        self.up4 = UpsampleBlock(features[1], features[0], scale_factor=scale_factor)

        self.att_g1 = AttentionGate(F_g=features[3], F_l=features[3], F_int=features[2])
        self.att_g2 = AttentionGate(F_g=features[2], F_l=features[2], F_int=features[1])
        self.att_g3 = AttentionGate(F_g=features[1], F_l=features[1], F_int=features[0])
        self.att_g4 = AttentionGate(F_g=features[0], F_l=features[0], F_int=features[0]//2)

        # decoder
        self.de1 = TPDMStage(features[4], features[3],depth=m_depths[1],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[1]])
        dpr_idx += m_depths[1]
        self.de2 = TPDMStage(features[3], features[2],depth=m_depths[0],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[0]])
        dpr_idx += m_depths[0]
        self.de3 = DSCResBlock(features[2], features[1],
                               act=act,
                               norm=norm,
                               groups=groups,
                               order=order,
                               drop_p=drop_p,
                               drop_path=dpr[dpr_idx])
        dpr_idx += 1
        self.de4 = DSCResBlock(features[1], features[0],
                               act=act,
                               norm=norm,
                               groups=groups,
                               order=order,
                               drop_p=drop_p,
                               drop_path=dpr[dpr_idx])

        self.out = nn.Conv3d(features[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        # Encoder
        x1 = self.en1(x)
        x1 = self.freq1(x1)

        x2 = self.down1(x1)
        x2 = self.en2(x2)
        x2 = self.freq2(x2)

        x3 = self.down2(x2)
        x3 = self.en3(x3)
        x3 = self.freq3(x3)

        x4 = self.down3(x3)
        x4 = self.en4(x4)
        x4 = self.freq4(x4)

        x5 = self.down4(x4)
        x5 = self.Bottleneck(x5)


        # Decoder
        x = self.up1(x5)
        skip1 = self.att_g1(x,x4)
        x = torch.cat([x, skip1], dim=1)
        x = self.de1(x)

        x = self.up2(x)
        skip2 = self.att_g2(x,x3)
        x = torch.cat([x, skip2], dim=1)
        x = self.de2(x)

        x = self.up3(x)
        skip3 = self.att_g3(x,x2)
        x = torch.cat([x, skip3], dim=1)
        x = self.de3(x)

        x = self.up4(x)
        skip4 = self.att_g4(x,x1)
        x = torch.cat([x, skip4], dim=1)
        x = self.de4(x)

        out = self.out(x)

        return out

class EFMNet(nn.Module):
    def __init__(self, in_channels,
                 num_classes,
                 m_depths=(2,2,1),
                 features=(16,32,64,128,256),
                 act="PRELU",
                 norm="GROUP",
                 groups=4,
                 order='POST',
                 drop_path_rate=0.1,
                 drop_p=0.0,
                 scale_factor=2,
                 mlp_ratio=2):
        super(EFMNet, self).__init__()

        depths = [1, 1, m_depths[0], m_depths[1], m_depths[2], m_depths[1], m_depths[0], 1, 1]
        total_depth = sum(depths)
        half_len = total_depth // 2 + 1
        dpr_half1 = torch.linspace(0, drop_path_rate, half_len)
        dpr_half2 = torch.linspace(drop_path_rate, 0, half_len)[1:]
        dpr = [x.item() for x in torch.cat([dpr_half1, dpr_half2])]

        self.en1 = DSCResBlock(in_channels, features[0],
                               act=act,
                               norm=norm,
                               groups=groups,
                               order=order,
                               drop_p=drop_p,
                               drop_path=dpr[0])
        self.en2 = DSCResBlock(features[0], features[1],
                               act=act,
                               norm=norm,
                               groups=groups,
                               order=order,
                               drop_p=drop_p,
                               drop_path=dpr[1])
        dpr_idx = 2
        self.en3 = TPDMStage(features[1], features[2],depth=m_depths[0],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[0]])
        dpr_idx += m_depths[0]
        self.en4 = TPDMStage(features[2], features[3],depth=m_depths[1],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[1]])
        dpr_idx += m_depths[1]

        self.Bottleneck = TPDMStage(features[3], features[4],depth=m_depths[2],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[2]])
        dpr_idx += m_depths[2]

        self.freq1 = Freqenhance(features[0], prior='high')
        self.freq2 = Freqenhance(features[1], prior='high')
        self.freq3 = Freqenhance(features[2], prior='low')
        self.freq4 = Freqenhance(features[3], prior='low')

        self.down1 = DownsampleBlock(features[0], features[0])
        self.down2 = DownsampleBlock(features[1], features[1])
        self.down3 = DownsampleBlock(features[2], features[2])
        self.down4 = DownsampleBlock(features[3], features[3])

        self.edgeguidedfusion = EdgeGuidedFusion(ch_list=[features[0], features[1], features[2], features[3]])

        self.up1 = UpsampleBlock(features[4], features[3], scale_factor=scale_factor)
        self.up2 = UpsampleBlock(features[3], features[2], scale_factor=scale_factor)
        self.up3 = UpsampleBlock(features[2], features[1], scale_factor=scale_factor)
        self.up4 = UpsampleBlock(features[1], features[0], scale_factor=scale_factor)

        # decoder
        self.de1 = TPDMStage(features[4], features[3],depth=m_depths[1],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[1]])
        dpr_idx += m_depths[1]
        self.de2 = TPDMStage(features[3], features[2],depth=m_depths[0],mlp_ratio=mlp_ratio,drop_paths=dpr[dpr_idx:dpr_idx+m_depths[0]])
        dpr_idx += m_depths[0]
        self.de3 = DSCResBlock(features[2], features[1],
                               act=act,
                               norm=norm,
                               groups=groups,
                               order=order,
                               drop_p=drop_p,
                               drop_path=dpr[dpr_idx])
        dpr_idx += 1
        self.de4 = DSCResBlock(features[1], features[0],
                               act=act,
                               norm=norm,
                               groups=groups,
                               order=order,
                               drop_p=drop_p,
                               drop_path=dpr[dpr_idx])

        self.out = nn.Conv3d(features[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        # Encoder
        x1 = self.en1(x)
        x1 = self.freq1(x1)

        x2 = self.down1(x1)
        x2 = self.en2(x2)
        x2 = self.freq2(x2)

        x3 = self.down2(x2)
        x3 = self.en3(x3)
        x3 = self.freq3(x3)

        x4 = self.down3(x3)
        x4 = self.en4(x4)
        x4 = self.freq4(x4)

        x5 = self.down4(x4)
        x5 = self.Bottleneck(x5)


        # skip connection
        y1,y2,y3,y4 = self.edgeguidedfusion(x1,x2,x3,x4)

        # Decoder
        x = self.up1(x5)
        x = torch.cat([x, y4], dim=1)
        x = self.de1(x)

        x = self.up2(x)
        x = torch.cat([x, y3], dim=1)
        x = self.de2(x)

        x = self.up3(x)
        x = torch.cat([x, y2], dim=1)
        x = self.de3(x)

        x = self.up4(x)
        x = torch.cat([x, y1], dim=1)
        x = self.de4(x)

        out = self.out(x)

        return out










