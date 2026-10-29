import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="mamba_ssm")

import torch
import torch.nn as nn
import torch.nn.functional as F
import ptwt

from mamba_ssm import Mamba
from torch.amp import autocast
from thop import profile, clever_format

_norm_dict = {'BATCH': nn.BatchNorm3d, 'INSTANCE': nn.InstanceNorm3d, 'GROUP': nn.GroupNorm}
_act_dict = {'RELU': nn.ReLU, 'LEAKY_RELU': nn.LeakyReLU, 'PRELU': nn.PReLU, 'GELU': nn.GELU}


class DSConv(nn.Module):
    """
    Depthwise Separable Convolution
    """
    def __init__(self, in_ch, out_ch, kernel_size=3,stride=1,padding=1):
        super(DSConv, self).__init__()

        # Depthwise Convolution
        self.depthwise = nn.Conv3d(in_ch, in_ch,kernel_size=kernel_size,stride=stride,padding=padding,groups=in_ch,bias=False)
        # Pointwise Convolution
        self.pointwise = nn.Conv3d(in_ch, out_ch, kernel_size=1,stride=1,padding=0, bias=False)
        init_weights(self)

    def forward(self,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DSCResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1,
                 act='PRELU', norm='GROUP', groups=4, order='POST', drop_p=0.0,drop_path=0.0):
        """
        Lightweight feature extraction module (only changes channels, input and output dimensions remain unchanged)
        :param order: Use pre-activation or post-activation
        :param drop_p: Dropout3d
        """
        super(DSCResBlock, self).__init__()

        self.order = order.upper()
        assert self.order in ['PRE','POST'],"The order parameter must be 'POST' or 'PRE'."

        # Select the number of channels targeted by Norm/Act according to the order
        if self.order == 'POST':
            ch_1 = out_ch
            ch_2 = out_ch
        else:
            ch_1 = in_ch
            ch_2 = out_ch

        self.act1 = _act_dict[act]() if act != 'PRELU' else _act_dict[act](num_parameters=ch_1)
        self.act2 = _act_dict[act]() if act != 'PRELU' else _act_dict[act](num_parameters=ch_2)

        if norm == 'GROUP':
            # Adaptive grouping
            g1 = safe_group(ch_1,groups)
            g2 = safe_group(ch_2,groups)
            self.norm1 = _norm_dict[norm](num_groups=g1, num_channels=ch_1)
            self.norm2 = _norm_dict[norm](num_groups=g2, num_channels=ch_2)
        else:
            self.norm1 = _norm_dict[norm](ch_1)
            self.norm2 = _norm_dict[norm](ch_2)

        self.conv1 = DSConv(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.conv2 = DSConv(out_ch, out_ch, kernel_size, stride=1, padding=padding)

        # Dimension Matching
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

        # Dropout (between conv1 and conv2)
        self.dropout = nn.Dropout3d(p=drop_p) if drop_p > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        init_weights(self)

    def forward(self,x):
        identity = self.shortcut(x)
        if self.order == 'POST':
            out = self.act1(self.norm1(self.conv1(x)))
            out = self.dropout(out)
            out = self.norm2(self.conv2(out))
            out = self.drop_path(out)
            out += identity
            out = self.act2(out)
        else:
            out = self.conv1(self.act1(self.norm1(x)))
            out = self.dropout(out)
            out = self.conv2(self.act2(self.norm2(out)))
            out = self.drop_path(out)
            out += identity
        return out


class DownsampleBlock(nn.Module):
    """
    Downsampling module: responsible for halving the size while keeping the number of channels unchanged
    """
    def __init__(self, in_ch, out_ch, stride=2,norm='GROUP', act='PRELU', groups=4):
        super(DownsampleBlock, self).__init__()
        self.down = DSConv(in_ch, out_ch, kernel_size=3, padding=1, stride=stride)
        if norm == 'GROUP':
            g = safe_group(out_ch,groups)
            self.norm = _norm_dict[norm](num_groups=g, num_channels=out_ch)
        else:
            self.norm = _norm_dict[norm](out_ch)

        self.act = _act_dict[act]() if act != 'PRELU' else _act_dict[act](num_parameters=out_ch)
        init_weights(self)

    def forward(self, x):
        return self.act(self.norm(self.down(x)))


class UpsampleBlock(nn.Module):
    """
    Upsampling module:Use trilinear interpolation followed by depthwise separable convolution
    """
    def __init__(self,in_ch,out_ch,scale_factor=2,norm='GROUP', act='PRELU', groups=4):
        super(UpsampleBlock,self).__init__()
        self.up = nn.Upsample(
            scale_factor=scale_factor,
            mode='trilinear',
            align_corners=False
        )
        self.conv = DSConv(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=1
        )
        if norm == 'GROUP':
            g = safe_group(out_ch,groups)
            self.norm = _norm_dict[norm](num_groups=g, num_channels=out_ch)
        else:
            self.norm = _norm_dict[norm](out_ch)

        self.act = _act_dict[act]() if act != 'PRELU' else _act_dict[act](num_parameters=out_ch)
        init_weights(self)

    def forward(self,x):
        return self.act(self.norm(self.conv(self.up(x))))

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class TPDMStage(nn.Module):
    def __init__(self,in_ch,out_ch,depth=2,mlp_ratio=2,drop_paths=None):
        super().__init__()
        self.proj = nn.Sequential(
            DSConv(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(
                8 if out_ch >= 32 else 4,
                out_ch
            ),
            nn.PReLU(out_ch)
        )
        if drop_paths is None:
            drop_paths = [0.0] * depth
        self.blocks = nn.Sequential(
            *[TPDMBlock(out_ch,mlp_ratio=mlp_ratio,drop_path=drop_paths[i])
                for i in range(depth)])

    def forward(self,x):
        x = self.proj(x)
        x = self.blocks(x)
        return x


class TPDMBlock(nn.Module):
    def __init__(self, dim,mlp_ratio=2, drop_path=0.0):
        super(TPDMBlock, self).__init__()

        self.MDS = MultiDirectionalscan(dim)
        self.mlp = MLP(dim,mlp_ratio=mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,x):
        # Mamba Residual
        x = x + self.drop_path(self.MDS(x))
        # MLP Residual
        x =  x + self.drop_path(self.mlp(x))
        return x


class MultiDirectionalscan(nn.Module):
    def __init__(self,
                 dim,
                 d_state=16,
                 d_conv=4,
                 expand=2):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.dir_embed = nn.Parameter(torch.randn(6,1,dim))  # Direction encoding for six directions
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

        self.fb_fusion = FBGateFusion(dim)
        self.fusion = Fusion(dim)

    @autocast("cuda",enabled=False)
    def forward(self,x):
        orig_dtype = x.dtype
        x = x.float()
        scan = t_plane_scan(x)    # [B,6,C,L]
        B, K, C, L = scan.shape   # [B,6,C,L]

        scan = scan.transpose(-1, -2)  # [B,6,L,C]
        scan = scan + self.dir_embed.unsqueeze(0)  # Add direction encoding, broadcast [1,6,1,C] to [B,6,L,C].
        x_flat = scan.reshape(B * K, L, C)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)   # [B*6,L,C]
        mamba_out = x_mamba.transpose(-1, -2).reshape(B, K, C, L)  # [B,6,C,L]
        out_tuple = t_plane_scan_re(x, mamba_out)  # Recover[B,C,H,W,D]
        cor = self.fb_fusion(out_tuple[0], out_tuple[1])
        sag = self.fb_fusion(out_tuple[2], out_tuple[3])
        ax = self.fb_fusion(out_tuple[4], out_tuple[5])
        out = self.fusion(cor, sag, ax)
        return out.to(orig_dtype)


class MLP(nn.Module):
    """
    MLP module
    """
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hidden_dim = dim * mlp_ratio
        self.up = nn.Conv3d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.down = nn.Conv3d(hidden_dim, dim, 1)
        self.norm = nn.GroupNorm(8 if dim >= 32 else 4,dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.up(x)
        x = self.act(x)
        x = self.down(x)
        return x


class FBGateFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(dim * 2, dim, 1),
            nn.Sigmoid()
        )

    def forward(self,fwd,bwd):
        plane = torch.cat([fwd,bwd],dim=1)
        g = self.gate(plane)

        out = g * fwd + (1 - g) * bwd
        return out


class Fusion(nn.Module):
    def __init__(self, dim,reduction=4):
        super().__init__()
        hidden = (dim * 3) // reduction
        self.fc = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim * 3)
        )

    def forward(self, cor, sag, ax):
        B, C, H, W, D = cor.shape

        feat = torch.cat([cor, sag, ax],dim=1)
        pool = feat.mean(dim=(2, 3, 4))
        weight = self.fc(pool)
        weight = weight.reshape(B, 3, C, 1, 1, 1)
        weight = torch.softmax(weight,dim=1)
        out = (
                weight[:, 0] * cor +
                weight[:, 1] * sag +
                weight[:, 2] * ax
        )

        return out


class Freqenhance(nn.Module):
    """
    3D Haar Frequency Attention Module
    input: [B, C, H, W, D]
    output: [B, C, H, W, D]
    """
    def __init__(self, channels, prior='balanced', wavelet='haar', reduction=4):
        super().__init__()
        self.wavelet = wavelet

        self.weight_gen = nn.Sequential(
            nn.Conv3d(channels * 8, channels // reduction, 1, bias=False),
            nn.GroupNorm(1, channels // reduction),
            nn.PReLU(),
            nn.Conv3d(channels // reduction, 2, 1, bias=False),
        )

        # 引入弱先验
        if prior == 'high':
            low_init = 1e-3
            high_init = 1e-2
        elif prior == 'low':
            low_init = 1e-2
            high_init = 1e-3
        else:
            low_init = 1e-3
            high_init = 1e-3
        self.alpha_low = nn.Parameter(
            torch.tensor(low_init)
        )
        self.alpha_high = nn.Parameter(
            torch.tensor(high_init)
        )

    def forward(self, x):
        B, C, H, W, D = x.shape
        original_dtype = x.dtype

        # 3D Haar Frequency Domain Decomposition
        x_fp32 = x.to(torch.float32)
        coeffs = ptwt.wavedec3(x_fp32, self.wavelet, level=1, mode='reflect')
        low, high = coeffs

        subbands = [low]
        key_order = ['aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']
        subbands += [high[k] for k in key_order]

        #  [B, 8C, H/2, W/2, D/2]
        cat_down = torch.cat(subbands, dim=1)
        # [B, 2, H/2, W/2, D/2]
        w_down = torch.sigmoid(self.weight_gen(cat_down))

        target = (H, W, D)
        w = F.interpolate(w_down, size=target, mode='trilinear', align_corners=False)
        w_low, w_high = w[:, 0:1], w[:, 1:2]

        low_freq = F.interpolate(low, size=target, mode='trilinear', align_corners=False)
        high_freq = x_fp32 - low_freq

        out = x_fp32 + self.alpha_low * (w_low * low_freq) + self.alpha_high * (w_high * high_freq)
        return out.to(original_dtype)

class EdgeGuidedFusion(nn.Module):
    """
    Edge Mask-Guided 3D Multi-Scale Fusion Module
    """
    def __init__(self, ch_list=[16, 32, 64, 128], eps=1e-4):
        super(EdgeGuidedFusion, self).__init__()
        c1, c2, c3, c4 = ch_list

        self.eps = eps

        self.align1 = nn.Conv3d(c1, c2, kernel_size=1, bias=False)
        self.align3 = nn.Conv3d(c3, c2, kernel_size=1, bias=False)
        self.align4 = nn.Conv3d(c4, c2, kernel_size=1, bias=False)

        # BiFPN Learnable Weights
        self.weights = nn.Parameter(torch.ones(4, dtype=torch.float32))
        self.relu = nn.ReLU()

        self.refine = nn.Sequential(
            DSConv(c2, c2),
            nn.GroupNorm(8, c2),
            nn.PReLU(c2),
        )

        self.mask_gen1 = nn.Sequential(
            DSConv(c1, 1),
            nn.Sigmoid()
        )
        self.mask_gen2 = nn.Sequential(
            DSConv(c2, 1),
            nn.Sigmoid()
        )

        # Gated Residual Coefficient
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.up_to_x1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.down_to_x3 = nn.Conv3d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False)
        self.down_to_x4 = nn.Conv3d(c2, c4, kernel_size=3, stride=2, padding=1, bias=False)
        self.down_to_x4_2 = nn.Conv3d(c4, c4, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self,x1, x2, x3, x4):

        target_size = x2.shape[2:]  # target size

        d1 = F.interpolate(self.align1(x1), size=target_size, mode='trilinear', align_corners=False)
        d2 = x2
        d3 = F.interpolate(self.align3(x3), size=target_size, mode='trilinear', align_corners=False)
        d4 = F.interpolate(self.align4(x4), size=target_size, mode='trilinear', align_corners=False)

        w = self.relu(self.weights)
        w = w / (torch.sum(w) + self.eps)
        fused = self.refine(w[0]*d1 + w[1]*d2 + w[2]*d3 + w[3]*d4) # [B, 32, 64³]


        y1_base = self.up_to_x1(fused)  # [B, 16, 128³]
        mask1 = self.mask_gen1(x1)  # [B, 1, 128, 128, 128]

        y1 = y1_base + self.gamma1 * (y1_base * mask1)
        y2_base = fused
        mask2 = self.mask_gen2(x2)

        y2 = y2_base + self.gamma2 * (y2_base * mask2)

        fused_down3 = self.down_to_x3(fused)  # [B, 64, 32³]
        y3 = x3 + fused_down3

        fused_down4 = self.down_to_x4(fused)
        fused_down4 = self.down_to_x4_2(fused_down4)
        y4 = x4 + fused_down4

        self.mask1 = mask1
        self.mask2 = mask2
        return y1, y2, y3, y4


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(F_int, F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(F_int, F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)  # 64
        x1 = self.W_x(x)  # 64

        # add -> ReLU -> Sigmoid
        psi = self.relu(g1 + x1)
        attention_mask = self.psi(psi)  #  [B, 1, 16, 16, 16]
        return x * attention_mask


class Concat(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super(Concat, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv3d(
                in_ch + skip_ch,
                out_ch,
                kernel_size=1,
                bias=False
            ),
            nn.GroupNorm(
                8 if out_ch >= 32 else 4,
                out_ch
            ),
            nn.PReLU(out_ch)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.fuse(x)
        return x


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

        # During the encoding stage, the number of channels increases while the dimension remains unchanged.
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

        self.freq1 = Freqenhance(features[0],prior='high')
        self.freq2 = Freqenhance(features[1],prior='high')
        self.freq3 = Freqenhance(features[2],prior='low')
        self.freq4 = Freqenhance(features[3],prior='low')

        # Downsample the number of channels remains the same, size is halved
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

def t_plane_scan(x :torch.Tensor):
    B, C, H, W, D = x.shape
    L = H * W * D

    x_plane1 = x.contiguous()
    scan_plane1_fwd = x_plane1.view(B,C,L)
    scan_plane1_bwd = scan_plane1_fwd.flip(dims=[-1])

    x_plane2 = x.permute(0, 1, 4, 2, 3).contiguous()
    scan_plane2_fwd = x_plane2.view(B ,C ,L)
    scan_plane2_bwd = scan_plane2_fwd.flip(dims=[-1])

    x_plane3 = x.permute(0, 1, 3, 4, 2).contiguous()
    scan_plane3_fwd = x_plane3.view(B ,C ,L)
    scan_plane3_bwd = scan_plane3_fwd.flip(dims=[-1])

    return torch.stack([
        scan_plane1_fwd, scan_plane1_bwd,
        scan_plane2_fwd, scan_plane2_bwd,
        scan_plane3_fwd, scan_plane3_bwd
    ], dim=1)


def t_plane_scan_re(x: torch.Tensor, y: torch.Tensor):
    H, W, D = x.shape[2:]
    B, _, C, L = y.shape
    y_plane1_fwd, y_plane1_bwd, y_plane2_fwd, y_plane2_bwd, y_plane3_fwd, y_plane3_bwd = y.unbind(dim=1)

    out_plane1_fwd = y_plane1_fwd.view(B, C, H, W, D)
    out_plane1_bwd = y_plane1_bwd.flip(dims=[-1]).view(B, C, H, W, D)

    out_plane2_fwd = y_plane2_fwd.view(B, C, D, H, W).permute(0, 1, 3, 4, 2)
    out_plane2_bwd = y_plane2_bwd.flip(dims=[-1]).view(B, C, D, H, W).permute(0, 1, 3, 4, 2)

    out_plane3_fwd = y_plane3_fwd.view(B, C, W, D, H).permute(0, 1, 4, 2, 3)
    out_plane3_bwd = y_plane3_bwd.flip(dims=[-1]).view(B, C, W, D, H).permute(0, 1, 4, 2, 3)
    out = (out_plane1_fwd,out_plane1_bwd,out_plane2_fwd,out_plane2_bwd,out_plane3_fwd,out_plane3_bwd)
    return out

def safe_group(num_channels, max_groups):
    g = min(max_groups,num_channels)
    while num_channels % g != 0:
        g -= 1
    return g


# Kaiming initialization
def init_weights(module, prelu_init=0.25):
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=prelu_init, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
            if getattr(m, 'weight', None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.PReLU):
            nn.init.constant_(m.weight, val=prelu_init)


if __name__ == '__main__':
    x = torch.randn(1, 4, 128, 128, 128).to("cuda")
    net = EFMNet(in_channels=4, num_classes=3).to("cuda")

    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(f"Params: {total_params / 1e6:.2f} M")
    print(f"Trainable Params: {trainable_params / 1e6:.2f} M")
    print(f"Model size (fp32): {total_params * 4 / (1024 ** 2):.2f} MB")



