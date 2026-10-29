import torch
import torch.nn as nn

_norm_dict = {'BATCH': nn.BatchNorm3d, 'INSTANCE': nn.InstanceNorm3d, 'GROUP': nn.GroupNorm}
_act_dict = {'RELU': nn.ReLU, 'LEAKY_RELU': nn.LeakyReLU, 'PRELU': nn.PReLU, 'GELU': nn.GELU}

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3,stride=1,padding=1):
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv3d(in_ch, in_ch,kernel_size=kernel_size,stride=stride,padding=padding,groups=in_ch,bias=False)
        # 逐点卷积
        self.pointwise = nn.Conv3d(in_ch, out_ch, kernel_size=1,stride=1,padding=0, bias=False)
        init_weights(self)

    def forward(self,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DSCResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1,
                 act='PRELU', norm='GROUP', groups=4, order='POST', drop_p=0.0):
        """
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
            g1 = safe_group(ch_1,groups)
            g2 = safe_group(ch_2,groups)
            self.norm1 = _norm_dict[norm](num_groups=g1, num_channels=ch_1)
            self.norm2 = _norm_dict[norm](num_groups=g2, num_channels=ch_2)
        else:
            self.norm1 = _norm_dict[norm](ch_1)
            self.norm2 = _norm_dict[norm](ch_2)

        self.conv1 = DSConv(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.conv2 = DSConv(out_ch, out_ch, kernel_size, stride=1, padding=padding)

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

        self.dropout = nn.Dropout3d(p=drop_p) if drop_p > 0 else nn.Identity()
        init_weights(self)

    def forward(self,x):
        identity = self.shortcut(x)
        if self.order == 'POST':
            out = self.act1(self.norm1(self.conv1(x)))
            out = self.dropout(out)
            out = self.norm2(self.conv2(out))
            out += identity
            out = self.act2(out)
        else:
            out = self.conv1(self.act1(self.norm1(x)))
            out = self.dropout(out)
            out = self.conv2(self.act2(self.norm2(out)))
            out += identity
        return out


class DownsampleBlock(nn.Module):
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
    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 4, 128, 128, 128).to(device)

    dsconv = DSConv(in_ch=4, out_ch=3).to(device)
    dscresblock = DSCResBlock(in_ch=4, out_ch=16,act='PRELU',order='PRE',norm='GROUP').to(device)
    Downsample = DownsampleBlock(in_ch=16, out_ch=16,act='PRELU',norm='GROUP').to(device)
    Upsample = UpsampleBlock(in_ch=16, out_ch=32,act='PRELU',norm='GROUP').to(device)

    y = dscresblock(x)
    z = Downsample(y)
    v = Upsample(z)
    print(y.shape)
    print(z.shape)
    print(v.shape)