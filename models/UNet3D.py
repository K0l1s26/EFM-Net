"""3D U-Net architecture adapted from Cicek et al. (MICCAI 2016)."""

import torch
import torch.nn as nn


class _ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )


class Paper3DUNet(nn.Module):
    """Paper-faithful 3D U-Net with a same-size BraTS output.

    The encoder/decoder widths, two 3x3x3 convolutions per resolution,
    2x2x2 max pooling, transposed convolutions, and concatenation skips follow
    the original 3D U-Net. Padding is changed from ``valid`` to ``same`` so a
    BraTS patch keeps its spatial size. The model returns raw logits.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 3) -> None:
        super().__init__()

        self.enc1 = nn.Sequential(
            _ConvBNReLU(in_channels, 32),
            _ConvBNReLU(32, 64),
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(
            _ConvBNReLU(64, 64),
            _ConvBNReLU(64, 128),
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = nn.Sequential(
            _ConvBNReLU(128, 128),
            _ConvBNReLU(128, 256),
        )
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            _ConvBNReLU(256, 256),
            _ConvBNReLU(256, 512),
        )

        self.up3 = nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            _ConvBNReLU(512 + 256, 256),
            _ConvBNReLU(256, 256),
        )

        self.up2 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            _ConvBNReLU(256 + 128, 128),
            _ConvBNReLU(128, 128),
        )

        self.up1 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            _ConvBNReLU(128 + 64, 64),
            _ConvBNReLU(64, 64),
        )

        self.out_conv = nn.Conv3d(64, out_channels, kernel_size=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm3d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"3D U-Net expects [B, C, D, H, W], got {tuple(x.shape)}")
        if any(size % 8 != 0 for size in x.shape[2:]):
            raise ValueError(
                "Each spatial input dimension must be divisible by 8; "
                f"got {tuple(x.shape[2:])}."
            )

        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool1(skip1))
        skip3 = self.enc3(self.pool2(skip2))
        x = self.bottleneck(self.pool3(skip3))

        x = self.dec3(torch.cat((self.up3(x), skip3), dim=1))
        x = self.dec2(torch.cat((self.up2(x), skip2), dim=1))
        x = self.dec1(torch.cat((self.up1(x), skip1), dim=1))
        return self.out_conv(x)

