import torch
import torch.nn as nn

from model.unet_plain import DoubleConv


class AttentionGate(nn.Module):
    """
    Attention Gate for U-Net skip connections.
    Reference: Attention U-Net (Oktay et al.)
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.theta = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.phi = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if skip.size(-1) != gate.size(-1) or skip.size(-2) != gate.size(-2):
            gate = nn.functional.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        f = self.relu(self.theta(skip) + self.phi(gate))
        alpha = self.psi(f)
        return skip * alpha


class UpAttn(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.attn = AttentionGate(
            gate_channels=in_channels,
            skip_channels=skip_channels,
            inter_channels=max(out_channels // 2, 16),
        )
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = self.attn(skip, x)
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net（基于标准 U-Net，skip connection 加 attention gate）
    """

    def __init__(self, num_classes: int = 2, base_channels: int = 64):
        super().__init__()
        self.inc = DoubleConv(3, base_channels)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels, base_channels * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 2, base_channels * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 4, base_channels * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 8, base_channels * 16))

        self.up1 = UpAttn(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = UpAttn(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = UpAttn(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = UpAttn(base_channels * 2, base_channels, base_channels)

        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

