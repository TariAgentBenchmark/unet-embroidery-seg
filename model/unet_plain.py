import torch
import torch.nn as nn


class ECABlock(nn.Module):
    """Efficient Channel Attention."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for the bottleneck feature map."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        dilations = (1, 6, 12, 18)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1 if dilation == 1 else 3,
                        padding=0 if dilation == 1 else dilation,
                        dilation=1 if dilation == 1 else dilation,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for dilation in dilations
            ]
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        merged_channels = out_channels * (len(dilations) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(merged_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [branch(x) for branch in self.branches]
        pooled = self.image_pool(x)
        pooled = nn.functional.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
        outputs.append(pooled)
        return self.project(torch.cat(outputs, dim=1))


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_eca: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if use_eca:
            layers.append(ECABlock(out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_eca: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, use_eca=use_eca),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_eca: bool = False, use_sa: bool = False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.sa = SpatialAttention() if use_sa else nn.Identity()
        self.conv = DoubleConv(in_channels, out_channels, use_eca=use_eca)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            dh = skip.size(-2) - x.size(-2)
            dw = skip.size(-1) - x.size(-1)
            x = nn.functional.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        skip = self.sa(skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetPlain(nn.Module):
    """
    标准 U-Net 基线，可按需启用 ASPP / ECA / SA 用于消融实验。
    输出通道 = num_classes（含背景）。
    """

    def __init__(
        self,
        num_classes: int = 2,
        base_channels: int = 64,
        use_aspp: bool = False,
        use_eca: bool = False,
        use_sa: bool = False,
    ):
        super().__init__()
        self.inc = DoubleConv(3, base_channels, use_eca=use_eca)
        self.down1 = Down(base_channels, base_channels * 2, use_eca=use_eca)
        self.down2 = Down(base_channels * 2, base_channels * 4, use_eca=use_eca)
        self.down3 = Down(base_channels * 4, base_channels * 8, use_eca=use_eca)
        self.down4 = Down(base_channels * 8, base_channels * 16, use_eca=use_eca)
        self.aspp = ASPP(base_channels * 16, base_channels * 16) if use_aspp else nn.Identity()

        self.up1 = Up(base_channels * 16 + base_channels * 8, base_channels * 8, use_eca=use_eca, use_sa=use_sa)
        self.up2 = Up(base_channels * 8 + base_channels * 4, base_channels * 4, use_eca=use_eca, use_sa=use_sa)
        self.up3 = Up(base_channels * 4 + base_channels * 2, base_channels * 2, use_eca=use_eca, use_sa=use_sa)
        self.up4 = Up(base_channels * 2 + base_channels, base_channels, use_eca=use_eca, use_sa=use_sa)

        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.aspp(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
