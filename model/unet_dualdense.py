import torch
import torch.nn as nn


class _DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        cur_channels = in_channels
        for _ in range(num_layers):
            self.layers.append(_DenseLayer(cur_channels, growth_rate))
            cur_channels += growth_rate
        self.out_channels = cur_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        return torch.cat(features, dim=1)


class DenseConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, growth_rate: int = 32, num_layers: int = 3):
        super().__init__()
        self.dense = DenseBlock(in_channels, growth_rate=growth_rate, num_layers=num_layers)
        self.trans = nn.Sequential(
            nn.Conv2d(self.dense.out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trans(self.dense(x))


class UpDense(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, growth_rate: int = 32, num_layers: int = 3):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DenseConvBlock(in_channels + skip_channels, out_channels, growth_rate=growth_rate, num_layers=num_layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class DualDenseUNet(nn.Module):
    """
    Dense Block 版 U-Net（用 DenseConvBlock 替代 DoubleConv）。
    这里的“DualDense”指每个 stage 采用 dense-style 特征聚合，编码/解码两路均为 dense blocks。
    """

    def __init__(
        self,
        num_classes: int = 2,
        base_channels: int = 64,
        growth_rate: int = 32,
        num_layers: int = 3,
    ):
        super().__init__()

        self.inc = DenseConvBlock(3, base_channels, growth_rate=growth_rate, num_layers=num_layers)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DenseConvBlock(base_channels, base_channels * 2, growth_rate, num_layers))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DenseConvBlock(base_channels * 2, base_channels * 4, growth_rate, num_layers))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DenseConvBlock(base_channels * 4, base_channels * 8, growth_rate, num_layers))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DenseConvBlock(base_channels * 8, base_channels * 16, growth_rate, num_layers))

        self.up1 = UpDense(base_channels * 16, base_channels * 8, base_channels * 8, growth_rate, num_layers)
        self.up2 = UpDense(base_channels * 8, base_channels * 4, base_channels * 4, growth_rate, num_layers)
        self.up3 = UpDense(base_channels * 4, base_channels * 2, base_channels * 2, growth_rate, num_layers)
        self.up4 = UpDense(base_channels * 2, base_channels, base_channels, growth_rate, num_layers)

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

