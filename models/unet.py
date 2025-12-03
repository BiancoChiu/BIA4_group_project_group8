import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset

class DoubleConv(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """maxpool + double conv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upsample + concat(skip) + double conv"""

    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        # x1: decoder feature, x2: encoder skip
        x1 = self.up(x1)

        # padding 对齐（理论上你现在都是统一尺寸，这步主要是避免整除误差）
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=5, bilinear=True):
        """
        num_classes: 包含 padding label 的总 channel 数
        （例如真实 4 类 + padding=4 -> 可以用 5；如果是二分类，可以 num_classes=2）
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # 你可以改小一些通道数以节省显存
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down5 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64 // factor, bilinear)
        self.up5 = Up(64, 32, bilinear)

        self.outc = OutConv(32, num_classes)

    def forward(self, x):
        x1 = self.inc(x)      # [B,32,H,W]
        x2 = self.down1(x1)   # [B,64,H/2,W/2]
        x3 = self.down2(x2)   # ...
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x = self.up1(x6, x5)
        x = self.up2(x,  x4)
        x = self.up3(x,  x3)
        x = self.up4(x,  x2)
        x = self.up5(x,  x1)

        logits = self.outc(x)   # [B,num_classes,H,W]
        return logits