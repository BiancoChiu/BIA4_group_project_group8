import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np
import os


class SimpleWingCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # /2

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # /4

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # /8
        )

        self.gap = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.fc(x)

class DeeperWingCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        # 4 个 stage，每个 stage 两个 Conv + BN + ReLU，然后 MaxPool
        self.stage1 = conv_block(in_channels, 32)
        self.pool1  = nn.MaxPool2d(2)   # 尺寸 /2

        self.stage2 = conv_block(32, 64)
        self.pool2  = nn.MaxPool2d(2)   # /4

        self.stage3 = conv_block(64, 128)
        self.pool3  = nn.MaxPool2d(2)   # /8

        self.stage4 = conv_block(128, 256)
        self.pool4  = nn.MaxPool2d(2)   # /16

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # -> (B, 256, 1, 1)

        # 两层全连接 + Dropout
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.pool1(x)

        x = self.stage2(x)
        x = self.pool2(x)

        x = self.stage3(x)
        x = self.pool3(x)

        x = self.stage4(x)
        x = self.pool4(x)

        x = self.gap(x)                  # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)        # (B, 256)
        logits = self.classifier(x)      # (B, num_classes)
        return logits

class DeeperWingCNN2(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.stage1 = conv_block(in_channels, 32)
        self.pool1  = nn.MaxPool2d(2)   # 尺寸 /2

        self.stage2 = conv_block(32, 64)
        self.pool2  = nn.MaxPool2d(2)   # /4

        self.stage3 = conv_block(64, 128)
        self.pool3  = nn.MaxPool2d(2)   # /8

        self.stage4 = conv_block(128, 256)
        self.pool4  = nn.MaxPool2d(2)   # /16

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # -> (B, 256, 1, 1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )


        self.emb_dim = 256

    def encode(self, x):
        """
        返回 CNN 分支的 embedding（在最后一层 Linear 之前截断）:
        x: [B, 1, H, W]
        out: [B, 256]
        """
        x = self.stage1(x)
        x = self.pool1(x)

        x = self.stage2(x)
        x = self.pool2(x)

        x = self.stage3(x)
        x = self.pool3(x)

        x = self.stage4(x)
        x = self.pool4(x)

        x = self.gap(x)                 # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)       # (B, 256)

        # classifier: [0]=Linear(256->256), [1]=ReLU, [2]=Dropout, [3]=Linear->num_classes
        x = self.classifier[0](x)       # Linear
        x = self.classifier[1](x)       # ReLU
        x = self.classifier[2](x)       # Dropout
        return x                        # (B, 256)

    def forward(self, x):

        x = self.stage1(x)
        x = self.pool1(x)

        x = self.stage2(x)
        x = self.pool2(x)

        x = self.stage3(x)
        x = self.pool3(x)

        x = self.stage4(x)
        x = self.pool4(x)

        x = self.gap(x)                  # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)        # (B, 256)
        logits = self.classifier(x)      # (B, num_classes)
        return logits
