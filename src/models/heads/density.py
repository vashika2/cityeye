import torch
import torch.nn as nn
from typing import List


class DensityHead(nn.Module):
    def __init__(self, in_channels, num_classes=4, dropout=0.3):
        super().__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, in_channels//4, 1), nn.ReLU(inplace=True), nn.Conv2d(in_channels//4, 1, 1), nn.Sigmoid())
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(in_channels, 512), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(dropout/2), nn.Linear(256, num_classes))

    def forward(self, features: List[torch.Tensor]):
        x = features[-1]
        x = x * self.attention(x)
        x = self.gap(x)
        return self.classifier(x)
