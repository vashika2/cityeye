import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class FPN(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
            for in_ch in in_channels
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            for _ in in_channels
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(out_channels) for _ in in_channels
        ])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features):
        c3, c4, c5 = features
        laterals = [l(f) for l, f in zip(self.lateral_convs, [c3, c4, c5])]
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] += F.interpolate(laterals[i+1], size=laterals[i].shape[-2:], mode="nearest")
        return [F.relu(bn(conv(lat))) for conv, bn, lat in zip(self.output_convs, self.bns, laterals)]


def get_fpn(fpn_type: str, in_channels: List[int], out_channels: int = 256) -> nn.Module:
    if fpn_type in ("fpn", "bifpn"):
        return FPN(in_channels, out_channels)
    else:
        raise ValueError(f"Unknown FPN type: {fpn_type}")
