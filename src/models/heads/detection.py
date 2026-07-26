import torch
import torch.nn as nn
from typing import List


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))
    def forward(self, x):
        return x * self.scale


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_layers=4):
        super().__init__()
        self.num_classes = num_classes
        cls_tower, reg_tower = [], []
        for _ in range(num_layers):
            cls_tower.extend([nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False), nn.GroupNorm(32, in_channels), nn.ReLU(inplace=True)])
            reg_tower.extend([nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False), nn.GroupNorm(32, in_channels), nn.ReLU(inplace=True)])
        self.cls_tower = nn.Sequential(*cls_tower)
        self.reg_tower = nn.Sequential(*reg_tower)
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(3)])
        prior_prob = 0.01
        nn.init.constant_(self.cls_logits.bias, -torch.log(torch.tensor((1-prior_prob)/prior_prob)).item())

    def forward(self, features: List[torch.Tensor]):
        cls_outputs, reg_outputs, ctr_outputs = [], [], []
        for i, feature in enumerate(features):
            cls_feat = self.cls_tower(feature)
            reg_feat = self.reg_tower(feature)
            cls_outputs.append(self.cls_logits(cls_feat))
            reg_outputs.append(self.scales[i](self.bbox_pred(reg_feat)).exp())
            ctr_outputs.append(self.centerness(reg_feat))
        return cls_outputs, reg_outputs, ctr_outputs
