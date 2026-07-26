import torch
import torch.nn as nn
from typing import List


class ViolationHead(nn.Module):
    def __init__(self, in_channels, num_violation_types=3, dropout=0.3):
        super().__init__()
        self.feature_tower = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False), nn.GroupNorm(32, in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False), nn.GroupNorm(32, in_channels), nn.ReLU(inplace=True))
        self.spatial_attention = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared_classifier = nn.Sequential(nn.Flatten(), nn.Linear(in_channels, 256), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(256, num_violation_types))

    def forward(self, features: List[torch.Tensor]):
        x = features[0]
        x = self.feature_tower(x)
        attention = self.spatial_attention(x)
        x = x * attention
        x = self.pool(x)
        return self.shared_classifier(x), attention

    def predict(self, features, threshold=0.5):
        violation_names = ["no_helmet", "signal_jump", "wrong_way"]
        logits, _ = self.forward(features)
        probs = torch.sigmoid(logits)
        return [[{"type": name, "confidence": round(probs[b, i].item(), 3)} for i, name in enumerate(violation_names) if probs[b, i].item() > threshold] for b in range(probs.shape[0])]
