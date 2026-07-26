import torch
import torch.nn as nn
from typing import List


class AnomalyHead(nn.Module):
    def __init__(self, in_channels, num_anomaly_types=3, dropout=0.3):
        super().__init__()
        self.scale_aggregators = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, 128, 1, bias=False), nn.GroupNorm(16, 128), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(4), nn.Flatten())
            for _ in range(3)])
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_channels, in_channels//4), nn.ReLU(inplace=True), nn.Linear(in_channels//4, in_channels), nn.Sigmoid())
        fused_dim = 128 * 4 * 4 * 3
        self.fusion = nn.Sequential(nn.Linear(fused_dim, 512), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(512, 128), nn.ReLU(inplace=True))
        self.anomaly_classifiers = nn.ModuleList([nn.Linear(128, 1) for _ in range(num_anomaly_types)])
        self.localization = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1, bias=False), nn.GroupNorm(8, 64), nn.ReLU(inplace=True), nn.Conv2d(64, num_anomaly_types, 1))

    def forward(self, features: List[torch.Tensor]):
        p3, p4, p5 = features
        ca_weights = self.channel_attention(p4).unsqueeze(-1).unsqueeze(-1)
        p4_attended = p4 * ca_weights
        scale_feats = [agg(feat) for agg, feat in zip(self.scale_aggregators, [p3, p4_attended, p5])]
        fused = self.fusion(torch.cat(scale_feats, dim=1))
        logits = torch.cat([c(fused) for c in self.anomaly_classifiers], dim=1)
        heatmaps = self.localization(p3)
        return logits, heatmaps

    def predict(self, features, threshold=0.4):
        anomaly_names = ["stopped_vehicle", "accident", "pedestrian_on_road"]
        logits, _ = self.forward(features)
        probs = torch.sigmoid(logits)
        return [[{"type": name, "confidence": round(probs[b, i].item(), 3)} for i, name in enumerate(anomaly_names) if probs[b, i].item() > threshold] for b in range(probs.shape[0])]
