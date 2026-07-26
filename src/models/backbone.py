import torch
import torch.nn as nn
import torchvision.models as tv
from typing import List, Tuple


class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_bn: bool = True):
        super().__init__()
        weights = tv.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = tv.resnet50(weights=weights)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.out_channels = [512, 1024, 2048]
        if freeze_bn:
            self._freeze_bn()

    def _freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5

    def freeze_stages(self, num_stages: int = 1):
        stages = [self.stem, self.layer1, self.layer2, self.layer3]
        for i, stage in enumerate(stages[:num_stages]):
            for param in stage.parameters():
                param.requires_grad = False


def get_backbone(name: str, pretrained: bool = True) -> nn.Module:
    if name == "resnet50":
        return ResNet50Backbone(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone: {name}")
