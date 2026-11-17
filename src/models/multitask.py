import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from omegaconf import DictConfig

from src.models.backbone import get_backbone
from src.models.fpn import get_fpn
from src.models.heads.detection import DetectionHead
from src.models.heads.density import DensityHead
from src.models.heads.violation import ViolationHead
from src.models.heads.anomaly import AnomalyHead


class CityEyeModel(nn.Module):
    """
    Full CityEye multi-task model.

    Architecture:
        Input image (B, 3, 640, 640)
            ↓
        Backbone (ResNet-50)
            → C3 (B, 512,  80, 80)
            → C4 (B, 1024, 40, 40)
            → C5 (B, 2048, 20, 20)
            ↓
        FPN
            → P3 (B, 256, 80, 80)
            → P4 (B, 256, 40, 40)
            → P5 (B, 256, 20, 20)
            ↓
        ┌──────────┬──────────┬──────────┬──────────┐
        ↓          ↓          ↓          ↓
    Detection  Density   Violation  Anomaly
      Head      Head       Head       Head
        ↓          ↓          ↓          ↓
    cls/reg/  congestion  violation  anomaly
    centerness  level     flags      flags+heatmap

    All four heads share the same backbone+FPN features —
    the backbone runs only once per image regardless of how many
    tasks are enabled.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # Backbone
        self.backbone = get_backbone(
            name=cfg.model.backbone,
            pretrained=cfg.model.pretrained,
        )

        # FPN
        self.fpn = get_fpn(
            fpn_type=cfg.model.get("fpn_type", "fpn"),
            in_channels=self.backbone.out_channels,
            out_channels=cfg.model.fpn_channels,
        )

        fpn_ch = cfg.model.fpn_channels

        # Task heads — enabled/disabled via config
        self.heads = nn.ModuleDict()

        if cfg.tasks.detection.enabled:
            self.heads["detection"] = DetectionHead(
                in_channels=fpn_ch,
                num_classes=len(cfg.tasks.detection.classes),
            )

        if cfg.tasks.congestion.enabled:
            self.heads["density"] = DensityHead(
                in_channels=fpn_ch,
                num_classes=4,
            )

        if cfg.tasks.violation.enabled:
            self.heads["violation"] = ViolationHead(
                in_channels=fpn_ch,
                num_violation_types=len(cfg.tasks.violation.types),
            )

        if cfg.tasks.anomaly.enabled:
            self.heads["anomaly"] = AnomalyHead(
                in_channels=fpn_ch,
                num_anomaly_types=len(cfg.tasks.anomaly.types),
            )

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, any]:
        """
        Args:
            x: Input tensor (B, 3, H, W)
        Returns:
            Dict of outputs per enabled task head
        """
        # Shared feature extraction — runs once
        backbone_features = self.backbone(x)
        fpn_features = self.fpn(backbone_features)

        outputs = {}

        if "detection" in self.heads:
            cls_out, reg_out, ctr_out = self.heads["detection"](fpn_features)
            outputs["detection"] = {
                "cls": cls_out,
                "reg": reg_out,
                "centerness": ctr_out,
            }

        if "density" in self.heads:
            outputs["density"] = self.heads["density"](fpn_features)

        if "violation" in self.heads:
            viol_logits, viol_attention = self.heads["violation"](fpn_features)
            outputs["violation"] = {
                "logits": viol_logits,
                "attention": viol_attention,
            }

        if "anomaly" in self.heads:
            anom_logits, anom_heatmaps = self.heads["anomaly"](fpn_features)
            outputs["anomaly"] = {
                "logits": anom_logits,
                "heatmaps": anom_heatmaps,
            }

        return outputs

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor,
        detection_threshold: float = 0.4,
        violation_threshold: float = 0.5,
        anomaly_threshold: float = 0.4,
    ) -> Dict:
        """
        Inference-time prediction — returns human-readable results.

        Args:
            x: Input tensor (B, 3, H, W)
            detection_threshold: Confidence threshold for detections
            violation_threshold: Confidence threshold for violations
            anomaly_threshold: Confidence threshold for anomalies
        Returns:
            Dict with structured predictions per task
        """
        self.eval()
        outputs = self.forward(x)
        results = {}

        # Density / congestion
        if "density" in outputs:
            probs = torch.softmax(outputs["density"], dim=1)
            pred_class = probs.argmax(dim=1)
            level_names = ["free_flow", "moderate", "dense", "gridlock"]
            results["congestion"] = {
                "level": [level_names[c.item()] for c in pred_class],
                "confidence": probs.max(dim=1).values.tolist(),
            }

        # Violations
        if "violation" in outputs:
            results["violations"] = self.heads["violation"].predict(
                [outputs["violation"]["logits"].unsqueeze(-1).unsqueeze(-1)],
                threshold=violation_threshold,
            )

        # Anomalies
        if "anomaly" in outputs:
            results["anomalies"] = self.heads["anomaly"].predict(
                list(self.fpn(self.backbone(x))),
                threshold=anomaly_threshold,
            )

        return results

    def freeze_backbone(self, num_stages: int = 2):
        """
        Freeze backbone stages during early training.
        Prevents destroying pretrained ImageNet features.
        """
        if hasattr(self.backbone, "freeze_stages"):
            self.backbone.freeze_stages(num_stages)
            print(f"Froze first {num_stages} backbone stages")

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Unfroze all backbone parameters")

    def get_parameter_groups(self, base_lr: float) -> List[Dict]:
        """
        Returns parameter groups with different learning rates.
        Backbone gets lower LR (pretrained weights), heads get higher LR.
        This is standard practice in transfer learning.
        """
        return [
            {
                "params": self.backbone.parameters(),
                "lr": base_lr * 0.1,
                "name": "backbone",
            },
            {
                "params": self.fpn.parameters(),
                "lr": base_lr * 0.5,
                "name": "fpn",
            },
            {
                "params": self.heads.parameters(),
                "lr": base_lr,
                "name": "heads",
            },
        ]

    def count_parameters(self) -> Dict[str, int]:
        """Returns parameter count per component — useful for debugging."""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        counts = {
            "backbone": count(self.backbone),
            "fpn": count(self.fpn),
        }
        for name, head in self.heads.items():
            counts[name + "_head"] = count(head)
        counts["total"] = sum(counts.values())
        return counts


def build_model(cfg: DictConfig) -> CityEyeModel:
    """Factory function — builds and returns the full CityEye model."""
    model = CityEyeModel(cfg)

    total_params = model.count_parameters()
    print(f"CityEye model built:")
    for name, count in total_params.items():
        print(f"  {name}: {count:,} parameters")

    return model