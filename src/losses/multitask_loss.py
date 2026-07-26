import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, num_tasks=4):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        device = losses[0].device
        total = torch.zeros(1, device=device)
        weights = []
        for i, loss in enumerate(losses):
            log_var = self.log_vars[i].to(device)
            precision = torch.exp(-log_var)
            total = total + precision * loss.to(device) + log_var
            weights.append(precision.item())
        return total.squeeze(), weights


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred_prob = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * (1 - p_t) ** self.gamma * ce_loss).mean()


class CityEyeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.uncertainty = UncertaintyWeightedLoss(num_tasks=4)
        self.focal = FocalLoss()

    def forward(self, outputs, targets):
        task_losses = []
        loss_dict = {}
        if "density" in outputs:
            loss = F.cross_entropy(outputs["density"], targets["congestion"].long())
            task_losses.append(loss)
            loss_dict["density"] = loss.item()
        if "violation" in outputs:
            loss = F.binary_cross_entropy_with_logits(outputs["violation"]["logits"], targets["violations"].float())
            task_losses.append(loss)
            loss_dict["violation"] = loss.item()
        if "anomaly" in outputs:
            loss = F.binary_cross_entropy_with_logits(outputs["anomaly"]["logits"], targets["anomalies"].float())
            task_losses.append(loss)
            loss_dict["anomaly"] = loss.item()
        if task_losses:
            total_loss, weights = self.uncertainty(task_losses)
            loss_dict["weights"] = weights
            loss_dict["total"] = total_loss.item()
        else:
            total_loss = torch.zeros(1, requires_grad=True)
        return total_loss, loss_dict
