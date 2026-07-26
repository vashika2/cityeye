import torch
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from src.models.cityeye import CityEyeModel, build_model
from src.losses.multitask_loss import CityEyeLoss


class CityEyeTrainer(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(dict(cfg))
        self.model = build_model(cfg)
        self.loss_fn = CityEyeLoss()
        self.model.freeze_backbone(num_stages=2)
        self.freeze_epochs = cfg.training.freeze_backbone_epochs

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch["image"])
        targets = {"congestion": batch["congestion"], "violations": batch["violations"], "anomalies": batch["anomalies"]}
        total_loss, loss_dict = self.loss_fn(outputs, targets)
        self.log("train/total_loss", total_loss, prog_bar=True)
        self.log("train/density_loss", loss_dict.get("density", 0))
        self.log("train/violation_loss", loss_dict.get("violation", 0))
        self.log("train/anomaly_loss", loss_dict.get("anomaly", 0))
        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch["image"])
        targets = {"congestion": batch["congestion"], "violations": batch["violations"], "anomalies": batch["anomalies"]}
        total_loss, loss_dict = self.loss_fn(outputs, targets)
        self.log("val/total_loss", total_loss, prog_bar=True)
        if "density" in outputs:
            preds = outputs["density"].argmax(dim=1)
            acc = (preds == batch["congestion"]).float().mean()
            self.log("val/congestion_accuracy", acc, prog_bar=True)
        return total_loss

    def on_train_epoch_start(self):
        if self.current_epoch == self.freeze_epochs:
            self.model.unfreeze_backbone()
            print(f"\nEpoch {self.current_epoch}: Unfreezing backbone")

    def configure_optimizers(self):
        param_groups = self.model.get_parameter_groups(self.cfg.training.lr)
        optimizer = AdamW(param_groups, weight_decay=self.cfg.training.weight_decay)
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=3)
        cosine = CosineAnnealingLR(optimizer, T_max=self.cfg.training.epochs-3, eta_min=self.cfg.training.lr*0.01)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[3])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["cfg"] = self.cfg
