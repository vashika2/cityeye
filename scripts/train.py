import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os, torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from src.training.trainer import CityEyeTrainer
from src.datasets.traffic_dataset import TrafficDataModule


def train(cfg_path="configs/default.yaml"):
    cfg = OmegaConf.load(cfg_path)
    print(f"Training CityEye | Backbone: {cfg.model.backbone} | Batch: {cfg.training.batch_size} | Epochs: {cfg.training.epochs}")
    L.seed_everything(42)
    data = TrafficDataModule(cfg)
    trainer_module = CityEyeTrainer(cfg)
    callbacks = [
        ModelCheckpoint(dirpath=cfg.training.output_dir, filename="cityeye-{epoch:02d}-{val/total_loss:.3f}", monitor="val/total_loss", mode="min", save_top_k=3, save_last=True),
        ModelCheckpoint(dirpath=cfg.training.output_dir, filename="cityeye-best-acc-{val/congestion_accuracy:.3f}", monitor="val/congestion_accuracy", mode="max", save_top_k=1),
        EarlyStopping(monitor="val/total_loss", patience=10, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]
    os.environ["WANDB_MODE"] = "disabled"
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="mps" if torch.backends.mps.is_available() else "gpu" if torch.cuda.is_available() else "cpu",
        devices=1, precision="32", callbacks=callbacks,
        logger=CSVLogger("runs/logs", name="cityeye"),
        gradient_clip_val=1.0, accumulate_grad_batches=4,
        log_every_n_steps=10, val_check_interval=1.0, enable_model_summary=True,
    )
    trainer.fit(trainer_module, train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader())
    print(f"Training complete. Best model saved to: {cfg.training.output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
