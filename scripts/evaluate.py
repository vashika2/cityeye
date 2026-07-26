import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch, torch.serialization, json
from omegaconf import OmegaConf, DictConfig
from src.training.trainer import CityEyeTrainer
from src.datasets.traffic_dataset import TrafficDataModule

torch.serialization.add_safe_globals([DictConfig])
CONGESTION_LABELS = ["free_flow", "moderate", "dense", "gridlock"]


def evaluate(cfg_path, checkpoint_path):
    cfg = OmegaConf.load(cfg_path)
    model = CityEyeTrainer.load_from_checkpoint(checkpoint_path, cfg=cfg, map_location="cpu", weights_only=False)
    model.eval()
    val_loader = TrafficDataModule(cfg).val_dataloader()
    correct, total, all_results = 0, 0, []
    with torch.no_grad():
        for batch in val_loader:
            outputs = model.model(batch["image"])
            if "density" in outputs:
                preds = outputs["density"].argmax(dim=1)
                targets = batch["congestion"]
                correct += (preds == targets).sum().item()
                total += len(targets)
                for i in range(len(preds)):
                    all_results.append({"predicted": CONGESTION_LABELS[preds[i].item()], "actual": CONGESTION_LABELS[targets[i].item()], "correct": preds[i].item() == targets[i].item()})
    accuracy = correct / total if total > 0 else 0
    print(f"Congestion Accuracy: {accuracy:.3f} ({correct}/{total})")
    from collections import defaultdict
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in all_results:
        per_class[r["actual"]]["total"] += 1
        if r["correct"]: per_class[r["actual"]]["correct"] += 1
    for cls, counts in per_class.items():
        print(f"  {cls}: {counts['correct']}/{counts['total']}")
    output_path = Path("runs/evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"overall_accuracy": accuracy, "per_class": {k: dict(v) for k, v in per_class.items()}, "predictions": all_results}, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)
