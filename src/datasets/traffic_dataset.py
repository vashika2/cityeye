import json, cv2, torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from src.preprocessing.pipeline import get_transforms

CONGESTION_LABELS = {"free_flow": 0, "moderate": 1, "dense": 2, "gridlock": 3}


class TrafficDataset(Dataset):
    def __init__(self, data_dir, split="train", image_size=(640, 640)):
        self.data_dir = Path(data_dir)
        self.transforms = get_transforms(split, image_size)
        ann_path = self.data_dir / f"{split}.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")
        with open(ann_path) as f:
            data = json.load(f)
        self.images = data["images"]
        self.annotations = {}
        for ann in data.get("annotations", []):
            self.annotations.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]
        image = cv2.cvtColor(cv2.imread(str(self.data_dir / "images" / img_info["file_name"])), cv2.COLOR_BGR2RGB)
        anns = self.annotations.get(img_id, [])
        transformed = self.transforms(image=image, bboxes=[a["bbox"] for a in anns], labels=[a["category_id"] for a in anns])
        violation_types = ["no_helmet", "signal_jump", "wrong_way"]
        anomaly_types = ["stopped_vehicle", "accident", "pedestrian_on_road"]
        return {
            "image": transformed["image"],
            "boxes": torch.tensor(transformed["bboxes"], dtype=torch.float32) if transformed["bboxes"] else torch.zeros((0, 4)),
            "labels": torch.tensor(transformed["labels"], dtype=torch.long),
            "congestion": torch.tensor(CONGESTION_LABELS.get(img_info.get("congestion_label", "free_flow"), 0), dtype=torch.long),
            "violations": torch.tensor([1.0 if v in img_info.get("violations", []) else 0.0 for v in violation_types], dtype=torch.float32),
            "anomalies": torch.tensor([1.0 if a in img_info.get("anomalies", []) else 0.0 for a in anomaly_types], dtype=torch.float32),
            "image_id": torch.tensor(img_id),
        }


def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "congestion": torch.stack([b["congestion"] for b in batch]),
        "violations": torch.stack([b["violations"] for b in batch]),
        "anomalies": torch.stack([b["anomalies"] for b in batch]),
        "image_id": torch.stack([b["image_id"] for b in batch]),
    }


class TrafficDataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_dir = cfg.data.processed_dir
        self.image_size = tuple(cfg.data.image_size)
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.data.num_workers

    def train_dataloader(self):
        return DataLoader(TrafficDataset(self.data_dir, "train", self.image_size), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(TrafficDataset(self.data_dir, "val", self.image_size), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True)
