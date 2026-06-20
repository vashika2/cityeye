import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf
from src.inference.run import run_inference

if __name__ == "__main__":
    cfg = OmegaConf.load("configs/default.yaml")
    run_inference(cfg)