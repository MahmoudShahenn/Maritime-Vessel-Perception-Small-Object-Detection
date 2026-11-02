"""
Training Pipeline for SeaShips Vessel Detection (CVOD01)
Supports baseline and small-object enhanced experiments using Ultralytics YOLO.
Monitors training progress, exports checkpoints, logs curves, and records run metrics.
"""

import os
import sys
import argparse
import json
import yaml
import shutil
from pathlib import Path
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Train SeaShips Maritime Vessel Detector")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to YAML training configuration")
    parser.add_argument("--model", type=str, default=None, help="YOLO model checkpoint or architecture (e.g. yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override input image resolution")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('0', 'cpu', etc.)")
    parser.add_argument("--name", type=str, default="baseline_yolov8s_640", help="Run / experiment name")
    parser.add_argument("--project", type=str, default="outputs/models", help="Destination folder for artifacts")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers")
    return parser.parse_args()

def load_yaml(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def run_training():
    args = parse_args()

    # Load configuration
    cfg = {}
    if os.path.exists(args.config):
        print(f"[*] Loading configuration from {args.config}")
        cfg = load_yaml(args.config)
    else:
        print(f"[!] Warning: Config {args.config} not found, using defaults.")

    # Apply overrides
    model_name = args.model or cfg.get("model", "yolov8s.pt")
    epochs = args.epochs or cfg.get("epochs", 25)
    batch = args.batch or cfg.get("batch", 16)
    imgsz = args.imgsz or cfg.get("imgsz", 640)
    data_yaml = cfg.get("data", "configs/dataset.yaml")
    run_name = args.name
    project = args.project
    workers = args.workers

    # Inspect device
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = "0"
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"[+] CUDA GPU Detected: {gpu_name} ({vram_mb:.0f} MB VRAM)")
    else:
        device = "cpu"
        print("[!] No CUDA GPU detected, training on CPU.")

    print(f"[*] Configuration Summary:")
    print(f"    - Experiment: {run_name}")
    print(f"    - Model:      {model_name}")
    print(f"    - Image Size: {imgsz}x{imgsz}")
    print(f"    - Batch Size: {batch}")
    print(f"    - Epochs:     {epochs}")
    print(f"    - Device:     {device}")
    print(f"    - Data:       {data_yaml}")

    # Import ultralytics
    from ultralytics import YOLO

    model = YOLO(model_name)

    # Collect maritime augmentation settings from cfg
    train_kwargs = {
        "data": data_yaml,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "workers": workers,
        "project": project,
        "name": run_name,
        "exist_ok": True,
        "pretrained": cfg.get("pretrained", True),
        "optimizer": cfg.get("optimizer", "auto"),
        "lr0": cfg.get("lr0", 0.01),
        "lrf": cfg.get("lrf", 0.01),
        "momentum": cfg.get("momentum", 0.937),
        "weight_decay": cfg.get("weight_decay", 0.0005),
        "warmup_epochs": cfg.get("warmup_epochs", 3.0),
        "patience": cfg.get("patience", 10),
        "fliplr": cfg.get("fliplr", 0.5),
        "flipud": cfg.get("flipud", 0.0),      # Strict 0.0 for maritime
        "degrees": cfg.get("degrees", 2.0),
        "translate": cfg.get("translate", 0.1),
        "scale": cfg.get("scale", 0.5),
        "shear": cfg.get("shear", 0.0),
        "perspective": cfg.get("perspective", 0.0),
        "hsv_h": cfg.get("hsv_h", 0.015),
        "hsv_s": cfg.get("hsv_s", 0.4),
        "hsv_v": cfg.get("hsv_v", 0.4),
        "mosaic": cfg.get("mosaic", 1.0),
        "mixup": cfg.get("mixup", 0.0),
        "save": True,
        "val": True,
        "plots": True,
    }

    if "box" in cfg:
        train_kwargs["box"] = cfg["box"]
    if "cls" in cfg:
        train_kwargs["cls"] = cfg["cls"]
    if "dfl" in cfg:
        train_kwargs["dfl"] = cfg["dfl"]
    if "close_mosaic" in cfg:
        train_kwargs["close_mosaic"] = cfg["close_mosaic"]

    print("[*] Launching training process...")
    results = model.train(**train_kwargs)

    # Copy best weights to a clean canonical location
    best_pt = None
    if hasattr(model, "trainer") and hasattr(model.trainer, "best") and model.trainer.best:
        best_pt = Path(model.trainer.best)
    
    if best_pt is None or not best_pt.exists():
        candidates = [
            Path(project) / run_name / "weights" / "best.pt",
            Path("runs/detect") / project / run_name / "weights" / "best.pt",
            Path("runs/detect") / run_name / "weights" / "best.pt"
        ]
        best_pt = next((p for p in candidates if p.exists()), None)

    if best_pt and best_pt.exists():
        canonical_dest = Path(project) / f"{run_name}_best.pt"
        canonical_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, canonical_dest)
        print(f"[+] Canonical best model copied to: {canonical_dest}")
    else:
        print(f"[!] Note: Best weights path could not be located automatically.")

    print(f"[+] Training complete for {run_name}!")
    return results

if __name__ == "__main__":
    run_training()

