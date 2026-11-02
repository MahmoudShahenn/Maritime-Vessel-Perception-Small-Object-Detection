"""
Image Inference Script for Maritime Vessel Detection (CVOD01)
Runs inference on single or multiple images, draws high-contrast maritime bounding boxes,
and saves annotated predictions.
"""

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import cv2
import torch
from ultralytics import YOLO

from src.utils import CLASS_NAMES, draw_maritime_detections

def parse_args():
    parser = argparse.ArgumentParser(description="Run Maritime Vessel Detection on Images")
    parser.add_argument("--source", type=str, required=True, help="Path to image file or directory of images")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model weights (.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="Device ('0', 'cpu')")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions/images", help="Output directory")
    return parser.parse_args()

def run_image_inference():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(args.source)
    if source_path.is_file():
        image_files = [source_path]
    elif source_path.is_dir():
        exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
        image_files = []
        for ext in exts:
            image_files.extend(list(source_path.glob(ext)))
        image_files = sorted(image_files)
    else:
        print(f"[!] Invalid source path: {args.source}")
        sys.exit(1)

    print(f"[*] Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"[*] Processing {len(image_files)} image(s)...")
    for img_p in image_files:
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue

        results = model.predict(
            str(img_p),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False
        )[0]

        boxes = []
        scores = []
        labels = []

        for b in results.boxes:
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            c = float(b.conf[0].cpu().item())
            cls_id = int(b.cls[0].cpu().item())
            boxes.append(xyxy)
            scores.append(c)
            labels.append(cls_id)

        annotated = draw_maritime_detections(
            image=img_bgr,
            boxes=boxes,
            scores=scores,
            labels=labels,
            class_names=CLASS_NAMES,
            conf_thresh=args.conf
        )

        dst_path = out_dir / f"pred_{img_p.name}"
        cv2.imwrite(str(dst_path), annotated)
        print(f"[+] Saved prediction ({len(boxes)} vessels detected): {dst_path}")

if __name__ == "__main__":
    run_image_inference()

