"""
Comprehensive Evaluation Pipeline for SeaShips Vessel Detection (CVOD01)
Evaluates trained models on the untouched test partition:
- Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95
- Per-class metrics
- Size-stratified metrics (Small, Medium, Large under COCO standards and relative maritime scales)
- False Positive (FP) and False Negative (FN) failure case identification & export
- Latency & FPS profiling
"""

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import time
import cv2
import numpy as np
from collections import defaultdict
import torch
from ultralytics import YOLO

from src.utils import (
    CLASS_NAMES,
    compute_box_iou,
    categorize_scale,
    draw_maritime_detections,
    benchmark_inference_speed,
    xywh_norm_to_xyxy_pixels
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SeaShips Vessel Detector")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--data", type=str, default="configs/dataset.yaml", help="Path to dataset YAML")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="Device ('0', 'cpu')")
    parser.add_argument("--name", type=str, default="evaluation_run", help="Experiment name")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--max-error-samples", type=int, default=15, help="Number of FP/FN sample images to save")
    return parser.parse_args()

def evaluate_model():
    args = parse_args()
    out_dir = Path(args.output_dir)
    results_dir = out_dir / "results"
    figures_dir = out_dir / "figures"
    errors_dir = out_dir / "predictions" / "errors"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading model from: {args.model}")
    model = YOLO(args.model)

    print(f"[*] Running official validation on split='{args.split}' at imgsz={args.imgsz}...")
    val_results = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_json=True,
        plots=True,
        verbose=False
    )

    # Extract primary metrics
    box_metrics = val_results.box
    precision = float(box_metrics.mp)
    recall = float(box_metrics.mr)
    map50 = float(box_metrics.map50)
    map50_95 = float(box_metrics.map)
    f1 = float(2 * (precision * recall) / (precision + recall + 1e-8))

    print(f"[+] Overall Evaluation Metrics ({args.split} set):")
    print(f"    - Precision:   {precision:.4f}")
    print(f"    - Recall:      {recall:.4f}")
    print(f"    - F1-Score:    {f1:.4f}")
    print(f"    - mAP@0.5:     {map50:.4f}")
    print(f"    - mAP@0.5:0.95:{map50_95:.4f}")

    # Per-class metrics
    per_class_metrics = {}
    for idx, cname in enumerate(CLASS_NAMES):
        if idx < len(box_metrics.p):
            p_c = float(box_metrics.p[idx])
            r_c = float(box_metrics.r[idx])
            f1_c = float(2 * (p_c * r_c) / (p_c + r_c + 1e-8))
            ap50_c = float(box_metrics.ap50[idx]) if idx < len(box_metrics.ap50) else 0.0
            ap_c = float(box_metrics.ap[idx]) if idx < len(box_metrics.ap) else 0.0
            per_class_metrics[cname] = {
                "precision": p_c,
                "recall": r_c,
                "f1": f1_c,
                "ap50": ap50_c,
                "ap50_95": ap_c
            }
            print(f"    - {cname:20s}: P={p_c:.3f}, R={r_c:.3f}, F1={f1_c:.3f}, AP50={ap50_c:.3f}, AP50-95={ap_c:.3f}")

    # Detailed size-stratified & error analysis on test images
    print(f"[*] Performing size-stratified evaluation & failure case extraction...")
    
    # Load test split images and labels
    split_file = Path(f"data/splits/{args.split}.txt")
    if not split_file.exists():
        split_file = Path(f"data/processed/images/{args.split}")
        test_images = sorted(list(split_file.glob("*.jpg")))
    else:
        with open(split_file, "r") as f:
            test_images = [Path(line.strip()) for line in f if line.strip()]

    size_stats = {
        "small": {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0},
        "medium": {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0},
        "large": {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0}
    }

    error_samples = []
    
    # Benchmark sample for speed
    speed_stats = {"fps": 0.0, "total_latency_ms": 0.0}
    if test_images:
        print(f"[*] Benchmarking inference speed on {len(test_images)} frames...")
        sample_img = str(test_images[0])
        speed_stats = benchmark_inference_speed(
            model=model,
            test_img_path=sample_img,
            imgsz=args.imgsz,
            device=args.device,
            iterations=50,
            warmup=15
        )
        print(f"[+] Benchmarked Speed: {speed_stats['fps']:.1f} FPS (Total Latency: {speed_stats['total_latency_ms']:.2f} ms: pre={speed_stats['preprocess_ms']:.2f}ms, inf={speed_stats['inference_ms']:.2f}ms, post={speed_stats['postprocess_ms']:.2f}ms)")

    # Analyze test images for detailed size and error cases
    analyzed_count = len(test_images)
    for i in range(analyzed_count):
        img_p = test_images[i]
        label_p = Path(str(img_p).replace("images", "labels").replace(".jpg", ".txt"))
        
        # Load ground truth
        gts = []
        if label_p.exists():
            with open(label_p, "r") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cid = int(parts[0])
                        xc, yc, nw, nh = map(float, parts[1:])
                        # Convert to 1920x1080 pixels
                        xmin, ymin, xmax, ymax = xywh_norm_to_xyxy_pixels((xc, yc, nw, nh), 1920, 1080)
                        area = (xmax - xmin) * (ymax - ymin)
                        gts.append({"cls": cid, "box": [xmin, ymin, xmax, ymax], "area": area, "matched": False})

        # Run prediction
        pred = model.predict(str(img_p), imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
        preds = []
        for b in pred.boxes:
            b_xyxy = b.xyxy[0].cpu().numpy().tolist()
            b_conf = float(b.conf[0].cpu().item())
            b_cls = int(b.cls[0].cpu().item())
            b_area = (b_xyxy[2] - b_xyxy[0]) * (b_xyxy[3] - b_xyxy[1])
            preds.append({"cls": b_cls, "conf": b_conf, "box": b_xyxy, "area": b_area, "matched": False})

        # Count GT scales
        for gt in gts:
            cat = categorize_scale(gt["area"])
            size_stats[cat]["gt_count"] += 1

        # Match predictions to GT with IoU threshold 0.5 (descending confidence order)
        preds.sort(key=lambda x: x["conf"], reverse=True)
        for p in preds:
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, gt in enumerate(gts):
                if gt["matched"] or gt["cls"] != p["cls"]:
                    continue
                iou = compute_box_iou(np.array(p["box"]), np.array(gt["box"]))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= 0.5 and best_gt_idx >= 0:
                p["matched"] = True
                matched_gt = gts[best_gt_idx]
                matched_gt["matched"] = True
                # Attribute True Positive to Ground Truth scale
                cat = categorize_scale(matched_gt["area"])
                size_stats[cat]["tp"] += 1
            else:
                # Attribute False Positive to Predicted scale
                p_cat = categorize_scale(p["area"])
                size_stats[p_cat]["fp"] += 1

        # Record missed GTs (False Negatives)
        for gt in gts:
            if not gt["matched"]:
                cat = categorize_scale(gt["area"])
                size_stats[cat]["fn"] += 1

        # Check for interesting failure case images
        has_fp = any(not p["matched"] for p in preds)
        has_fn = any(not gt["matched"] for gt in gts)
        if (has_fp or has_fn) and len(error_samples) < args.max_error_samples:
            img_bgr = cv2.imread(str(img_p))
            if img_bgr is not None:
                # Draw Ground Truths in Green
                gt_vis = img_bgr.copy()
                for gt in gts:
                    gx1, gy1, gx2, gy2 = map(int, gt["box"])
                    cv2.rectangle(gt_vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                    cv2.putText(gt_vis, f"GT: {CLASS_NAMES[gt['cls']]}", (gx1, max(gy1 - 6, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                # Draw Predictions in Magenta/Red
                pred_vis = gt_vis.copy()
                for p in preds:
                    px1, py1, px2, py2 = map(int, p["box"])
                    color = (255, 0, 255) if p["matched"] else (0, 0, 255)
                    status_text = "TP" if p["matched"] else "FP"
                    cv2.rectangle(pred_vis, (px1, py1), (px2, py2), color, 2)
                    cv2.putText(pred_vis, f"{status_text}: {CLASS_NAMES[p['cls']]} {p['conf']:.2f}",
                                (px1, min(py2 + 15, 1075)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                err_name = f"error_{img_p.stem}_fp{has_fp}_fn{has_fn}.jpg"
                cv2.imwrite(str(errors_dir / err_name), pred_vis)
                error_samples.append(err_name)

    # Compute size-stratified Precision, Recall, F1
    size_breakdown = {}
    for cat in ["small", "medium", "large"]:
        s_tp = size_stats[cat]["tp"]
        s_fp = size_stats[cat]["fp"]
        s_fn = size_stats[cat]["fn"]
        s_prec = s_tp / (s_tp + s_fp + 1e-8) if (s_tp + s_fp) > 0 else 0.0
        s_rec = s_tp / (s_tp + s_fn + 1e-8) if (s_tp + s_fn) > 0 else 0.0
        s_f1 = 2 * (s_prec * s_rec) / (s_prec + s_rec + 1e-8)
        size_breakdown[cat] = {
            "gt_instances": size_stats[cat]["gt_count"],
            "true_positives": s_tp,
            "false_positives": s_fp,
            "false_negatives": s_fn,
            "precision": float(s_prec),
            "recall": float(s_rec),
            "f1": float(s_f1)
        }
        print(f"    - Scale [{cat.upper():6s}]: GT={size_stats[cat]['gt_count']}, TP={s_tp}, FP={s_fp}, FN={s_fn} -> P={s_prec:.3f}, R={s_rec:.3f}, F1={s_f1:.3f}")

    # Compile Full Evaluation Report
    eval_report = {
        "experiment_name": args.name,
        "model_checkpoint": args.model,
        "split": args.split,
        "imgsz": args.imgsz,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "device": args.device,
        "overall_metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "map50": map50,
            "map50_95": map50_95
        },
        "per_class_metrics": per_class_metrics,
        "size_breakdown": size_breakdown,
        "latency_profile": speed_stats,
        "saved_error_samples": error_samples
    }

    report_path = results_dir / f"{args.name}_evaluation_report.json"
    with open(report_path, "w") as rf:
        json.dump(eval_report, rf, indent=2)

    print(f"[+] Evaluation report exported to: {report_path}")
    return eval_report

if __name__ == "__main__":
    evaluate_model()

