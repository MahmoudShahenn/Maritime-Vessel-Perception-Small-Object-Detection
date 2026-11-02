"""
Reconciled Numerical Audit & Evaluation Script (CVOD01)
Maritime Vessel Detection & Small-Object Perception

Mathematically reconciled and verified evaluation script:
1. Fixes ground-truth scale attribution bug so TP + FN == GT across all scale categories.
2. Computes both:
   A) Discrete point-in-time matching metrics at fixed operational thresholds (conf=0.25, IoU=0.5).
   B) Ultralytics PR-curve integrated benchmark metrics (mAP@0.5, mAP@0.5:0.95, curve-optimal max-F1 point).
   C) Operational confidence sweep across [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70].
"""

import sys
import json
from pathlib import Path
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import (
    CLASS_NAMES,
    categorize_scale,
    compute_box_iou,
    xywh_norm_to_xyxy_pixels
)

def run_reconciled_audit(model_path: str = "outputs/models/baseline_yolov8s_640_best.pt",
                         conf_thresh: float = 0.25,
                         nms_iou: float = 0.6,
                         match_iou: float = 0.5):
    print(f"[*] Loading model from {model_path}...")
    model = YOLO(model_path)

    # 1. Ultralytics official PR-curve validation on test split
    print(f"[*] Running official model.val() on split='test'...")
    val_res = model.val(data="configs/dataset.yaml", split="test", imgsz=640, conf=conf_thresh, iou=nms_iou, device="0", verbose=False)
    box = val_res.box
    nt_per_class = val_res.nt_per_class.tolist()

    ultralytics_curve_metrics = {
        "macro_precision_mp": float(box.mp),
        "macro_recall_mr": float(box.mr),
        "weighted_recall": float((box.r * (val_res.nt_per_class / val_res.nt_per_class.sum())).sum()),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": {}
    }
    for cid, cname in enumerate(CLASS_NAMES):
        ultralytics_curve_metrics["per_class"][cname] = {
            "gt_count": int(nt_per_class[cid]),
            "p_curve_opt": float(box.p[cid]),
            "r_curve_opt": float(box.r[cid]),
            "f1_curve_opt": float(box.f1[cid]),
            "ap50": float(box.ap50[cid]),
            "ap50_95": float(box.ap[cid])
        }

    # 2. Load ground truths and raw predictions at base conf=0.10
    print(f"[*] Loading test set annotations and generating predictions at base conf=0.10...")
    with open("data/splits/test.txt", "r") as f:
        test_images = [Path(line.strip()) for line in f if line.strip()]

    all_gts = []
    all_preds_raw = []

    for img_p in test_images:
        label_p = Path(str(img_p).replace("images", "labels").replace(".jpg", ".txt"))
        gts = []
        if label_p.exists():
            with open(label_p, "r") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cid = int(parts[0])
                        xc, yc, nw, nh = map(float, parts[1:])
                        xmin, ymin, xmax, ymax = xywh_norm_to_xyxy_pixels((xc, yc, nw, nh), 1920, 1080)
                        area = (xmax - xmin) * (ymax - ymin)
                        gts.append({"cls": cid, "box": [xmin, ymin, xmax, ymax], "area": area})
        all_gts.append(gts)

        pred = model.predict(str(img_p), imgsz=640, conf=0.10, iou=nms_iou, device="0", verbose=False)[0]
        preds = []
        for b in pred.boxes:
            b_xyxy = b.xyxy[0].cpu().numpy().tolist()
            b_conf = float(b.conf[0].cpu().item())
            b_cls = int(b.cls[0].cpu().item())
            b_area = (b_xyxy[2] - b_xyxy[0]) * (b_xyxy[3] - b_xyxy[1])
            preds.append({"cls": b_cls, "conf": b_conf, "box": b_xyxy, "area": b_area})
        all_preds_raw.append(preds)

    def evaluate_at_threshold(thresh):
        class_stats = {c: {"gt": 0, "tp": 0, "fp": 0, "fn": 0} for c in range(6)}
        coco_size_stats = {
            "small": {"gt": 0, "tp": 0, "fp": 0, "fn": 0},
            "medium": {"gt": 0, "tp": 0, "fp": 0, "fn": 0},
            "large": {"gt": 0, "tp": 0, "fp": 0, "fn": 0}
        }
        maritime_size_stats = {
            "distant_small": {"gt": 0, "tp": 0, "fp": 0, "fn": 0},
            "near_large": {"gt": 0, "tp": 0, "fp": 0, "fn": 0}
        }

        for gts_raw, preds_raw in zip(all_gts, all_preds_raw):
            gts = [dict(g, matched=False) for g in gts_raw]
            preds = [dict(p, matched=False) for p in preds_raw if p["conf"] >= thresh]

            # Ground truth accounting
            for gt in gts:
                class_stats[gt["cls"]]["gt"] += 1
                cat = categorize_scale(gt["area"])
                coco_size_stats[cat]["gt"] += 1
                m_cat = "distant_small" if gt["area"] < 10368 else "near_large"
                maritime_size_stats[m_cat]["gt"] += 1

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

                if best_iou >= match_iou and best_gt_idx >= 0:
                    p["matched"] = True
                    matched_gt = gts[best_gt_idx]
                    matched_gt["matched"] = True
                    
                    class_stats[matched_gt["cls"]]["tp"] += 1
                    cat = categorize_scale(matched_gt["area"])
                    coco_size_stats[cat]["tp"] += 1
                    m_cat = "distant_small" if matched_gt["area"] < 10368 else "near_large"
                    maritime_size_stats[m_cat]["tp"] += 1
                else:
                    class_stats[p["cls"]]["fp"] += 1
                    cat = categorize_scale(p["area"])
                    coco_size_stats[cat]["fp"] += 1
                    m_cat = "distant_small" if p["area"] < 10368 else "near_large"
                    maritime_size_stats[m_cat]["fp"] += 1

            for gt in gts:
                if not gt["matched"]:
                    class_stats[gt["cls"]]["fn"] += 1
                    cat = categorize_scale(gt["area"])
                    coco_size_stats[cat]["fn"] += 1
                    m_cat = "distant_small" if gt["area"] < 10368 else "near_large"
                    maritime_size_stats[m_cat]["fn"] += 1

        tot_gt = sum(class_stats[c]["gt"] for c in range(6))
        tot_tp = sum(class_stats[c]["tp"] for c in range(6))
        tot_fp = sum(class_stats[c]["fp"] for c in range(6))
        tot_fn = sum(class_stats[c]["fn"] for c in range(6))
        p = tot_tp / (tot_tp + tot_fp) if tot_tp + tot_fp > 0 else 0.0
        r = tot_tp / (tot_tp + tot_fn) if tot_tp + tot_fn > 0 else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0

        return {
            "overall": {
                "total_gt": tot_gt,
                "total_tp": tot_tp,
                "total_fp": tot_fp,
                "total_fn": tot_fn,
                "tp_plus_fn": tot_tp + tot_fn,
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1)
            },
            "class_stats": class_stats,
            "coco_size": coco_size_stats,
            "maritime_size": maritime_size_stats
        }

    # Evaluate at default operational threshold (conf_thresh=0.25)
    print(f"[*] Reconciling discrete evaluation at operational conf={conf_thresh}...")
    base_eval = evaluate_at_threshold(conf_thresh)
    tot_gt = base_eval["overall"]["total_gt"]
    tot_tp = base_eval["overall"]["total_tp"]
    tot_fp = base_eval["overall"]["total_fp"]
    tot_fn = base_eval["overall"]["total_fn"]

    discrete_class_results = {}
    class_p_list, class_r_list, class_f1_list = [], [], []
    for cid in range(6):
        cname = CLASS_NAMES[cid]
        s = base_eval["class_stats"][cid]
        gt, tp, fp, fn = s["gt"], s["tp"], s["fp"], s["fn"]
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        class_p_list.append(p)
        class_r_list.append(r)
        class_f1_list.append(f1)
        discrete_class_results[cname] = {
            "gt": gt, "tp": tp, "fp": fp, "fn": fn,
            "tp_plus_fn": tp + fn,
            "precision": float(p), "recall": float(r), "f1": float(f1)
        }

    reconciled_coco_size = {}
    for cat in ["small", "medium", "large"]:
        s = base_eval["coco_size"][cat]
        gt, tp, fp, fn = s["gt"], s["tp"], s["fp"], s["fn"]
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        reconciled_coco_size[cat] = {
            "gt": gt, "tp": tp, "fp": fp, "fn": fn,
            "tp_plus_fn": tp + fn,
            "precision": float(p), "recall": float(r), "f1": float(f1)
        }

    reconciled_maritime_size = {}
    for cat in ["distant_small", "near_large"]:
        s = base_eval["maritime_size"][cat]
        gt, tp, fp, fn = s["gt"], s["tp"], s["fp"], s["fn"]
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        reconciled_maritime_size[cat] = {
            "gt": gt, "tp": tp, "fp": fp, "fn": fn,
            "tp_plus_fn": tp + fn,
            "precision": float(p), "recall": float(r), "f1": float(f1)
        }

    discrete_overall = {
        "total_gt": tot_gt,
        "total_tp": tot_tp,
        "total_fp": tot_fp,
        "total_fn": tot_fn,
        "tp_plus_fn": tot_tp + tot_fn,
        "micro_precision": float(base_eval["overall"]["precision"]),
        "micro_recall": float(base_eval["overall"]["recall"]),
        "micro_f1": float(base_eval["overall"]["f1"]),
        "macro_precision": float(np.mean(class_p_list)),
        "macro_recall": float(np.mean(class_r_list)),
        "macro_f1": float(np.mean(class_f1_list))
    }

    # Evaluate across sweep of confidence operating thresholds
    print("[*] Computing operational confidence curve sweep...")
    conf_sweep_results = {}
    for th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]:
        th_eval = evaluate_at_threshold(th)["overall"]
        conf_sweep_results[f"conf_{th:.2f}"] = {
            "conf_threshold": th,
            "gt": th_eval["total_gt"],
            "tp": th_eval["total_tp"],
            "fp": th_eval["total_fp"],
            "fn": th_eval["total_fn"],
            "tp_plus_fn": th_eval["tp_plus_fn"],
            "precision": th_eval["precision"],
            "recall": th_eval["recall"],
            "f1": th_eval["f1"]
        }

    reconciliation_data = {
        "evaluation_settings": {
            "model_path": model_path,
            "split": "test",
            "image_count": len(test_images),
            "conf_threshold": conf_thresh,
            "nms_iou_threshold": nms_iou,
            "matching_iou_threshold": match_iou
        },
        "ultralytics_pr_curve_evaluation": ultralytics_curve_metrics,
        "discrete_matching_evaluation": {
            "overall": discrete_overall,
            "per_class": discrete_class_results,
            "coco_size_stratified": reconciled_coco_size,
            "maritime_size_stratified": reconciled_maritime_size
        },
        "confidence_threshold_operational_sweep": conf_sweep_results
    }

    out_json = Path("outputs/results/reconciled_evaluation_audit.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(reconciliation_data, f, indent=2)

    print(f"\n[+] Reconciled evaluation audit saved to: {out_json}")
    return reconciliation_data

if __name__ == "__main__":
    run_reconciled_audit()
