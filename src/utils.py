"""
Utility functions for SeaShips Maritime Vessel Detection (CVOD01)
- Coordinate conversions & IoU computations
- Size-stratified evaluation utilities (COCO scale definitions)
- Visualization & bounding box rendering
- Benchmarking and latency measurement
"""

import time
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional

# Standard class mapping
CLASS_NAMES = [
    "ore carrier",
    "bulk cargo carrier",
    "general cargo ship",
    "container ship",
    "fishing boat",
    "passenger ship"
]

# High-contrast color palette for maritime visualization (RGB)
CLASS_COLORS = [
    (230, 25, 75),    # Red - Ore carrier
    (60, 180, 75),    # Green - Bulk cargo
    (255, 225, 25),   # Yellow - General cargo
    (0, 130, 200),    # Blue - Container ship
    (245, 130, 48),   # Orange - Fishing boat
    (145, 30, 180),   # Purple - Passenger ship
]

def xywh_norm_to_xyxy_pixels(box: Tuple[float, float, float, float], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Convert normalized YOLO (xc, yc, w, h) to absolute pixel (xmin, ymin, xmax, ymax)."""
    xc, yc, w, h = box
    xmin = int(round((xc - w / 2.0) * img_w))
    ymin = int(round((yc - h / 2.0) * img_h))
    xmax = int(round((xc + w / 2.0) * img_w))
    ymax = int(round((yc + h / 2.0) * img_h))
    return max(0, xmin), max(0, ymin), min(img_w, xmax), min(img_h, ymax)

def xyxy_pixels_to_xywh_norm(box: Tuple[float, float, float, float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert absolute pixel (xmin, ymin, xmax, ymax) to normalized YOLO (xc, yc, w, h)."""
    xmin, ymin, xmax, ymax = box
    xmin = max(0.0, min(float(img_w), float(xmin)))
    ymin = max(0.0, min(float(img_h), float(ymin)))
    xmax = max(0.0, min(float(img_w), float(xmax)))
    ymax = max(0.0, min(float(img_h), float(ymax)))

    box_w = max(0.0, xmax - xmin)
    box_h = max(0.0, ymax - ymin)

    xc = (xmin + xmax) / (2.0 * img_w) if img_w > 0 else 0.0
    yc = (ymin + ymax) / (2.0 * img_h) if img_h > 0 else 0.0
    nw = box_w / float(img_w) if img_w > 0 else 0.0
    nh = box_h / float(img_h) if img_h > 0 else 0.0

    return xc, yc, nw, nh

def compute_box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) of two boxes in [xmin, ymin, xmax, ymax]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    return (intersection / union) if union > 0 else 0.0

def categorize_scale(area: float, small_thresh: float = 32**2, medium_thresh: float = 96**2) -> str:
    """Categorize box scale according to standard COCO definitions."""
    if area < small_thresh:
        return "small"
    elif area < medium_thresh:
        return "medium"
    return "large"

def draw_maritime_detections(
    image: np.ndarray,
    boxes: List[Tuple[float, float, float, float]],
    scores: List[float],
    labels: List[int],
    class_names: Optional[List[str]] = None,
    conf_thresh: float = 0.25,
    is_gt: bool = False
) -> np.ndarray:
    """
    Renders bounding boxes and labels onto RGB/BGR image for clear maritime visualization.
    """
    img = image.copy()
    if class_names is None:
        class_names = CLASS_NAMES

    h, w = img.shape[:2]

    for i, box in enumerate(boxes):
        score = scores[i] if scores is not None and len(scores) > i else 1.0
        if score < conf_thresh:
            continue

        label_id = int(labels[i]) if labels is not None and len(labels) > i else 0
        cname = class_names[label_id] if 0 <= label_id < len(class_names) else f"cls_{label_id}"

        # Check box format
        if max(box) <= 1.05:
            # Normalized xywh or xyxy
            # Assume xyxy if box[2] > box[0] else xywh
            xc, yc, bw, bh = box
            xmin, ymin, xmax, ymax = xywh_norm_to_xyxy_pixels((xc, yc, bw, bh), w, h)
        else:
            xmin, ymin, xmax, ymax = [int(round(v)) for v in box]

        xmin = max(0, min(w - 1, xmin))
        ymin = max(0, min(h - 1, ymin))
        xmax = max(0, min(w - 1, xmax))
        ymax = max(0, min(h - 1, ymax))

        # Pick color
        color_rgb = CLASS_COLORS[label_id % len(CLASS_COLORS)]
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        thickness = 2
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_bgr, thickness)

        caption = f"{cname}" if is_gt else f"{cname} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Label banner
        banner_y = max(ymin, th + 4)
        cv2.rectangle(img, (xmin, banner_y - th - 4), (xmin + tw + 4, banner_y + baseline), color_bgr, -1)
        cv2.putText(img, caption, (xmin + 2, banner_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img

def benchmark_inference_speed(model, test_img_path: str, imgsz: int = 640, device: str = "cpu", iterations: int = 100, warmup: int = 20) -> Dict[str, float]:
    """
    Measures realistic batch-1 inference latency and FPS with warmup.
    Returns: {preprocess_ms, inference_ms, postprocess_ms, total_latency_ms, fps}
    """
    import torch
    
    # Warmup
    for _ in range(warmup):
        _ = model.predict(test_img_path, imgsz=imgsz, device=device, verbose=False)

    latencies = []
    pre_ms = []
    inf_ms = []
    post_ms = []

    for _ in range(iterations):
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = model.predict(test_img_path, imgsz=imgsz, device=device, verbose=False)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)
        speed = results[0].speed
        pre_ms.append(speed.get("preprocess", 0.0))
        inf_ms.append(speed.get("inference", 0.0))
        post_ms.append(speed.get("postprocess", 0.0))

    mean_total = float(np.mean(latencies))
    fps = 1000.0 / mean_total if mean_total > 0 else 0.0

    return {
        "preprocess_ms": float(np.mean(pre_ms)),
        "inference_ms": float(np.mean(inf_ms)),
        "postprocess_ms": float(np.mean(post_ms)),
        "total_latency_ms": mean_total,
        "fps": fps
    }

