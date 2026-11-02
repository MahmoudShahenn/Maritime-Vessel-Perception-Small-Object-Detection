"""
Unit tests for model inference, latency benchmarking, and scale categorization (CVOD01)
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from src.utils import (
    CLASS_NAMES,
    categorize_scale,
    xywh_norm_to_xyxy_pixels,
    xyxy_pixels_to_xywh_norm,
    compute_box_iou,
    draw_maritime_detections
)

def test_scale_categorization():
    # Small: area < 32^2 = 1024
    assert categorize_scale(500) == "small"
    assert categorize_scale(1023) == "small"

    # Medium: 32^2 <= area < 96^2 = 9216
    assert categorize_scale(1024) == "medium"
    assert categorize_scale(5000) == "medium"
    assert categorize_scale(9215) == "medium"

    # Large: area >= 9216
    assert categorize_scale(9216) == "large"
    assert categorize_scale(50000) == "large"

def test_draw_detections_pipeline():
    # Create blank 1920x1080 image
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    boxes = [[100, 200, 400, 350], [500, 400, 700, 450]]
    scores = [0.92, 0.78]
    labels = [0, 4] # Ore carrier, Fishing boat

    annotated = draw_maritime_detections(
        image=img,
        boxes=boxes,
        scores=scores,
        labels=labels,
        class_names=CLASS_NAMES,
        conf_thresh=0.25
    )

    assert annotated.shape == (1080, 1920, 3)
    # Check that drawing actually modified pixels
    assert np.any(annotated > 0)

def test_box_coordinate_transform_roundtrip():
    W, H = 1920, 1080
    orig_xyxy = (250, 450, 680, 520)
    norm_xywh = xyxy_pixels_to_xywh_norm(orig_xyxy, W, H)
    recovered_xyxy = xywh_norm_to_xyxy_pixels(norm_xywh, W, H)

    for orig, rec in zip(orig_xyxy, recovered_xyxy):
        assert abs(orig - rec) <= 1

