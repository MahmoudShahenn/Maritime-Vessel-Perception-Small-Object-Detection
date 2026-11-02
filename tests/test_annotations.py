"""
Unit tests for SeaShips dataset annotations and coordinate conversion (CVOD01)
"""

import pytest
from pathlib import Path
from src.prepare_dataset import voc_to_yolo_bbox, convert_annotation
from src.utils import xywh_norm_to_xyxy_pixels, xyxy_pixels_to_xywh_norm, compute_box_iou

def test_voc_to_yolo_conversion_reversibility():
    img_w, img_h = 1920, 1080
    xmin, ymin, xmax, ymax = 633, 467, 944, 510

    # Convert to normalized YOLO
    yolo_box = voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h)
    assert yolo_box is not None
    xc, yc, nw, nh = yolo_box

    assert 0.0 <= xc <= 1.0
    assert 0.0 <= yc <= 1.0
    assert 0.0 <= nw <= 1.0
    assert 0.0 <= nh <= 1.0

    # Convert back to pixel xyxy
    r_xmin, r_ymin, r_xmax, r_ymax = xywh_norm_to_xyxy_pixels(yolo_box, img_w, img_h)

    # Allow 1-pixel rounding tolerance
    assert abs(r_xmin - xmin) <= 1
    assert abs(r_ymin - ymin) <= 1
    assert abs(r_xmax - xmax) <= 1
    assert abs(r_ymax - ymax) <= 1

def test_boundary_clamping():
    img_w, img_h = 1920, 1080
    # Box extending outside image frame
    xmin, ymin, xmax, ymax = -50, -20, 2000, 1200

    yolo_box = voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h)
    assert yolo_box is not None
    xc, yc, nw, nh = yolo_box

    # Should clamp exactly to image boundaries
    assert xc == pytest.approx(0.5, abs=1e-4)
    assert yc == pytest.approx(0.5, abs=1e-4)
    assert nw == pytest.approx(1.0, abs=1e-4)
    assert nh == pytest.approx(1.0, abs=1e-4)

def test_degenerate_zero_area_box():
    img_w, img_h = 1920, 1080
    assert voc_to_yolo_bbox(100, 100, 100, 200, img_w, img_h) is None
    assert voc_to_yolo_bbox(100, 100, 200, 100, img_w, img_h) is None
    assert voc_to_yolo_bbox(200, 200, 100, 100, img_w, img_h) is None

def test_iou_computation():
    box_a = [100, 100, 200, 200]
    box_b = [150, 100, 250, 200]
    # Inter: 50 * 100 = 5000. Union: 10000 + 10000 - 5000 = 15000. IoU = 1/3 ~ 0.3333
    iou = compute_box_iou(box_a, box_b)
    assert iou == pytest.approx(1.0 / 3.0, abs=1e-3)

    # Disjoint boxes
    box_c = [300, 300, 400, 400]
    assert compute_box_iou(box_a, box_c) == 0.0

def test_xml_parsing_real_sample():
    sample_xml = Path("data/raw/Annotations/000001.xml")
    if sample_xml.exists():
        lines, classes, err = convert_annotation(sample_xml)
        assert err is None
        assert len(lines) >= 1
        assert len(classes) >= 1
        first_line_parts = lines[0].split()
        assert len(first_line_parts) == 5
        cid = int(first_line_parts[0])
        assert 0 <= cid <= 5

