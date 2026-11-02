# Maritime Vessel Detection & Perception (CVOD01)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v8s-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, reproducible, production-grade computer vision pipeline for **maritime vessel detection, classification, and tracking** in coastal and open-sea surveillance video and RGB imagery. Specifically engineered to address **small and distant targets**, **dynamic sea clutter**, and **zero-leakage temporal dataset partitioning**.

---

## 1. Problem Overview & Key Challenges

Maritime surveillance requires reliable early detection of vessels traversing coastal corridors and harbor approaches. The core technical challenges addressed in this solution include:
- **Small & Distant Targets:** Distant vessels occupy less than 0.5% of a 1080p camera frame and are easily lost during naive image downsampling.
- **Dynamic Maritime Backgrounds:** Breaking waves, boat wakes, sun glint/glare, haze, and coastal structures produce severe false alarm risks.
- **Temporal Video Continuity:** Random frame-level train/test splits cause extreme data leakage. We enforce strict sequence-level disjoint partitions.
- **Real-Time Edge Constraints:** Edge patrol systems require sub-15ms inference latency (>60 FPS) on embedded/mobile GPUs (e.g., RTX 3050).

---

## 2. Repository Structure

```text
maritime-vessel-detection/
├── configs/
│   ├── baseline.yaml               # Baseline YOLOv8s (640px) training config
│   ├── small_object_enhanced.yaml  # High-res (800px) & small-object loss config
│   └── dataset.yaml                # Ultralytics dataset specification
├── data/
│   ├── raw/                        # Original Pascal VOC SeaShips data (7,000 images)
│   ├── processed/                  # Normalized YOLO images & labels (train/val/test)
│   └── splits/                     # Sequence-aware split manifests (train/val/test.txt)
├── docs/
│   ├── requirements_checklist.md   # Requirement verification matrix
│   ├── dataset_audit.md            # In-depth dataset & small-object audit report
│   └── technical_report.md         # Comprehensive engineering technical report
├── outputs/
│   ├── figures/                    # Class, bbox, and aspect-ratio diagnostic charts
│   ├── models/                     # Trained weights (best.pt) and training metrics
│   ├── predictions/                # Output detection images, error analysis, & demo video
│   └── results/                    # Quantitative JSON and CSV evaluation summaries
├── scripts/
│   └── run_pipeline.py             # End-to-end master orchestration script
├── src/
│   ├── __init__.py
│   ├── data_audit.py               # Dataset audit & VOC XML validation script
│   ├── prepare_dataset.py          # Sequence-aware partitioning & YOLO converter
│   ├── train.py                    # Training script with validation monitoring
│   ├── evaluate.py                 # Multi-metric & size-stratified evaluation
│   ├── predict_image.py            # Image inference and visualization
│   ├── predict_video.py            # Video inference with ByteTrack vessel tracking
│   └── utils.py                    # Maritime visualization, IoU, and latency profiling
├── tests/
│   ├── test_annotations.py         # Unit tests for XML parsing & box conversions
│   └── test_inference.py           # Unit tests for scale categorization & drawing
├── requirements.txt
└── README.md
```

---

## 3. Environment Setup

### Prerequisites
- Python 3.11+
- NVIDIA GPU with CUDA 12.1+ support (tested on NVIDIA GeForce RTX 3050 Laptop GPU, 4GB VRAM)

### Installation
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install -r requirements.txt
```

---

## 4. Dataset Audit & Preparation

### 1. Run Dataset Audit
Parses all 7,000 Pascal VOC XML files, checks image integrity, and produces class and bounding box distribution plots:
```powershell
python src/data_audit.py data/raw
```

### 2. Generate Leakage-Resistant Splits & YOLO Labels
Groups adjacent frames into 50-frame video chunks and generates a 70% train / 15% val / 15% test partition:
```powershell
python src/prepare_dataset.py
```
Outputs are stored in `data/processed/` and `data/splits/`.

---

## 5. Verification & Unit Tests

Run the automated test suite covering coordinate clamping, VOC-to-YOLO reversibility, IoU computation, and scale classification:
```powershell
python -m pytest -v tests/
```

---

## 6. Training

### Baseline Training (YOLOv8s @ 640px)
```powershell
python src/train.py --config configs/baseline.yaml --epochs 15 --batch 16 --imgsz 640 --name baseline_yolov8s_640
```

### Small-Object Enhanced Training (YOLOv8s @ 800px)
```powershell
python src/train.py --config configs/small_object_enhanced.yaml --epochs 15 --batch 8 --imgsz 800 --name enhanced_yolov8s_800
```

---

## 7. Quantitative Evaluation & Error Analysis

Evaluate on the untouched test partition (`data/splits/test.txt`):
```powershell
python src/evaluate.py --model outputs/models/baseline_yolov8s_640_best.pt --split test --imgsz 640 --conf 0.25 --iou 0.6 --name baseline_test_eval
```

This generates:
- Precision, Recall, F1, mAP@0.5, and mAP@0.5:0.95.
- Per-class metric breakdowns across all 6 vessel types.
- Size-stratified metrics (Small, Medium, Large under COCO standards).
- Failure case extraction (false positives and false negatives saved to `outputs/predictions/errors/`).
- Accurate batch-1 inference latency and FPS profile.

---

## 8. Inference & Video Demonstration

### Single or Batch Image Inference
```powershell
python src/predict_image.py --model outputs/models/baseline_yolov8s_640_best.pt --source data/processed/images/test/ --conf 0.25
```

### Video Inference Demo (with ByteTrack Multi-Object Tracking)
```powershell
python src/predict_video.py --model outputs/models/baseline_yolov8s_640_best.pt --track --output outputs/predictions/demo_output.mp4
```
If no input video is passed, the script automatically compiles a contiguous test sequence into a benchmark maritime surveillance video and runs detection and tracking with live telemetry HUD.

---

## 9. One-Click End-to-End Execution

To run the complete pipeline from scratch:
```powershell
python scripts/run_pipeline.py --epochs 15 --batch 16 --imgsz 640
```

---

## 10. Summary of Architectural & Augmentation Decisions

| Decision | Selection | Rationale |
|---|---|---|
| **Detector** | YOLOv8s (11.2M params) | Anchor-free head for elongated ship geometry; >100 FPS on edge hardware; strong gradient flow via PAN-FPN. |
| **Augmentation** | FlipLR, Scale (0.5–1.5), HSV, Mild Tilt (2°) | Physically grounded for maritime domain. Inverted flips (`flipud`) and transparent `mixup` are strictly disabled. |
| **Splitting** | Sequence-Aware 70/15/15 | Completely isolates contiguous video chunks, eliminating temporal data leakage between train and test. |
| **Small Objects** | High-Res Input (800px) & Box Loss Boost | Prevents severe downsampling destruction of distant targets that occupy <0.5% of the frame. |
| **Tracking** | ByteTrack | Associates low-confidence detection boxes across frames to maintain track continuity through occlusions and sea spray. |

