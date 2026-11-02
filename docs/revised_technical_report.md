# Technical Report: Maritime Vessel Perception & Small-Object Detection (CVOD01)
*Revised Eddition*

**Author:** Mahmoud Shaheen - Computer Vision Engineer - Perception Systems  
**Dataset:** SeaShips Benchmark Dataset (7,000 Coastline Surveillance Images, Full HD 1920×1080)  
**Hardware Platform:** NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MB VRAM, CUDA 12.1)  
**Framework Environment:** PyTorch 2.5.1+cu121, Torchvision 0.20.1, Ultralytics 8.4.190, OpenCV 5.0.0  
**Verification Date:** October 2026  

---

## 1. Executive Summary

Maritime surveillance systems deployed along coastal corridors, estuaries, and harbor approaches operate under environmental conditions fundamentally distinct from terrestrial computer vision. Maritime targets frequently operate several kilometers away, appearing as compact visual features that occupy less than **0.5% of the total pixel area**, obscured by wave action, boat wakes, solar glitter, sea spray, atmospheric haze, and coastal topography.

This revised technical report presents an evidence-based audit and final evaluation of an end-to-end perception solution for maritime vessel detection, classification, and video tracking. All quantitative metrics, instance counts, latency profiles, and error analyses contained in this document have been reconciled and verified against physical artifacts on disk.

### Principal Verified Results:
1. **Dataset Audit:** 7,000 images and 9,221 vessel annotations across 6 canonical ship classes (`ore carrier`, `bulk cargo carrier`, `general cargo ship`, `container ship`, `fishing boat`, `passenger ship`). 100% file integrity verified (0 missing images, 0 corrupt XML files, 0 invalid bounding boxes).
2. **Leakage-Reduced Temporal Partitioning:** Implemented a chunk-based split grouping consecutive surveillance frames into 140 contiguous 50-frame blocks (70% train / 15% val / 15% test). This reduces adjacent-frame redundancy compared to naive random shuffling.
3. **Verified Test Set Performance:** Evaluated on the untouched test split (1,050 Full HD images, 1,378 ground-truth vessel instances) across two complementary evaluation frameworks:
   - **Continuous PR-Curve Benchmark (Ultralytics protocol):**
     - **mAP@0.5:** 90.21%
     - **mAP@0.5:0.95:** 63.72%
     - **Curve-Optimal Macro Precision:** 93.55%
     - **Curve-Optimal Macro Recall:** 84.92% (Unweighted mean across classes)
     - **Curve-Optimal Instance-Weighted Recall:** 80.48% ($\sum nt_i \cdot r_i / \sum nt_i$)
     - **Macro F1:** 88.67% | **Harmonic Aggregate F1:** 89.03%
   - **Discrete Operational Point-in-Time Matching (`conf=0.25`, matching $\text{IoU} \ge 0.5$):**
     - **Ground Truth Targets:** 1,378
     - **True Positives (TP):** 1,221 | **False Positives (FP):** 292 | **False Negatives (FN):** 157
     - **Mathematical Conservation:** $\mathbf{TP + FN \equiv 1,221 + 157 = 1,378 \equiv GT}$ (100% verified)
     - **Micro-Averaged Precision:** 80.70% ($1,221 / 1,513$)
     - **Micro-Averaged Recall:** 88.61% ($1,221 / 1,378$)
     - **Micro-Averaged F1:** 84.47%
     - **Macro-Averaged Recall:** 91.65% | **Macro-Averaged Precision:** 76.17%
4. **Empirically Proven Small-Target Degradation:** Size-stratified evaluation across all 1,378 test targets reveals that while Foreground/Large vessels ($area \ge 96^2$ px²) achieve **85.32% F1** and **89.64% Recall**, Medium vessels ($area < 96^2$ px²) drop to **72.54% F1** and **74.47% Recall** (and under maritime relative scaling $<0.5\%$ area, **65.80% F1** and **66.67% Recall**). This proves that global mAP obscures substantial degradation on small, distant vessels.
5. **Real-Time Edge Throughput:** Measured batch-1 inference latency on the RTX 3050 GPU is **24.86 ms total latency** (11.92 ms pure neural forward pass), delivering **40.23 FPS** and satisfying real-time (>30 FPS) requirements.
6. **Video Tracking:** Validated a temporal video inference pipeline (`outputs/predictions/demo_output.mp4`) providing visual trajectory association and real-time telemetry HUD overlay.

---

## 2. Task and Dataset Specification

### 2.1 Provenance and License
- **Origin:** State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing (LIESMARS), Wuhan University (Zhenfeng Shao et al., *IEEE Transactions on Multimedia*, 2018).
- **License / Terms of Use:** Non-commercial research and educational benchmark dataset.
- **Dataset Version:** SeaShips(7000), a curated 7,000-image subset extracted from 10,080 coastline surveillance video segments.

### 2.2 Annotation Format and Integrity Audit
- **Format:** Pascal VOC XML annotations containing `<size>` (width, height, depth) and `<object>` elements with `<name>` and `<bndbox>` (`xmin, ymin, xmax, ymax`).
- **Suitability for Object Detection:** Excellent. All annotations contain 2D axis-aligned bounding boxes with valid pixel coordinates.
- **Audit Findings:** Verified via `src/data_audit.py` across all 7,000 XML files:
  - Missing or unreadable image files: **0**
  - Malformed or unparseable XML annotations: **0**
  - Degenerate bounding boxes ($x_{min} \ge x_{max}$ or $y_{min} \ge y_{max}$): **0**
  - Out-of-bounds coordinates ($<0$ or $>1920/1080$): **0**
  - Image Dimensions: 100% uniform Full HD ($1920 \times 1080 \times 3$).

### 2.3 Verified Ground-Truth Class Distribution

| Class ID | Class Label | Train Split (4,900 imgs) | Val Split (1,050 imgs) | Test Split (1,050 imgs) | Total Instances (7,000 imgs) | Total Share (%) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 0 | **Ore Carrier** | 1,456 | 255 | 488 | 2,199 | 23.85% |
| 1 | **Bulk Cargo Carrier** | 1,158 | 374 | 420 | 1,952 | 21.17% |
| 2 | **General Cargo Ship** | 1,363 | 121 | 21 | 1,505 | 16.32% |
| 3 | **Container Ship** | 757 | 31 | 113 | 901 | 9.77% |
| 4 | **Fishing Boat** | 1,412 | 491 | 287 | 2,190 | 23.75% |
| 5 | **Passenger Ship** | 347 | 78 | 49 | 474 | 5.14% |
| **All** | **Total** | **6,493** | **1,350** | **1,378** | **9,221** | **100.00%** |

*Note on Prior Discrepancy:* In the initial draft of the report, the ground-truth column displayed counts of 31, 121, 78, 255, 491, and 374 (which sum to 1,350). The audit traced these exact counts to the **validation split**. The test split contains **1,378 instances** with the class breakdown verified in the table above.

### 2.4 Object-Size Distribution and the Operational Definition of Small Targets

Under COCO benchmark standards:
- Small: $area < 32^2 = 1,024\text{ px}^2$
- Medium: $32^2 \le area < 96^2 = 9,216\text{ px}^2$
- Large: $area \ge 96^2 = 9,216\text{ px}^2$

#### Dataset-Wide Scale Breakdown (9,221 Objects):
- **COCO Small:** 15 objects (0.16%)
- **COCO Medium:** 784 objects (8.50%)
- **COCO Large:** 8,422 objects (91.34%)

#### Operational Analysis of Frame Occupancy:
In a 1080p surveillance image (2,073,600 pixels), a vessel of $120 \times 40$ pixels has an area of 4,800 pixels² (COCO "Medium"), but occupies **only 0.23% of the image frame**. When downsampled to $640 \times 640$, its height shrinks to just **~13 pixels**. Therefore, in this report, we evaluate both:
1. Standard COCO pixel thresholds.
2. An operational maritime definition: **Distant Small** ($area < 10,368\text{ px}^2$, corresponding to $<0.5\%$ of a 1080p frame) versus **Near Large** ($\ge 0.5\%$).

---

## 3. Data Splitting & Leakage Analysis

### 3.1 Splitting Procedure
The dataset was partitioned using `src/prepare_dataset.py` with fixed random seed `seed=42`:
- **Training Set:** 4,900 images (70.0%), 6,493 vessel instances
- **Validation Set:** 1,050 images (15.0%), 1,350 vessel instances
- **Untouched Test Set:** 1,050 images (15.0%), 1,378 vessel instances
- Exact file paths are recorded in `data/splits/train.txt`, `val.txt`, and `test.txt`.

### 3.2 Audit of Temporal Leakage Claims
- **Methodology Evaluated:** Grouping consecutive 50-frame numbered files (e.g., 000001–000050, 000051–000100) into 140 sequence chunks and assigning chunks to partitions.
- **Empirical Check:** A pixel-difference analysis between consecutive frames showed that while frames 1–3 and 49–50 have low frame-to-frame difference ($\Delta \approx 3.4 - 5.4$), significant scene cuts ($\Delta > 20.0$) also occur within numbered chunks (44 scene cuts detected in the first 200 frames).
- **Auditor Assessment:** Grouping by 50-frame chunks groups temporally proximate frames and significantly reduces frame-to-frame leakage compared to naive random shuffling. However, because SeaShips does not provide official source video segment IDs or camera location metadata, it **cannot be guaranteed** that identical vessels, background shorelines, or camera viewpoints do not reappear across different chunks.
- **Correction:** The claim of "zero leakage" or an "airtight split" has been corrected to a **leakage-reduced chunk-based split**, explicitly noting the absence of official sequence metadata.

---

## 4. Methodology & Architectural Decisions

### 4.1 Detector Architecture: YOLOv8s
- **Architecture:** Ultralytics YOLOv8s (11.2M parameters, 28.6 GFLOPs at 640px).
- **Backbone & Neck:** Modified CSPDarknet53 with C2f cross-stage blocks and a PAN-FPN feature pyramid fusing strides 8, 16, and 32.
- **Head:** Anchor-free decoupled head with independent classification and regression branches.
- **Loss Formulation:** Complete Intersection over Union (CIoU) combined with Distribution Focal Loss (DFL) for box regression, and Binary Cross-Entropy (BCE) for multi-class classification.

### 4.2 Augmentation Policy: Physically Grounded Maritime Design
Augmentations were selected to reflect realistic maritime physics while eliminating unrealistic artifacts:

| Augmentation Parameter | Value | Maritime Physical Justification |
|---|---|---|
| **Horizontal Flip (`fliplr`)** | 0.5 | Ships realistically navigate both port and starboard directions. |
| **Vertical Flip (`flipud`)** | **0.0 (Disabled)** | Inverted hulls floating in the sky destroy gravity priors. |
| **Scale Jitter (`scale`)** | 0.5 (0.5× to 1.5×) | Simulates vessels at varying distances from camera. |
| **HSV Jitter (`hsv_s, hsv_v`)** | 0.4, 0.4 | Simulates sunlight glare, overcast days, dusk, and maritime haze. |
| **Mast Vibration (`degrees`)** | 2.0° | Simulates camera mast sway from wind without tilting sea horizon. |
| **Mosaic** | 1.0 | Forces multi-scale context recognition across compound scenes. |
| **Mixup** | **0.0 (Disabled)** | Prevents semi-transparent "ghost ships" on the water. |

---

## 5. Experimental Setup

- **Hardware Platform:** NVIDIA GeForce RTX 3050 Laptop GPU (Dedicated VRAM: 4,096 MB, WDDM Driver: 566.07, CUDA Version: 12.1).
- **Operating System:** Microsoft Windows 11 Home (64-bit).
- **Software Dependencies:** Python 3.11.9, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, Ultralytics 8.4.190, OpenCV 5.0.0.93.
- **Training Hyperparameters:**
  - Configuration File: `configs/baseline.yaml`
  - Pretrained Initialization: `yolov8s.pt` (COCO weights)
  - Epochs: 10
  - Batch Size: 16
  - Input Resolution: 640 × 640
  - Optimizer: AdamW (`lr0=0.001`, `momentum=0.9`, `weight_decay=0.0005`)
  - Warmup: 3.0 epochs (`warmup_bias_lr=0.1`, `warmup_momentum=0.8`)
  - Precision: FP16 Automatic Mixed Precision (AMP)
  - Dataloader Workers: 2
  - Peak GPU VRAM Usage: 3.55 GB / 4.00 GB
- **Evaluation Settings:**
  - Evaluated Checkpoint: `outputs/models/baseline_yolov8s_640_best.pt`
  - Primary Evaluation Split: `test` (1,050 images, 1,378 targets)
  - Default Operating Thresholds: Confidence = 0.25, NMS IoU = 0.60, Max Detections = 300.
  - Model Selection Rule: Checkpoint with highest validation mAP@0.5 was selected for test set evaluation. Test data was never used for model selection or tuning.

---

## 6. Quantitative Results & Reconciled Metrics

### 6.1 Untouched Test Set Performance (Baseline @ 640px)

Source artifact: `outputs/results/reconciled_evaluation_audit.json` (re-evaluated on 1,050 test images, 1,378 ground-truth targets).

To ensure complete scientific rigor and eliminate ambiguity, results are reported under two distinct, mathematically precise evaluation frameworks:
1. **Continuous PR-Curve Benchmark Integration:** Evaluated using Ultralytics standard validation protocol (`model.val()`). Metrics represent the integrated area under the curve (mAP) and the operating point where the mean F1-curve achieves its global peak ($\text{conf} \approx 0.68$).
2. **Fixed-Threshold Operational Discrete Matching:** Evaluated at the operational deployment threshold (`conf=0.25`, matching $\text{IoU} \ge 0.5$). Every predicted box is explicitly matched to an unassigned ground-truth target of the same class in descending confidence order.

#### Summary of Test Set Performance Metrics:

| Metric Category | Metric Name | Value | Exact Calculation Formula / Protocol |
|---|---|:---:|---|
| **PR-Curve Benchmark** | **mAP@0.5** | **90.21%** | Mean Average Precision at IoU threshold 0.50 |
| **PR-Curve Benchmark** | **mAP@0.5:0.95** | **63.72%** | Mean Average Precision averaged across 10 IoU thresholds (0.50 to 0.95, step 0.05) |
| **PR-Curve Benchmark** | **Curve-Optimal Macro Precision** | **93.55%** | Unweighted arithmetic mean of class precisions at max-F1 threshold ($\text{conf} \approx 0.68$) |
| **PR-Curve Benchmark** | **Curve-Optimal Macro Recall** | **84.92%** | Unweighted arithmetic mean of class recalls: $\frac{1}{6} \sum_{i=1}^{6} r_i$ |
| **PR-Curve Benchmark** | **Curve-Optimal Weighted Recall** | **80.48%** | Instance-weighted class recall: $\frac{\sum_{i=1}^{6} nt_i \cdot r_i}{\sum nt_i} = \frac{1,109.07}{1,378}$ |
| **PR-Curve Benchmark** | **Harmonic Aggregate F1** | **89.03%** | $2 \cdot \frac{P_{macro} \cdot R_{macro}}{P_{macro} + R_{macro}} = 2 \cdot \frac{0.9355 \cdot 0.8492}{0.9355 + 0.8492}$ |
| **PR-Curve Benchmark** | **Macro-Averaged F1** | **88.67%** | Arithmetic mean of individual class F1 scores: $\frac{1}{6}\sum_{i=1}^{6} F_{1,i}$ |
| **Operational Discrete** | **Total Ground Truth (GT)** | **1,378** | Complete instance population in test split |
| **Operational Discrete** | **True Positives (TP)** | **1,221** | Detections matched to GT at $\text{conf} \ge 0.25, \text{IoU} \ge 0.50$ |
| **Operational Discrete** | **False Positives (FP)** | **292** | Detections without matching GT at $\text{conf} \ge 0.25$ |
| **Operational Discrete** | **False Negatives (FN)** | **157** | Ground truth targets un-detected at $\text{conf} \ge 0.25$ |
| **Operational Discrete** | **Conservation Check** | **1,378** | $\mathbf{TP + FN \equiv 1,221 + 157 = 1,378 \equiv GT}$ (Zero discrepancy) |
| **Operational Discrete** | **Micro-Averaged Precision** | **80.70%** | $\frac{TP}{TP + FP} = \frac{1,221}{1,221 + 292} = \frac{1,221}{1,513}$ |
| **Operational Discrete** | **Micro-Averaged Recall** | **88.61%** | $\frac{TP}{TP + FN} = \frac{1,221}{1,221 + 157} = \frac{1,221}{1,378}$ |
| **Operational Discrete** | **Micro-Averaged F1** | **84.47%** | $2 \cdot \frac{0.8070 \cdot 0.8861}{0.8070 + 0.8861}$ |
| **Operational Discrete** | **Macro-Averaged Precision** | **76.17%** | $\frac{1}{6} \sum_{i=1}^{6} P_{discrete,i}$ |
| **Operational Discrete** | **Macro-Averaged Recall** | **91.65%** | $\frac{1}{6} \sum_{i=1}^{6} R_{discrete,i}$ |
| **Operational Discrete** | **Macro-Averaged F1** | **80.87%** | $\frac{1}{6} \sum_{i=1}^{6} F_{1,discrete,i}$ |

---

### 6.2 Per-Class Breakdown (Test Split: 1,050 images, 1,378 targets)

The table below reconciles both evaluation methodologies side-by-side:

| Class Label | GT Count | Discrete TP | Discrete FP | Discrete FN | TP + FN | Discrete Precision | Discrete Recall | Discrete F1 | Curve-Opt Prec | Curve-Opt Rec | AP@0.5 | AP@0.5:0.95 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ore Carrier** | 488 | 451 | 62 | 37 | **488** | 87.91% | 92.42% | 90.11% | 95.55% | 83.62% | 90.48% | 51.57% |
| **Bulk Cargo Carrier** | 420 | 330 | 70 | 90 | **420** | 82.50% | 78.57% | 80.49% | 94.58% | 70.00% | 77.15% | 55.96% |
| **General Cargo Ship** | 21 | 21 | 42 | 0 | **21** | 33.33% | **100.00%** | 50.00% | 91.10% | 97.56% | 97.77% | 72.36% |
| **Container Ship** | 113 | 109 | 3 | 4 | **113** | 97.32% | 96.46% | 96.89% | 99.74% | 96.46% | 96.50% | 72.52% |
| **Fishing Boat** | 287 | 266 | 107 | 21 | **287** | 71.31% | 92.68% | 80.61% | 82.87% | 83.28% | 87.23% | 57.82% |
| **Passenger Ship** | 49 | 44 | 8 | 5 | **49** | 84.62% | 89.80% | 87.13% | 97.47% | 78.59% | 92.15% | 72.08% |
| **Total / Summary** | **1,378** | **1,221** | **292** | **157** | **1,378** | **80.70% (Micro)** / **76.17% (Macro)** | **88.61% (Micro)** / **91.65% (Macro)** | **84.47% (Micro)** / **80.87% (Macro)** | **93.55% (Macro)** | **84.92% (Macro)** / **80.48% (Wt)** | **90.21%** | **63.72%** |

#### Reconciliation of Class-Level Discrepancies:
- **General Cargo Ship (GT=21):** In discrete evaluation at `conf=0.25`, exactly 21 of 21 ground-truth instances are detected ($\text{TP}=21, \text{FN}=0$), yielding a true discrete recall of **100.00%**. The previously reported **97.56% recall** is traced to Ultralytics' continuous PR-curve interpolation (`box.r[2] = np.interp(...)` across 1,000 threshold bins at $\text{conf} \approx 0.68$), where one marginal target fell below the high peak-F1 threshold.
- **Bulk Cargo Carrier vs. Fishing Boat False Positives:** `fishing boat` accounts for 107 false positives and `bulk cargo carrier` accounts for 70 false positives. Error auditing reveals that small harbor craft, floating buoys, and shoreline container cranes frequently trigger low-confidence detections in these classes at `conf=0.25`.

---

### 6.3 Size-Stratified Evaluation Across Full Test Population (1,378 Targets)

Evaluated via `scripts/audit_metrics.py` over all 1,050 test images with ground-truth-anchored scale attribution:

#### A. Standard COCO Area Thresholds

| Scale Category | Definition | GT Count | TP | FP | FN | TP + FN | Precision (%) | Recall (%) | F1-Score (%) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Small** | $area < 32^2$ (<1,024 px²) | 0 | 0 | 0 | 0 | **0** | N/A* | N/A* | N/A* |
| **Medium** | $32^2 \le area < 96^2$ (1,024–9,216 px²) | 94 | 70 | 29 | 24 | **94** | **70.71%** | **74.47%** | **72.54%** |
| **Large** | $area \ge 96^2$ ($\ge$9,216 px²) | 1,284 | 1,151 | 263 | 133 | **1,284** | **81.40%** | **89.64%** | **85.32%** |
| **All Scales** | **Full Population** | **1,378** | **1,221** | **292** | **157** | **1,378** | **80.70%** | **88.61%** | **84.47%** |

*\*Note: No ground-truth targets in the test split are under $32^2$ pixels; therefore, standard COCO Small metrics cannot be computed on this split.*

#### B. Operational Maritime Relative Area Thresholds

| Scale Category | Frame Occupancy | GT Count | TP | FP | FN | TP + FN | Precision (%) | Recall (%) | F1-Score (%) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Distant Small** | $area < 0.5\%$ frame (<10,368 px²) | 114 | 76 | 41 | 38 | **114** | **64.96%** | **66.67%** | **65.80%** |
| **Near Large** | $area \ge 0.5\%$ frame ($\ge$10,368 px²) | 1,264 | 1,145 | 251 | 119 | **1,264** | **82.02%** | **90.59%** | **86.09%** |
| **All Scales** | **Full Population** | **1,378** | **1,221** | **292** | **157** | **1,378** | **80.70%** | **88.61%** | **84.47%** |

#### Resolution of Scale Conservation Discrepancy:
In prior report drafts, $TP + FN$ did not equal $GT$ in the size tables (e.g. Medium showed $58+24=82 \ne 94$, Large showed $1,163+133=1,296 \ne 1,284$). The audit identified the exact software defect in `src/evaluate.py`: true positives were binned by the *predicted* bounding box area (`categorize_scale(p["area"])`), whereas ground truth and false negatives were binned by the *annotated ground truth* area (`categorize_scale(gt["area"])`). Whenever a matched prediction's area crossed the 9,216 px² threshold relative to the ground truth, the count drifted across columns. By anchoring true positive scale assignment strictly to the ground-truth object's area, $\mathbf{TP + FN \equiv GT}$ holds with 100% mathematical precision across all categories.

#### Key Takeaway on Small Target Degradation:
The reconciled full-population audit confirms that **overall mAP@0.5 (90.21%) obscures severe small-object degradation**. Distant vessels ($<0.5\%$ frame area) suffer a **23.9 percentage point drop in recall** (66.67% vs 90.59%) and a **20.3 percentage point drop in F1** (65.80% vs 86.09%) relative to near-shore vessels.

---

### 6.4 Empirical Operating Point Curve (Confidence Threshold Trade-Off)

To determine the optimal operating threshold for maritime deployment, the detector was evaluated across a fine-grained confidence sweep from $\text{conf}=0.10$ to $\text{conf}=0.70$ on all 1,050 test images:

| Operating Threshold | Ground Truth | True Positives (TP) | False Positives (FP) | False Negatives (FN) | TP + FN Check | Precision (%) | Recall (%) | F1-Score (%) | Operational Assessment |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **conf = 0.10** | 1,378 | 1,257 | 460 | 121 | **1,378** | 73.21% | **91.22%** | 81.23% | Maximum sensitivity; high false alarms from waves |
| **conf = 0.15** | 1,378 | 1,246 | 389 | 132 | **1,378** | 76.21% | **90.42%** | 82.71% | High-security coastal perimeter alert; tracks faint distant hulls |
| **conf = 0.20** | 1,378 | 1,236 | 340 | 142 | **1,378** | 78.43% | 89.70% | 83.68% | Balanced high-recall surveillance mode |
| **conf = 0.25 (Default)** | 1,378 | 1,221 | 292 | 157 | **1,378** | **80.70%** | **88.61%** | **84.47%** | **Standard operational baseline (optimal edge compromise)** |
| **conf = 0.30** | 1,378 | 1,214 | 259 | 164 | **1,378** | 82.42% | 88.10% | 85.16% | Moderate false-alarm suppression |
| **conf = 0.35** | 1,378 | 1,200 | 227 | 178 | **1,378** | 84.09% | 87.08% | 85.56% | Harbor navigation corridor monitoring |
| **conf = 0.40** | 1,378 | 1,188 | 195 | 190 | **1,378** | 85.90% | 86.21% | 86.06% | High precision; misses marginal distant hulls |
| **conf = 0.45** | 1,378 | 1,171 | 163 | 207 | **1,378** | 87.78% | 84.98% | 86.36% | Low false-alarm rate; automated alerting |
| **conf = 0.50** | 1,378 | 1,159 | 141 | 219 | **1,378** | **89.15%** | **84.11%** | **86.56%** | **Peak discrete F1 operating point**; suppresses sun glare |
| **conf = 0.60** | 1,378 | 1,108 | 97 | 270 | **1,378** | 91.95% | 80.41% | 85.79% | High-certainty tracking; drops distant craft |
| **conf = 0.70** | 1,378 | 965 | 57 | 413 | **1,378** | **94.42%** | **70.03%** | 80.42% | Extreme certainty; misses 30% of targets |

#### Resolution of the Constant Precision/Recall Paradox:
In the previous report draft, precision (93.55%) and recall (84.92%) were mistakenly reported as identical across `conf=0.15` and `conf=0.25`, and recall was unchanged at `conf=0.50`. The audit determined that the previous script extracted `val_results.box.mp` and `val_results.box.mr` from Ultralytics `model.val()`. In Ultralytics (`ultralytics.utils.metrics.ap_per_class`), `box.mp` and `box.mr` are *not* evaluated at the threshold passed via the `conf` parameter; rather, they are evaluated at the global maximum of the PR-curve harmonic mean (`i = smooth(f1_curve.mean(0), 0.1).argmax()`, which occurs at $\text{conf} \approx 0.68$). Because 0.15, 0.25, and 0.50 are all below 0.68, pre-filtering detections below 0.68 had zero effect on the curve peak, returning identical numbers. 

As shown in the verified discrete table above, true operational precision and recall vary monotonically as expected: lowering confidence from 0.50 to 0.15 increases discrete recall from **84.11% to 90.42%** (+6.31 percentage points) while precision adjusts from **89.15% to 76.21%** (-12.94 percentage points).

---

## 7. Error Analysis

Ground-truth-verified failure case images from `outputs/predictions/errors/`:

### 7.1 Quantitative Error Breakdown
Across 1,050 test images (1,378 ground-truth targets at default operational `conf=0.25`, matching $\text{IoU} \ge 0.5$):
- **True Positives (TP):** 1,221 detections correctly matched to ground truth.
- **False Negatives (FN):** 157 missed vessels.
- **False Positives (FP):** 292 false alarms.
- **Total Ground Truth Conservation:** $\mathbf{TP + FN \equiv 1,221 + 157 = 1,378 \equiv GT}$ (Zero discrepancy).
- **Total Detections Generated:** $\mathbf{TP + FP = 1,221 + 292 = 1,513}$ detections.
- **Operational Precision:** $\frac{1,221}{1,513} = \mathbf{80.70\%}$
- **Operational Recall:** $\frac{1,221}{1,378} = \mathbf{88.61\%}$
- **Operational F1:** $\mathbf{84.47\%}$

### 7.2 Failure Mode Patterns and Case Studies
1. **Shoreline Infrastructure Confusion (False Positives):**
   - *Example:* `error_000163_fpTrue_fnFalse.jpg`
   - *Description:* Fixed harbor container cranes and industrial wharf structures are detected as cargo ships.
   - *Root Cause:* High visual similarity between shore gantry cranes and ship-mounted cargo derricks.
   - *Proposed Fix:* Introduce negative background tiles of port facilities or apply a static land/water geographic mask.
2. **Distant Low-Contrast Vessels in Sea Haze (False Negatives):**
   - *Example:* `error_000355_fpFalse_fnTrue.jpg`, `error_000358_fpFalse_fnTrue.jpg`
   - *Description:* Distant fishing boats near the horizon on overcast days are completely missed.
   - *Root Cause:* Low chromatic contrast against hazy skies combined with $640\times640$ downsampling loss.
   - *Proposed Fix:* Local horizon slicing (SAHI) or adaptive contrast enhancement (CLAHE).
3. **Class Ambiguity (Bulk Cargo vs. Ore Carrier):**
   - *Example:* `error_000167_fpTrue_fnTrue.jpg`
   - *Description:* Model predicts `bulk cargo carrier` where ground truth is `ore carrier`.
   - *Root Cause:* Both classes share similar low-profile hull lines and deck layouts from perpendicular viewing angles.

---

## 8. Runtime Benchmarking & Video Demo

### 8.1 Measured Single-Frame Latency Breakdown
Measured on **NVIDIA GeForce RTX 3050 Laptop GPU** (batch size = 1, input = 640×640, 50 timed iterations, 15 warmup iterations):

| Pipeline Component | Measured Time (ms) | Fraction of Pipeline (%) |
|---|:---:|:---:|
| **Preprocessing** (Letterbox resize, BGR-to-RGB, normalization) | 2.04 ms | 8.21% |
| **Model Forward Pass** (PyTorch FP16 inference on GPU) | **11.92 ms** | **47.95%** |
| **Postprocessing** (Non-Maximum Suppression, coordinate mapping) | 2.14 ms | 8.61% |
| **Telemetry HUD & Drawing** | 8.76 ms | 35.23% |
| **Total End-to-End Latency** | **24.86 ms** | **100.00%** |
| **Measured Real-Time Throughput** | **40.23 FPS** | — |

*Calculation Check:* $\text{FPS} = \frac{1000\text{ ms}}{24.86\text{ ms}} = 40.225 \approx 40.23\text{ FPS}$.

### 8.2 Architectural Latency Projections (Untested)
> [!NOTE]
> The following optimizations represent technical projections based on published benchmarks, not local hardware measurements:
> - **TensorRT FP16 Quantization:** Projected to reduce neural forward pass from 11.9 ms to ~4.5 ms, enabling theoretical frame rates $>100$ FPS.
> - **NVIDIA Jetson Orin Nano:** Projected to achieve 25–35 FPS in INT8 mode under a 15W power envelope.

### 8.3 Video Demonstration and Tracking
- **Video Artifact:** `outputs/predictions/demo_output.mp4`
- **Specification:** 90 frames, 1920×1080 resolution, encoded via OpenCV `mp4v` codec.
- **Tracking Method:** Spatial-temporal IoU tracker (`SimpleMaritimeTracker`) maintaining continuous track IDs and bounding box smoothing.
- **Evaluation Status:** **Qualitative observation only.** Because SeaShips annotations do not contain multi-object tracking ground-truth trajectories, formal MOT metrics (MOTA, IDF1, HOTA) cannot be quantitatively assessed on this dataset. Visual inspection of `demo_output.mp4` confirms smooth ID persistence across the 90-frame transit.

---

## 9. Current Limitations & Future Work

### 9.1 Verified Current Limitations
1. **Absence of Night/Thermal Data:** The dataset is composed entirely of daylight optical RGB imagery. Night-time or severe fog performance remains unverified.
2. **Small-Target Degradation:** The 25% recall deficit on distant vessels ($<0.5\%$ frame area) remains the primary operational limitation under 640px input resolution.
3. **No Verified Ground-Truth Tracking:** SeaShips lacks temporal identity labels, limiting tracking evaluation to qualitative inspection.

### 9.2 Planned Future Experiments
1. **High-Resolution Horizon Slicing (SAHI):** Evaluating sliced inference on the upper 35% horizon strip at native 1080p resolution to recover small-target recall without processing full-frame empty sea.
2. **Adversarial Port Background Training:** Collecting negative background tiles of harbors and wharves to suppress gantry crane false positives.
3. **TensorRT Compilation:** Building and benchmarking an INT8/FP16 TensorRT engine on target edge hardware.

---

## 10. Reproducibility Appendix

### 10.1 Key Artifact Paths
- **Trained Checkpoint:** `outputs/models/baseline_yolov8s_640_best.pt`
- **Split Manifests:** `data/splits/train.txt`, `data/splits/val.txt`, `data/splits/test.txt`
- **Dataset YAML:** `configs/dataset.yaml`
- **Training Config:** `configs/baseline.yaml`
- **Raw Evaluation Artifacts:**
  - `outputs/results/baseline_test_eval_evaluation_report.json`
  - `outputs/results/verified_test_metrics.json`
  - `outputs/results/dataset_audit_summary.json`
- **Generated Figures:** `outputs/figures/results.png`, `confusion_matrix.png`, `BoxPR_curve.png`, `class_distribution.png`
- **Demonstration Video:** `outputs/predictions/demo_output.mp4`
- **Visual Failure Cases:** `outputs/predictions/errors/*.jpg`

### 10.2 Exact Execution Commands

```powershell
# 1. Environment Activation
.\.venv\Scripts\Activate.ps1

# 2. Run Test Suite
.\.venv\Scripts\python.exe -m pytest -v tests/

# 3. Dataset Audit & Figure Generation
.\.venv\Scripts\python.exe src/data_audit.py data/raw

# 4. Generate Splits & YOLO Labels
.\.venv\Scripts\python.exe src/prepare_dataset.py

# 5. Model Training (10 Epochs, GPU)
.\.venv\Scripts\python.exe src/train.py --config configs/baseline.yaml --epochs 10 --batch 16 --imgsz 640 --name baseline_yolov8s_640

# 6. Model Evaluation on Test Split
.\.venv\Scripts\python.exe src/evaluate.py --model outputs/models/baseline_yolov8s_640_best.pt --split test --imgsz 640 --name baseline_test_eval

# 7. Audit & Size Verification Script
.\.venv\Scripts\python.exe scripts/audit_metrics.py

# 8. Video Inference Demo
.\.venv\Scripts\python.exe src/predict_video.py --model outputs/models/baseline_yolov8s_640_best.pt --track --output outputs/predictions/demo_output.mp4
```

