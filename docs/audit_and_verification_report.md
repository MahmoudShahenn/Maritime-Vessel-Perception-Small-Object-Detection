# Evidence-Based Audit & Verification Report (CVOD01)
*Post-Audit Numerical & Mathematical Reconciliation Edition*

**Role:** Senior Computer Vision Engineer & ML Experiment Auditor  
**Scope:** Verification and mathematical reconciliation of all data, code, configurations, logs, checkpoints, metrics, and claims for the SeaShips Maritime Vessel Perception Project.  
**Audit Date:** October 2026  
**Primary Source Artifact:** `outputs/results/reconciled_evaluation_audit.json`

---

## 1. Audit Matrix of Mathematical Discrepancies and Root-Cause Resolutions

| Audit Finding / Discrepancy | Previous Unreconciled State | Reconciled Verified State | Exact Root Cause & Software Location | Verification Status |
|---|---|---|---|:---:|
| **1. Overall Recall Mismatch** | Reported Recall = 84.92%, but reported counts: GT=1,378, TP=1,221, FN=157 ($1,221 / 1,378 = 88.61\%$) | **Discrete Point-in-Time Recall = 88.61%**<br>**Curve-Optimal Macro Recall = 84.92%** | Conflation of two distinct evaluation protocols: 84.92% is Ultralytics `box.mr` (unweighted macro-average recall at the PR curve-optimal max-F1 point $\text{conf} \approx 0.68$). 88.61% is the discrete operational matching recall at fixed threshold $\text{conf}=0.25, \text{IoU} \ge 0.5$. | **Verified & Reconciled** |
| **2. COCO Size Table Conservation** | Medium: TP=58, FN=24, GT=94 ($TP+FN=82 \ne 94$)<br>Large: TP=1,163, FN=133, GT=1,284 ($TP+FN=1,296 \ne 1,284$) | **Medium:** TP=70, FN=24, **GT=94** ($TP+FN \equiv 94$)<br>**Large:** TP=1,151, FN=133, **GT=1,284** ($TP+FN \equiv 1,284$) | In `src/evaluate.py` (lines 193-197), TP was binned by *predicted* box area (`p["area"]`), while GT and FN were binned by *ground-truth* area (`gt["area"]`). Predictions crossing the 9,216 px² boundary drifted categories. Fixed by attributing matched TP strictly to ground-truth object area. | **Verified & Reconciled** |
| **3. Maritime Relative Size Conservation** | Distant Small: TP=71, FN=38, GT=114 ($TP+FN=109 \ne 114$)<br>Near Large: TP=1,150, FN=119, GT=1,264 ($TP+FN=1,269 \ne 1,264$) | **Distant Small:** TP=76, FN=38, **GT=114** ($TP+FN \equiv 114$)<br>**Near Large:** TP=1,145, FN=119, **GT=1,264** ($TP+FN \equiv 1,264$) | Same scale attribution defect as COCO table. Anchoring matched TP to ground-truth area enforces $\mathbf{TP + FN \equiv GT}$ with 100% mathematical precision. | **Verified & Reconciled** |
| **4. General Cargo Ship Recall** | Reported 21 GT instances and 97.56% recall (mathematically impossible from an integer TP divided by 21) | **Discrete Matching:** TP=21, FN=0, **Recall = 100.00%** (21/21)<br>**PR-Curve Optimal:** Interpolated Recall = 97.56% | 97.56% is the continuous interpolated recall from Ultralytics PR-curve (`box.r[2]` at conf $\approx 0.68$). In operational discrete matching at $\text{conf}=0.25$, all 21 targets are detected with 0 false negatives. | **Verified & Reconciled** |
| **5. Weighted vs. Macro Recall Mismatch** | Weighted sum of per-class recalls is $\approx 80.48\%$, but reported overall recall is 84.92% | **Instance-Weighted Recall:** 80.48% ($\sum nt_i \cdot r_i / \sum nt_i$)<br>**Unweighted Macro Recall:** 84.92% ($\frac{1}{6} \sum r_i$) | Ultralytics `box.mr` is unweighted macro recall. The prior report labeled it "Aggregate TP / (TP + FN)" which mathematically defines instance-weighted (micro) recall. Both formulas and values are now explicitly defined and separated. | **Verified & Reconciled** |
| **6. Flat Precision/Recall Across Thresholds** | Precision (93.55%) and Recall (84.92%) identical across $\text{conf}=0.15$ and $\text{conf}=0.25$, and recall unchanged at $0.50$ | **conf=0.15:** P=76.21%, R=90.42%, F1=82.71%<br>**conf=0.25:** P=80.70%, R=88.61%, F1=84.47%<br>**conf=0.50:** P=89.15%, R=84.11%, F1=86.56% | Ultralytics `model.val()` extracts `box.mp` and `box.mr` from `ap_per_class` at the global maximum of the F1-curve (`smooth(f1_curve.mean(0), 0.1).argmax()`, conf $\approx 0.68$). Thresholds below 0.68 do not change the curve peak. The report now provides the true discrete matching curve across all operating thresholds. | **Verified & Reconciled** |

---

## 2. Reconciled Ground-Truth and Scale Inventory (Test Set: 1,050 Images)

### 2.1 Class Distribution
- **Ore Carrier:** 488 targets (35.41%)
- **Bulk Cargo Carrier:** 420 targets (30.48%)
- **Fishing Boat:** 287 targets (20.83%)
- **Container Ship:** 113 targets (8.20%)
- **Passenger Ship:** 49 targets (3.56%)
- **General Cargo Ship:** 21 targets (1.52%)
- **Total Test Ground Truth:** **1,378 instances** (Sum: $488 + 420 + 287 + 113 + 49 + 21 = 1,378$)

### 2.2 Scale Distribution & Exact Conservation Check
- **COCO Small ($<32^2$ px²):** 0 GT, 0 TP, 0 FP, 0 FN, **TP + FN = 0**
- **COCO Medium ($32^2 \le area < 96^2$ px²):** 94 GT, 70 TP, 29 FP, 24 FN, **TP + FN = 94** (P=70.71%, R=74.47%, F1=72.54%)
- **COCO Large ($area \ge 96^2$ px²):** 1,284 GT, 1,151 TP, 263 FP, 133 FN, **TP + FN = 1,284** (P=81.40%, R=89.64%, F1=85.32%)
- **Maritime Distant Small ($<0.5\%$ frame area):** 114 GT, 76 TP, 41 FP, 38 FN, **TP + FN = 114** (P=64.96%, R=66.67%, F1=65.80%)
- **Maritime Near Large ($\ge 0.5\%$ frame area):** 1,264 GT, 1,145 TP, 251 FP, 119 FN, **TP + FN = 1,264** (P=82.02%, R=90.59%, F1=86.09%)
- **Overall Total Conservation:** GT = 1,378, TP = 1,221, FP = 292, FN = 157, $\mathbf{TP + FN \equiv 1,378}$.

---

## 3. Metrics Consistency Matrix

| Metric Identifier | Exact Source File | Source JSON Key Path | Verified Measured Value |
|---|---|---|:---:|
| Test mAP@0.5 | `reconciled_evaluation_audit.json` | `ultralytics_pr_curve_evaluation.map50` | 0.9021 (90.21%) |
| Test mAP@0.5:0.95 | `reconciled_evaluation_audit.json` | `ultralytics_pr_curve_evaluation.map50_95` | 0.6372 (63.72%) |
| Curve-Optimal Macro Precision | `reconciled_evaluation_audit.json` | `ultralytics_pr_curve_evaluation.macro_precision_mp` | 0.9355 (93.55%) |
| Curve-Optimal Macro Recall | `reconciled_evaluation_audit.json` | `ultralytics_pr_curve_evaluation.macro_recall_mr` | 0.8492 (84.92%) |
| Curve-Optimal Weighted Recall | `reconciled_evaluation_audit.json` | `ultralytics_pr_curve_evaluation.weighted_recall` | 0.8048 (80.48%) |
| Discrete Micro Precision (`conf=0.25`) | `reconciled_evaluation_audit.json` | `discrete_matching_evaluation.overall.micro_precision` | 0.8070 (80.70%) |
| Discrete Micro Recall (`conf=0.25`) | `reconciled_evaluation_audit.json` | `discrete_matching_evaluation.overall.micro_recall` | 0.8861 (88.61%) |
| Discrete Micro F1 (`conf=0.25`) | `reconciled_evaluation_audit.json` | `discrete_matching_evaluation.overall.micro_f1` | 0.8447 (84.47%) |
| Discrete Macro Precision (`conf=0.25`) | `reconciled_evaluation_audit.json` | `discrete_matching_evaluation.overall.macro_precision` | 0.7617 (76.17%) |
| Discrete Macro Recall (`conf=0.25`) | `reconciled_evaluation_audit.json` | `discrete_matching_evaluation.overall.macro_recall` | 0.9165 (91.65%) |
| Discrete Macro F1 (`conf=0.25`) | `reconciled_evaluation_audit.json` | `discrete_matching_evaluation.overall.macro_f1` | 0.8087 (80.87%) |
| Measured Real-Time FPS | `baseline_test_eval_evaluation_report.json` | `latency_profile.fps` | 40.23 FPS (RTX 3050 Laptop GPU) |
| Total Frame Latency | `baseline_test_eval_evaluation_report.json` | `latency_profile.total_latency_ms` | 24.86 ms |

---

## 4. Auditor Conclusion and Sign-off

The numerical audit has completed with **100% mathematical consistency**. All six discrepancies raised during report review have been traced to their exact source-code root causes, corrected in both code and data artifacts, and verified against physical evaluation files on disk. Conservation of ground truth ($\mathbf{TP + FN \equiv GT}$) holds with zero discrepancy across all classes, scales, and confidence operating points.
