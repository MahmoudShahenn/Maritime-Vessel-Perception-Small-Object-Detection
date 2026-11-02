# SeaShips Dataset Audit & Data Analysis Report

## 1. Executive Summary
- **Dataset Title:** SeaShips Benchmark Dataset (7,000 Curated Coastline Surveillance Subset)
- **Primary Domain:** Maritime Object Detection & Coastline Surveillance
- **Image Resolution:** Uniform 1920 × 1080 RGB (Full HD) across 7000 images.
- **Total Annotated Images:** 7,000
- **Total Ship Objects:** 9,221
- **Mean Density:** 1.32 objects/image (Max: 5)
- **Annotation Format:** Pascal VOC XML (`<bndbox>` with `xmin, ymin, xmax, ymax`)
- **Corrupt / Empty Files:** Missing Images: 0, Empty XMLs: 0, Invalid BBoxes: 0

---

## 2. Vessel Category Distribution & Class Imbalance

| Vessel Class | Object Count | Dataset Share (%) | Small (<32²) | Medium (32²-96²) | Large (>96²) |
|---|---|---|---|---|---|
| **Bulk cargo carrier** | 1,952 | 21.2% | 0 (0.0%) | 44 (2.3%) | 1,908 (97.7%) |
| **Container ship** | 901 | 9.8% | 0 (0.0%) | 28 (3.1%) | 873 (96.9%) |
| **Fishing boat** | 2,190 | 23.8% | 11 (0.5%) | 611 (27.9%) | 1,568 (71.6%) |
| **General cargo ship** | 1,505 | 16.3% | 0 (0.0%) | 12 (0.8%) | 1,493 (99.2%) |
| **Ore carrier** | 2,199 | 23.8% | 0 (0.0%) | 39 (1.8%) | 2,160 (98.2%) |
| **Passenger ship** | 474 | 5.1% | 4 (0.8%) | 50 (10.5%) | 420 (88.6%) |
| **Total** | **9,221** | **100.0%** | **15** | **784** | **8,422** |

### Observations:
1. **Class Balance:** The dataset covers 6 canonical maritime classes. The distribution has moderate imbalance: `bulk cargo carrier` and `ore carrier` are frequent, while `passenger ship` and `fishing boat` have distinctive geometric and scale footprints.
2. **Aspect Ratios:** Ships in maritime surveillance views (coastal cameras facing the horizon or harbor channels) exhibit high aspect ratios (width >> height), typically between 2.0 and 6.0, as ships are elongated along the waterline.

---

## 3. Scale & Small-Object Distribution Analysis

The central operational challenge of maritime surveillance is detecting small, distant, and low-contrast targets against dynamic sea backgrounds.

### Bounding Box Size Statistics (Pixels):
- **Minimum Area:** 400.0 px²
- **10th Percentile (P10):** 10858.0 px²
- **Median Area:** 75570.0 px²
- **Mean Area:** 121306.5 px²
- **90th Percentile (P90):** 296217.0 px²
- **Maximum Area:** 813656.0 px²

### Normalized Screen Area (% of 1920×1080 = 2,073,600 px):
- **P10 Relative Area:** 0.524% of total image area
- **Median Relative Area:** 3.64% of total image area
- **Mean Relative Area:** 5.85% of total image area

### COCO Scale Standard vs Maritime Operational Context:
- Standard COCO defines:
  - **Small:** area < $32^2 = 1,024$ px²
  - **Medium:** $32^2 \le$ area < $96^2 = 9,216$ px²
  - **Large:** area $\ge 96^2 = 9,216$ px²
- On 1080p imagery, an object of $100 \times 30$ px ($3,000$ px²) covers only **0.14%** of the frame, yet is categorized as "medium" under COCO definitions!
- Under standard input scaling ($640 \times 640$), an object with height 30px at 1080p gets downsampled to **~10 pixels**, which is easily lost in deep convolutional feature maps with stride 32.
- **Engineering Implication:** 
  1. We must evaluate higher resolutions (e.g., 800px / 1024px) or feature architectures that retain high-resolution shallow features (such as P2 / stride-4 feature maps or multi-scale FPN).
  2. Sliced inference (SAHI / tiling) or high-resolution test-time inference is particularly advantageous for distant ships.

---

## 4. Sequence Continuity & Data Leakage Prevention Strategy

### Sequence Analysis:
- The 7,000 images originate from coastline surveillance video streams where cameras record moving ships at fixed positions.
- Sequentially numbered filenames (e.g., `000001.jpg` through `000050.jpg`) represent temporally adjacent frames of the same vessel passage.
- **Danger of Naive Random Splitting:** A standard random split (e.g., `train_test_split(shuffle=True)`) will place frame $t$ in train and frame $t+1$ in test. This produces severe data leakage: the model memorizes the exact vessel appearance, background coastline, wave pattern, and lighting, yielding artificially inflated test mAP that catastrophically collapses in real-world deployment.

### Splitting Protocol:
1. **Official Pascal VOC ImageSets Partition:**
   - Train: 1,750 images (25%)
   - Val: 1,750 images (25%)
   - Trainval: 3,500 images (50%)
   - Test: 3,500 images (50%)
   The official partition splits the data cleanly into disjoint blocks/sequences.
2. **Assignment Target Split (~70% Train, ~15% Val, ~15% Test):**
   - We construct sequential video-block partitions using a chunk-based sequence grouping (grouping contiguous blocks of ~50-100 consecutive frames belonging to the same transit sequence).
   - This ensures complete video-level isolation: **no sequence in the test set appears in the training or validation sets**.
   - Fixed random seed (`seed=42`) is enforced throughout for deterministic reproduction.

---

## 5. Artifact Audit Verification

| Verification Item | Result | Status |
|---|---|---|
| Total Images Accessible | 7000 / 7000 | PASS |
| Bounding Boxes within [0, W] and [0, H] | 100% Valid | PASS |
| Class Label Consistency | 6 Canonical Classes (No Typos) | PASS |
| Multi-Class Co-occurrence | Verified (Cargo + Fishing / Mixed) | PASS |
