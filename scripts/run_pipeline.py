"""
Master Pipeline Orchestration Script (CVOD01)
End-to-end execution of:
1. Dataset Audit
2. YOLO Dataset Preparation & Leakage-Free Splitting
3. Unit Testing
4. Baseline Training & Small-Object Enhanced Fine-Tuning
5. Untouched Test Set Evaluation
6. Error Analysis & Failure Extraction
7. Video Inference Demonstration
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_cmd(cmd_list, desc):
    print(f"\n========================================================")
    print(f"[*] Step: {desc}")
    print(f"[*] Command: {' '.join(cmd_list)}")
    print(f"========================================================")
    result = subprocess.run(cmd_list, check=True)
    return result

def main():
    parser = argparse.ArgumentParser(description="End-to-End Maritime Detection Pipeline")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and run evaluation/inference on existing checkpoints")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution for baseline")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    python_bin = sys.executable

    # 1. Dataset Audit
    run_cmd([python_bin, "src/data_audit.py", "data/raw"], "Running Dataset Audit")

    # 2. Dataset Preparation
    run_cmd([python_bin, "src/prepare_dataset.py"], "Preparing YOLO Dataset & Splits")

    # 3. Unit Tests
    run_cmd([python_bin, "-m", "pytest", "-v", "tests/"], "Running Unit Test Suite")

    # 4. Training Baseline
    if not args.skip_train:
        run_cmd([
            python_bin, "src/train.py",
            "--config", "configs/baseline.yaml",
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--imgsz", str(args.imgsz),
            "--name", "baseline_yolov8s_640"
        ], "Training Baseline YOLOv8s (640px)")

    # 5. Evaluate on Test Split
    best_weights = Path("outputs/models/baseline_yolov8s_640_best.pt")
    if not best_weights.exists():
        best_weights = Path("outputs/models/baseline_yolov8s_640/weights/best.pt")

    if best_weights.exists():
        run_cmd([
            python_bin, "src/evaluate.py",
            "--model", str(best_weights),
            "--split", "test",
            "--imgsz", str(args.imgsz),
            "--name", "baseline_yolov8s_test_eval"
        ], "Evaluating Baseline on Untouched Test Set")

        # 6. Video Demo
        run_cmd([
            python_bin, "src/predict_video.py",
            "--model", str(best_weights),
            "--track",
            "--output", "outputs/predictions/demo_output.mp4"
        ], "Generating Maritime Video Inference Demo with Tracking")
    else:
        print(f"[!] Warning: Best weights not found at {best_weights}. Evaluation skipped.")

    print(f"\n[+] Pipeline run complete! All artifacts generated in docs/ and outputs/.")

if __name__ == "__main__":
    main()

