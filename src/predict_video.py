"""
Video Inference & Demonstration Pipeline for Maritime Vessel Detection (CVOD01)
Processes RGB video or image sequences, executes detector, overlays tracking IDs and detections,
measures real-time FPS, and outputs an annotated MP4 video.
"""

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.utils import CLASS_NAMES, CLASS_COLORS, compute_box_iou

class SimpleMaritimeTracker:
    """Lightweight, self-contained IoU tracker for maritime video."""
    def __init__(self, iou_thresh: float = 0.25, max_lost: int = 15):
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks = {}  # id -> {box, cls, conf, lost}

    def update(self, detections: list):
        # detections: list of {"box": [x1, y1, x2, y2], "cls": int, "conf": float}
        updated_tracks = {}
        matched_dets = set()

        # Match to existing tracks
        for tid, tinfo in self.tracks.items():
            best_iou = 0.0
            best_det_idx = -1
            for d_idx, det in enumerate(detections):
                if d_idx in matched_dets:
                    continue
                if det["cls"] != tinfo["cls"]:
                    continue
                iou = compute_box_iou(np.array(tinfo["box"]), np.array(det["box"]))
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = d_idx

            if best_iou >= self.iou_thresh and best_det_idx >= 0:
                matched_dets.add(best_det_idx)
                det = detections[best_det_idx]
                updated_tracks[tid] = {
                    "box": det["box"],
                    "cls": det["cls"],
                    "conf": det["conf"],
                    "lost": 0
                }
            else:
                if tinfo["lost"] + 1 < self.max_lost:
                    updated_tracks[tid] = {
                        "box": tinfo["box"],
                        "cls": tinfo["cls"],
                        "conf": tinfo["conf"],
                        "lost": tinfo["lost"] + 1
                    }

        # Initialize new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                updated_tracks[self.next_id] = {
                    "box": det["box"],
                    "cls": det["cls"],
                    "conf": det["conf"],
                    "lost": 0
                }
                self.next_id += 1

        self.tracks = updated_tracks
        # Return active tracks visible in current frame
        active = []
        for tid, tinfo in self.tracks.items():
            if tinfo["lost"] == 0:
                active.append({
                    "id": tid,
                    "box": tinfo["box"],
                    "cls": tinfo["cls"],
                    "conf": tinfo["conf"]
                })
        return active

def parse_args():
    parser = argparse.ArgumentParser(description="Maritime Vessel Video Inference Demo")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file")
    parser.add_argument("--image-seq", type=str, default=None, help="Directory of frames to compile and infer as video")
    parser.add_argument("--model", type=str, required=True, help="Trained YOLO model path (.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="Device ('0', 'cpu')")
    parser.add_argument("--track", action="store_true", help="Enable ByteTrack temporal vessel tracking")
    parser.add_argument("--output", type=str, default="outputs/predictions/demo_output.mp4", help="Output MP4 path")
    return parser.parse_args()

def generate_sample_video_from_test_sequence(test_dir: str = "data/processed/images/test", output_video: str = "data/sample_test_seq.mp4", num_frames: int = 90):
    """Compiles consecutive test frames into a benchmark maritime surveillance video."""
    p = Path(test_dir)
    img_files = sorted(list(p.glob("*.jpg")))
    if not img_files:
        raise RuntimeError(f"No test images found in {test_dir} to construct demo video.")

    selected_files = img_files[:min(len(img_files), num_frames)]
    first_img = cv2.imread(str(selected_files[0]))
    h, w = first_img.shape[:2]

    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 15.0, (w, h))

    for img_p in selected_files:
        frame = cv2.imread(str(img_p))
        if frame is not None:
            out.write(frame)
    out.release()
    print(f"[+] Compiled {len(selected_files)} test frames into synthetic surveillance video: {output_video}")
    return output_video

def run_video_inference():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_input = args.video
    if video_input is None or not os.path.exists(video_input):
        if args.image_seq and os.path.exists(args.image_seq):
            video_input = generate_sample_video_from_test_sequence(args.image_seq, "data/sample_test_seq.mp4")
        else:
            print("[*] No video provided. Generating test sequence video from test partition...")
            video_input = generate_sample_video_from_test_sequence("data/processed/images/test", "data/sample_test_seq.mp4")

    print(f"[*] Opening video source: {video_input}")
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video file {video_input}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[*] Input Video Properties: {width}x{height} @ {fps:.1f} FPS, Total Frames: {total_frames}")

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"[*] Loading model: {args.model}")
    model = YOLO(args.model)

    frame_idx = 0
    processing_times = []
    total_detections = 0

    print(f"[*] Starting video processing (Tracking={'Enabled' if args.track else 'Disabled'})...")
    start_total_time = time.perf_counter()
    tracker = SimpleMaritimeTracker(iou_thresh=0.25, max_lost=15) if args.track else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t0 = time.perf_counter()

        results = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False
        )[0]

        # Extract detections
        raw_dets = []
        if results.boxes is not None:
            for b in results.boxes:
                raw_dets.append({
                    "box": b.xyxy[0].cpu().numpy().tolist(),
                    "cls": int(b.cls[0].cpu().item()),
                    "conf": float(b.conf[0].cpu().item())
                })

        if tracker is not None:
            display_targets = tracker.update(raw_dets)
        else:
            display_targets = [{"id": None, "box": d["box"], "cls": d["cls"], "conf": d["conf"]} for d in raw_dets]

        t1 = time.perf_counter()
        dt_ms = (t1 - t0) * 1000.0
        processing_times.append(dt_ms)
        current_fps = 1000.0 / dt_ms if dt_ms > 0 else 0.0

        # Draw detections
        annotated = frame.copy()
        num_vessels = len(display_targets)
        total_detections += num_vessels

        for t in display_targets:
            xyxy = [int(round(v)) for v in t["box"]]
            conf = t["conf"]
            cls_id = t["cls"]
            track_id = t["id"]

            cname = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
            color_rgb = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

            x1, y1, x2, y2 = xyxy
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)

            if track_id is not None:
                label = f"ID#{track_id} {cname} {conf:.2f}"
            else:
                label = f"{cname} {conf:.2f}"

            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            by = max(y1, th + 4)
            cv2.rectangle(annotated, (x1, by - th - 4), (x1 + tw + 4, by + bl), color_bgr, -1)
            cv2.putText(annotated, label, (x1 + 2, by - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Telemetry HUD banner (top-left)
        hud_text = f"FPS: {current_fps:.1f} | Latency: {dt_ms:.1f}ms | Targets: {num_vessels} | Frame: {frame_idx}/{total_frames}"
        cv2.rectangle(annotated, (15, 15), (600, 50), (0, 0, 0), -1)
        cv2.putText(annotated, hud_text, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        writer.write(annotated)

        if frame_idx % 20 == 0:
            print(f"    - Frame {frame_idx}/{total_frames}: {current_fps:.1f} FPS, {num_vessels} targets")

    cap.release()
    writer.release()
    total_elapsed = time.perf_counter() - start_total_time
    avg_latency = float(np.mean(processing_times)) if processing_times else 0.0
    effective_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    print(f"[+] Video processing complete!")
    print(f"    - Processed Frames:  {frame_idx}")
    print(f"    - Total Duration:    {total_elapsed:.2f} s")
    print(f"    - Average Latency:   {avg_latency:.2f} ms")
    print(f"    - Effective FPS:     {effective_fps:.1f} FPS")
    print(f"    - Total Detections:  {total_detections}")
    print(f"    - Output Saved To:   {output_path.resolve()}")

    return {
        "output_path": str(output_path.resolve()),
        "processed_frames": frame_idx,
        "avg_latency_ms": avg_latency,
        "effective_fps": effective_fps,
        "total_detections": total_detections
    }

if __name__ == "__main__":
    run_video_inference()

