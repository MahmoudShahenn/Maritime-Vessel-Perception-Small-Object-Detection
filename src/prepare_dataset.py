"""
Dataset Preparation & YOLO Conversion Script for SeaShips (CVOD01)
- Converts Pascal VOC XML to normalized YOLO format.
- Groups consecutive surveillance frames to prevent video data leakage.
- Generates 70/15/15 train/val/test splits with deterministic seed.
- Populates data/processed/ or writes manifest files.
"""

import os
import sys
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict, Counter

CLASSES = [
    "ore carrier",
    "bulk cargo carrier",
    "general cargo ship",
    "container ship",
    "fishing boat",
    "passenger ship"
]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h):
    """Convert Pascal VOC (xmin, ymin, xmax, ymax) to YOLO (x_center, y_center, w, h)."""
    # Clamp coordinates to image boundaries
    xmin = max(0.0, min(float(xmin), float(img_w)))
    ymin = max(0.0, min(float(ymin), float(img_h)))
    xmax = max(0.0, min(float(xmax), float(img_w)))
    ymax = max(0.0, min(float(ymax), float(img_h)))

    box_w = xmax - xmin
    box_h = ymax - ymin

    if box_w <= 0 or box_h <= 0 or img_w <= 0 or img_h <= 0:
        return None

    x_center = (xmin + xmax) / (2.0 * img_w)
    y_center = (ymin + ymax) / (2.0 * img_h)
    norm_w = box_w / float(img_w)
    norm_h = box_h / float(img_h)

    # Clamping normalized values to [0.0, 1.0]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))

    return x_center, y_center, norm_w, norm_h

def convert_annotation(xml_path: Path):
    """Parse VOC XML and convert to list of YOLO formatted lines."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        return None, None, f"Failed to parse {xml_path}: {e}"

    size_elem = root.find("size")
    if size_elem is None:
        return None, None, "Missing <size> tag"

    img_w = float(size_elem.findtext("width", 1920))
    img_h = float(size_elem.findtext("height", 1080))

    yolo_lines = []
    classes_present = set()

    for obj in root.findall("object"):
        cname = obj.findtext("name", "").strip().lower()
        if cname not in CLASS_TO_IDX:
            continue

        cid = CLASS_TO_IDX[cname]
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = float(bndbox.findtext("xmin", 0))
            ymin = float(bndbox.findtext("ymin", 0))
            xmax = float(bndbox.findtext("xmax", 0))
            ymax = float(bndbox.findtext("ymax", 0))
        except ValueError:
            continue

        yolo_box = voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h)
        if yolo_box is None:
            continue

        xc, yc, nw, nh = yolo_box
        yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        classes_present.add(cid)

    return yolo_lines, classes_present, None

def partition_dataset_sequence_aware(xml_files: list, seed: int = 42, chunk_size: int = 50,
                                     train_ratio: float = 0.70, val_ratio: float = 0.15):
    """
    Groups consecutive surveillance frames into chunks (representing contiguous video sequences)
    to prevent temporal data leakage, then assigns chunks to train, val, and test partitions.
    """
    random.seed(seed)
    
    # Sort files deterministically by integer ID if numeric
    def get_sort_key(p: Path):
        try:
            return int(p.stem)
        except ValueError:
            return p.stem

    sorted_files = sorted(xml_files, key=get_sort_key)
    
    # Group into chunks
    chunks = []
    current_chunk = []
    for f in sorted_files:
        current_chunk.append(f)
        if len(current_chunk) >= chunk_size:
            chunks.append(current_chunk)
            current_chunk = []
    if current_chunk:
        chunks.append(current_chunk)

    print(f"[*] Total frames: {len(sorted_files)} grouped into {len(chunks)} sequence chunks (chunk size ~ {chunk_size})")

    # Shuffle sequence chunks with fixed seed
    shuffled_indices = list(range(len(chunks)))
    random.shuffle(shuffled_indices)

    num_train_chunks = int(round(len(chunks) * train_ratio))
    num_val_chunks = int(round(len(chunks) * val_ratio))

    train_indices = set(shuffled_indices[:num_train_chunks])
    val_indices = set(shuffled_indices[num_train_chunks:num_train_chunks + num_val_chunks])
    test_indices = set(shuffled_indices[num_train_chunks + num_val_chunks:])

    splits = {"train": [], "val": [], "test": []}
    for idx, chunk in enumerate(chunks):
        if idx in train_indices:
            splits["train"].extend(chunk)
        elif idx in val_indices:
            splits["val"].extend(chunk)
        else:
            splits["test"].extend(chunk)

    print(f"[+] Leakage-resistant split generated:")
    print(f"    Train: {len(splits['train'])} ({len(splits['train'])/len(sorted_files)*100:.1f}%)")
    print(f"    Val:   {len(splits['val'])} ({len(splits['val'])/len(sorted_files)*100:.1f}%)")
    print(f"    Test:  {len(splits['test'])} ({len(splits['test'])/len(sorted_files)*100:.1f}%)")

    return splits

def link_or_copy(src: Path, dst: Path):
    """Attempt hardlink, fallback to copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def prepare_yolo_dataset(raw_dir: str = "data/raw", processed_dir: str = "data/processed",
                         splits_dir: str = "data/splits", seed: int = 42):
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)
    split_path = Path(splits_dir)
    
    ann_dir = raw_path / "Annotations"
    img_dir = raw_path / "JPEGImages"

    if not ann_dir.exists() or not img_dir.exists():
        print(f"[!] Error: {raw_dir} is missing Annotations or JPEGImages.")
        sys.exit(1)

    xml_files = sorted(list(ann_dir.glob("*.xml")))
    print(f"[*] Processing {len(xml_files)} annotations...")

    splits = partition_dataset_sequence_aware(xml_files, seed=seed)

    split_path.mkdir(parents=True, exist_ok=True)

    # Class counter per split
    split_class_counts = {s: Counter() for s in ["train", "val", "test"]}

    for split_name, files in splits.items():
        img_out_dir = proc_path / "images" / split_name
        lbl_out_dir = proc_path / "labels" / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        manifest_lines = []

        for xml_file in files:
            stem = xml_file.stem
            # Locate image
            img_cands = [img_dir / f"{stem}.jpg", img_dir / f"{stem}.JPG", img_dir / f"{stem}.png"]
            src_img = next((p for p in img_cands if p.exists()), None)
            if src_img is None:
                continue

            yolo_lines, classes_present, err = convert_annotation(xml_file)
            if err:
                continue

            # Record class counts
            for c in classes_present:
                split_class_counts[split_name][CLASSES[c]] += 1

            # Write label file
            dst_lbl = lbl_out_dir / f"{stem}.txt"
            with open(dst_lbl, "w") as f:
                f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

            # Link or copy image
            dst_img = img_out_dir / f"{stem}.jpg"
            link_or_copy(src_img, dst_img)

            # Record manifest
            manifest_lines.append(str(dst_img.resolve()))

        # Save manifest
        manifest_file = split_path / f"{split_name}.txt"
        with open(manifest_file, "w") as f:
            f.write("\n".join(manifest_lines) + "\n")
        print(f"[+] Saved {split_name} manifest: {len(manifest_lines)} images to {manifest_file}")

    # Generate dataset.yaml for Ultralytics
    dataset_yaml = f"""# SeaShips YOLO Dataset Configuration (CVOD01)
path: {proc_path.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: ore carrier
  1: bulk cargo carrier
  2: general cargo ship
  3: container ship
  4: fishing boat
  5: passenger ship
"""
    yaml_path = Path("configs/dataset.yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(dataset_yaml)
    print(f"[+] Created Ultralytics dataset configuration at {yaml_path}")

    # Save split distribution summary
    split_summary = {
        "split_counts": {s: len(splits[s]) for s in ["train", "val", "test"]},
        "classes_per_split": {s: dict(split_class_counts[s]) for s in ["train", "val", "test"]}
    }
    with open(split_path / "split_summary.json", "w") as f:
        import json
        json.dump(split_summary, f, indent=2)

    print(f"[+] Split summary saved to {split_path / 'split_summary.json'}")
    return split_summary

if __name__ == "__main__":
    prepare_yolo_dataset()

