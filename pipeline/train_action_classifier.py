"""
Convert COCO dataset → YOLO format, then train YOLOv11n action classifier.

Usage:
    python pipeline/train_action_classifier.py

Output:
    models/action_classifier/weights/best.pt
"""

from __future__ import annotations
import json
import os
import shutil
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_DIR = Path("basketball-player-detection-2-Forked on 8-19-2025.coco")
YOLO_DIR    = Path("data/action_classifier_yolo")
MODEL_DIR   = Path("models")

SPLITS = ["train", "valid", "test"]

# ── Step 1: COCO → YOLO conversion ────────────────────────────────────────────

def coco_to_yolo(dataset_dir: Path, yolo_dir: Path) -> dict:
    """Convert COCO annotations to YOLO txt format. Returns class name map."""
    print("Converting COCO → YOLO format...")
    class_map = {}

    for split in SPLITS:
        ann_file = dataset_dir / split / "_annotations.coco.json"
        if not ann_file.exists():
            print(f"  Skipping {split} (no annotations found)")
            continue

        with open(ann_file) as f:
            coco = json.load(f)

        # Build class map from first split encountered
        if not class_map:
            class_map = {c["id"]: c["name"] for c in coco["categories"]}

        img_dir   = yolo_dir / split / "images"
        label_dir = yolo_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        # Map image_id → file info
        id_to_img = {img["id"]: img for img in coco["images"]}

        # Group annotations by image
        ann_by_img: dict[int, list] = {}
        for ann in coco["annotations"]:
            ann_by_img.setdefault(ann["image_id"], []).append(ann)

        for img_id, img_info in id_to_img.items():
            src = dataset_dir / split / img_info["file_name"]
            dst = img_dir / img_info["file_name"]
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

            W, H = img_info["width"], img_info["height"]
            label_path = label_dir / (Path(img_info["file_name"]).stem + ".txt")

            anns = ann_by_img.get(img_id, [])
            lines = []
            for ann in anns:
                cat_id = ann["category_id"]
                x, y, w, h = [float(v) for v in ann["bbox"]]  # COCO: top-left x,y + w,h
                cx = (x + w / 2) / W
                cy = (y + h / 2) / H
                nw = w / W
                nh = h / H
                lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            label_path.write_text("\n".join(lines))

        print(f"  {split}: {len(id_to_img)} images, {len(coco['annotations'])} annotations")

    return class_map


def write_yaml(yolo_dir: Path, class_map: dict) -> Path:
    """Write data.yaml for ultralytics."""
    names = [class_map[i] for i in sorted(class_map)]
    yaml_path = yolo_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {yolo_dir.resolve()}\n"
        f"train: train/images\n"
        f"val:   valid/images\n"
        f"test:  test/images\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    print(f"\ndata.yaml written → {yaml_path}")
    return yaml_path


# ── Step 2: Train ─────────────────────────────────────────────────────────────

def train(yaml_path: Path) -> None:
    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    from ultralytics import YOLO

    print("\nStarting training...")
    model = YOLO("yolo11n.pt")   # downloads pretrained nano weights (~5MB)

    results = model.train(
        data=str(yaml_path.resolve()),
        epochs=80,
        imgsz=640,
        batch=4,            # conservative for Mac RAM
        device="mps",
        amp=False,          # disable AMP — fixes MPS stride/view bug in PyTorch 2.5
        project=str(MODEL_DIR),
        name="action_classifier",
        exist_ok=True,
        patience=20,        # early stop if no improvement for 20 epochs
        save_period=10,     # save checkpoint every 10 epochs
        verbose=True,
    )

    best = MODEL_DIR / "action_classifier" / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights → {best}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

    # Skip image conversion if already done, but always (re)write yaml
    if not (YOLO_DIR / "train" / "images").exists():
        class_map = coco_to_yolo(DATASET_DIR, YOLO_DIR)
    else:
        print("YOLO images already exist, skipping conversion.")
        with open(DATASET_DIR / "train" / "_annotations.coco.json") as f:
            coco = json.load(f)
        class_map = {c["id"]: c["name"] for c in coco["categories"]}

    yaml_path = write_yaml(YOLO_DIR, class_map)

    train(yaml_path)
