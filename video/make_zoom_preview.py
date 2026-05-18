"""
Generate a static-crop pose preview: computes a single fixed crop region
from the median player bbox, then applies that same crop to every frame.
No following, no dynamic zoom — just a stable windowed view of the player.

Usage:
    python video/make_zoom_preview.py --player video/1_Cameron_Boozer
    python video/make_zoom_preview.py --player video/1_Cameron_Boozer --clips 14 19
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CROP_PAD = 0.45  # padding on each side as fraction of bbox size


def _fixed_crop(bboxes: list[list | None], W: int, H: int) -> tuple[int, int, int, int]:
    """Compute a single fixed crop region from median bbox + padding."""
    valid = [b for b in bboxes if b is not None]
    if not valid:
        return 0, 0, W, H

    arr = np.array(valid, float)
    med = np.median(arr, axis=0)  # [x1, y1, x2, y2]
    x1, y1, x2, y2 = med
    bw, bh = x2 - x1, y2 - y1

    cx1 = max(0, int(x1 - bw * CROP_PAD))
    cy1 = max(0, int(y1 - bh * CROP_PAD))
    cx2 = min(W, int(x2 + bw * CROP_PAD))
    cy2 = min(H, int(y2 + bh * CROP_PAD))
    return cx1, cy1, cx2, cy2


def make_zoom(clip_path: Path, tracking_json: Path) -> Path:
    meta    = json.loads(tracking_json.read_text())
    fmeta   = meta["frames"]
    n_total = len(fmeta)

    raw_bboxes = [fm.get("bbox") for fm in fmeta]

    cap = cv2.VideoCapture(str(clip_path))
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cx1, cy1, cx2, cy2 = _fixed_crop(raw_bboxes, W, H)
    cW = cx2 - cx1
    cH = cy2 - cy1

    out_path = clip_path.parent / (clip_path.stem + "_pose_zoom.mp4")
    writer   = cv2.VideoWriter(str(out_path),
                               cv2.VideoWriter_fourcc(*"mp4v"), fps, (cW, cH))

    for _ in range(n_total):
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[cy1:cy2, cx1:cx2]
        writer.write(crop)

    cap.release()
    writer.release()
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True)
    parser.add_argument("--clips",  nargs="*", type=int)
    args = parser.parse_args()

    clips_dir = Path(args.player) / "jump_shot_clips"

    if args.clips:
        clip_paths = [clips_dir / f"clip_{n:03d}.mp4" for n in args.clips]
    else:
        clip_paths = sorted(p for p in clips_dir.glob("clip_???.mp4")
                            if "preview" not in p.stem and "pose" not in p.stem
                            and "zoom" not in p.stem)

    print(f"Player: {Path(args.player).name}  |  {len(clip_paths)} clips\n")

    for clip_path in clip_paths:
        tj = clips_dir / (clip_path.stem + "_tracking.json")
        if not clip_path.exists() or not tj.exists():
            print(f"  skipping {clip_path.name} (missing files)")
            continue
        out = make_zoom(clip_path, tj)
        print(f"  {clip_path.stem} → {out.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
