"""
Manually annotate the release frame for each clip.

Controls:
  a / left-arrow   previous frame
  d / right-arrow  next frame
  A / D            jump 5 frames
  r                mark current frame as release
  n                save and next clip
  q                quit

Usage:
    python video/annotate_release.py --player video/1_Cameron_Boozer
    python video/annotate_release.py --player video/2_Darryn_Peterson
    python video/annotate_release.py --player video/3_AJ_Dybantsa
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def annotate_clip(video_path: Path, tj_path: Path) -> str:
    """Returns 'saved', 'skip', or 'quit'."""
    meta       = json.loads(tj_path.read_text())
    fps        = meta.get("fps", 60.0)
    clip_start = meta.get("clip_start_frame", 0)
    existing   = meta.get("release_frame")

    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cur    = existing if existing is not None else 0
    cur    = max(0, min(cur, total - 1))
    marked = existing
    cache  = {}

    def get_frame(idx):
        if idx not in cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            cache[idx] = f if ret else np.zeros((H, W, 3), np.uint8)
        return cache[idx].copy()

    win = "annotate_release"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(W, 1280), min(H, 720))

    result = "skip"

    while True:
        frame = get_frame(cur)
        t = cur / fps

        cv2.putText(frame, f"frame {cur}/{total-1}   t={t:.2f}s",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if marked is not None:
            mt = marked / fps
            cv2.putText(frame, f"release marked: frame {marked}  ({mt:.2f}s)",
                        (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if cur == marked:
                cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 255, 0), 5)

        cv2.putText(frame, "a/d=frame   r=mark   n=next   q=quit",
                    (15, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow(win, frame)
        key = cv2.waitKey(30) & 0xFF

        if key == 255:
            continue

        # right: d, right-arrow (3 on macOS)
        if key in (ord('d'), 3, 83):
            cur = min(cur + 1, total - 1)
        # left: a, left-arrow (2 on macOS)
        elif key in (ord('a'), 2, 81):
            cur = max(cur - 1, 0)
        # jump 5 right
        elif key == ord('D'):
            cur = min(cur + 5, total - 1)
        # jump 5 left
        elif key == ord('A'):
            cur = max(cur - 5, 0)
        elif key == ord('r'):
            marked = cur
        elif key == ord('n'):
            if marked is not None:
                meta["release_frame"] = marked
                tj_path.write_text(json.dumps(meta, indent=2))
                result = "saved"
            else:
                result = "skip"
            break
        elif key in (ord('q'), 27):
            result = "quit"
            break

    cap.release()
    cv2.destroyWindow(win)
    cv2.waitKey(1)
    return result, marked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True)
    args = parser.parse_args()

    clips_dir = Path(args.player) / "jump_shot_clips"
    paths = sorted(p for p in clips_dir.glob("clip_*.mp4")
                   if "preview" not in p.stem and "pose" not in p.stem
                   and "zoom" not in p.stem)

    print(f"\nPlayer: {Path(args.player).name}  |  {len(paths)} clips")
    print("a/d = frame   r = mark release   n = next clip   q = quit\n")

    for i, clip_path in enumerate(paths):
        tj_path = clips_dir / (clip_path.stem + "_tracking.json")
        preview = clips_dir / (clip_path.stem + "_pose_preview.mp4")
        video   = preview if preview.exists() else clip_path

        if not tj_path.exists():
            print(f"  [{i+1}/{len(paths)}] skipping — no tracking.json")
            continue

        meta     = json.loads(tj_path.read_text())
        fps      = meta.get("fps", 60.0)
        existing = meta.get("release_frame")
        label    = f"  [release @ {existing/fps:.2f}s]" if existing else "  [not annotated]"
        print(f"[{i+1}/{len(paths)}]  {clip_path.stem}{label}")

        status, marked = annotate_clip(video, tj_path)

        if status == "saved":
            print(f"  → saved frame {marked} ({marked/fps:.2f}s)")
        elif status == "skip":
            print("  → skipped")
        elif status == "quit":
            print("\nQuit.")
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
