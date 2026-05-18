"""
Interactive clip review — OpenCV video window mode.

Video plays in a window. Press keys directly on the video window:
  k  →  keep, advance to next
  d  →  delete, advance to next
  u  →  undo last deletion (stays on current clip)
  r  →  replay from start
  q / ESC  →  quit (progress saved, resumes next run)

Skips already-reviewed clips automatically. Use --reset to start fresh.

Usage:
    python video/review_clips.py --player video/1_Cameron_Boozer
    python video/review_clips.py --player video/1_Cameron_Boozer --reset
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import subprocess

import cv2
import numpy as np


DISPLAY_W = 960
DISPLAY_H = 540
WINDOW    = "Clip Review"
FONT      = cv2.FONT_HERSHEY_SIMPLEX
RED       = (50, 50, 220)
WHITE     = (255, 255, 255)
GRAY      = (160, 160, 160)


def save_reviewed(index: dict, index_path: Path, reviewed: set[str]) -> None:
    index["reviewed_clips"] = sorted(reviewed)
    index_path.write_text(json.dumps(index, indent=2))


def load_frames(path: Path) -> tuple[list[np.ndarray], int]:
    """Decode all frames via ffmpeg pipe. Returns (frames, fps)."""
    # Get fps first
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        num, den = probe.stdout.strip().split("/")
        fps = int(num) // int(den)
    except Exception:
        fps = 30

    proc = subprocess.run(
        ["ffmpeg", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-vf", f"scale={DISPLAY_W}:{DISPLAY_H}",
         "-loglevel", "quiet", "pipe:1"],
        capture_output=True,
    )
    raw  = proc.stdout
    size = DISPLAY_W * DISPLAY_H * 3
    frames = [
        np.frombuffer(raw[off:off + size], dtype=np.uint8).reshape(DISPLAY_H, DISPLAY_W, 3).copy()
        for off in range(0, len(raw) - size + 1, size)
    ]
    return frames, fps


def draw_overlay(frame: np.ndarray, i: int, total: int,
                 clip_stem: str, marked_delete: bool, n_deleted: int) -> np.ndarray:
    h, w = frame.shape[:2]
    out  = frame.copy()

    cv2.rectangle(out, (0, 0), (w, 52), (20, 20, 20), -1)
    cv2.putText(out, f"[{i+1}/{total}]", (10, 36), FONT, 0.9, GRAY, 2)
    cv2.putText(out, clip_stem, (110, 36), FONT, 0.65, WHITE, 1)

    if marked_delete:
        badge = f" DELETE ({n_deleted} marked) "
        (bw, _), _ = cv2.getTextSize(badge, FONT, 0.65, 2)
        cv2.rectangle(out, (w - bw - 16, 10), (w - 6, 46), RED, -1)
        cv2.putText(out, badge, (w - bw - 12, 36), FONT, 0.65, WHITE, 2)

    cv2.rectangle(out, (0, h - 38), (w, h), (20, 20, 20), -1)
    cv2.putText(out, "k=keep    d=delete    u=undo    r=replay    q=quit",
                (10, h - 10), FONT, 0.55, GRAY, 1)
    return out


def confirm_in_window(to_delete: list[str]) -> bool:
    """Show a summary screen in the cv2 window. y=confirm, n/ESC=cancel."""
    bg = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    if not to_delete:
        lines = ["Review complete — nothing to delete.", "", "Press any key to exit."]
        for j, line in enumerate(lines):
            cv2.putText(bg, line, (40, 180 + j * 50), FONT, 0.8, WHITE, 2)
        cv2.imshow(WINDOW, bg)
        cv2.waitKey(0)
        return False

    lines = [f"Delete {len(to_delete)} clip(s)?", ""] + to_delete[:12]
    if len(to_delete) > 12:
        lines.append(f"  ... and {len(to_delete) - 12} more")
    lines += ["", "y = confirm delete    n / ESC = cancel"]
    for j, line in enumerate(lines):
        cv2.putText(bg, line, (40, 60 + j * 38), FONT, 0.65, WHITE, 1)
    cv2.imshow(WINDOW, bg)

    while True:
        key = cv2.waitKey(33) & 0xFF
        if key == ord('y') or key == ord('Y'):
            return True
        if key in (ord('n'), ord('N'), 27):
            return False


def review_clip(frames: list[np.ndarray], fps: int, i: int, total: int,
                clip_stem: str, to_delete: list[str]) -> str:
    """Loop frames in cv2 window; return action key."""
    delay  = max(1, int(1000 / fps))
    marked = clip_stem in to_delete
    fi     = 0

    while True:
        frame = draw_overlay(frames[fi % len(frames)], i, total,
                             clip_stem, marked, len(to_delete))
        cv2.imshow(WINDOW, frame)
        fi += 1

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('k') or key == ord('K'):
            return 'k'
        elif key == ord('d') or key == ord('D'):
            return 'd'
        elif key == ord('u') or key == ord('U'):
            return 'u'
        elif key == ord('r') or key == ord('R'):
            fi = 0
        elif key == ord('q') or key == ord('Q') or key == 27:
            return 'q'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True)
    parser.add_argument("--reset", action="store_true",
                        help="Clear reviewed history and start fresh")
    args = parser.parse_args()

    clips_dir  = Path(args.player) / "jump_shot_clips"
    index_path = clips_dir / "index.json"

    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    reviewed: set[str] = set()
    if not args.reset:
        reviewed = set(index.get("reviewed_clips", []))
    elif "reviewed_clips" in index:
        del index["reviewed_clips"]
        index_path.write_text(json.dumps(index, indent=2))

    all_previews = sorted(
        p for p in clips_dir.glob("clip_*_preview.mp4")
        if not p.name.endswith("_pose_preview.mp4")
    )
    previews = [p for p in all_previews
                if p.stem.replace("_preview", "") not in reviewed]

    skipped = len(all_previews) - len(previews)
    if skipped:
        print(f"Skipping {skipped} already-reviewed clips.")

    if not previews:
        print("All clips reviewed. Use --reset to start over.")
        return

    total = len(previews)
    print(f"\nPlayer: {Path(args.player).name}  |  {total} clips to review")
    print("Focus the video window, then use:  k=keep  d=delete  u=undo  r=replay  q=quit\n")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 600)

    to_delete: list[str] = []
    i = 0

    while i < len(previews):
        preview   = previews[i]
        clip_stem = preview.stem.replace("_preview", "")

        print(f"  loading [{i+1}/{total}] ...", end="\r", flush=True)
        frames, fps = load_frames(preview)
        if not frames:
            print(f"  [skip] {clip_stem} — could not decode")
            i += 1
            continue

        action = review_clip(frames, fps, i, total, clip_stem, to_delete)

        if action == 'k':
            if clip_stem in to_delete:
                to_delete.remove(clip_stem)
            reviewed.add(clip_stem)
            save_reviewed(index, index_path, reviewed)
            print(f"[{i+1}/{total}]  {clip_stem}  → kept")
            i += 1

        elif action == 'd':
            if clip_stem not in to_delete:
                to_delete.append(clip_stem)
            reviewed.add(clip_stem)
            save_reviewed(index, index_path, reviewed)
            print(f"[{i+1}/{total}]  {clip_stem}  → DELETE  ({len(to_delete)} marked)")
            i += 1

        elif action == 'u':
            if to_delete:
                undone = to_delete.pop()
                reviewed.discard(undone)
                save_reviewed(index, index_path, reviewed)
                # Go back to the undone clip
                undone_idx = next(
                    (j for j, p in enumerate(previews)
                     if p.stem.replace("_preview", "") == undone),
                    None
                )
                if undone_idx is not None:
                    print(f"  ← undo  {undone}  (going back)")
                    i = undone_idx
                else:
                    print(f"  ← undo  {undone}  (clip not in list, staying)")
            else:
                print("  nothing to undo")
            # stay (i not incremented unless we jumped back)

        elif action == 'q':
            print(f"\nQuit early. Progress saved ({len(reviewed)} reviewed).")
            break

    # Show confirmation screen inside cv2 window (avoids macOS freeze on destroyAllWindows)
    confirmed = confirm_in_window(to_delete)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if not confirmed:
        if to_delete:
            print("Cancelled. Delete marks saved — they'll show on next run.")
        else:
            print("\nNo clips marked for deletion. Done.")
        return

    for stem in to_delete:
        for suffix in [".mp4", "_preview.mp4", "_tracking.json",
                       "_pose.json", "_pose_preview.mp4", "_pose_preview_web.mp4",
                       "_pose_zoom.mp4", "_pose_zoom_web.mp4"]:
            p = clips_dir / f"{stem}{suffix}"
            if p.exists():
                p.unlink()
                print(f"  deleted  {p.name}")

    if index_path.exists():
        index = json.loads(index_path.read_text())
        before = len(index.get("clips", []))
        index["clips"] = [c for c in index.get("clips", [])
                          if Path(c["clip"]).stem not in to_delete]
        index["total_clips"] = len(index["clips"])
        for stem in to_delete:
            reviewed.discard(stem)
        index["reviewed_clips"] = sorted(reviewed)
        index_path.write_text(json.dumps(index, indent=2))
        print(f"\nindex.json: {before} → {index['total_clips']} clips")

    print(f"Done. Removed {len(to_delete)} clip(s).")


if __name__ == "__main__":
    main()
