"""
Review pose preview videos — OpenCV window mode.

Video plays automatically and loops. Controls on the video window:
  Space          pause / resume
  ← / →          previous / next frame  (works while paused or playing)
  k              keep, advance to next clip
  d              delete, advance to next clip
  u              undo last deletion (jump back to that clip)
  r              replay from start
  q / ESC        quit (progress saved, resumes next run)

Use --reset to clear review history and start fresh.

Usage:
    python video/review_pose.py --player video/1_Cameron_Boozer
    python video/review_pose.py --player video/1_Cameron_Boozer --reset
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


DISPLAY_W = 960
DISPLAY_H = 540
WINDOW    = "Pose Review"
FONT      = cv2.FONT_HERSHEY_SIMPLEX
RED       = (50, 50, 220)
WHITE     = (255, 255, 255)
GRAY      = (160, 160, 160)
CYAN      = (200, 200, 0)
YELLOW    = (0, 200, 200)

SUFFIXES = [
    ".mp4", "_preview.mp4", "_tracking.json",
    "_pose.json", "_pose_preview.mp4", "_pose_preview_web.mp4",
    "_pose_zoom.mp4", "_pose_zoom_web.mp4",
]


def save_reviewed(index: dict, index_path: Path, reviewed: set[str]) -> None:
    index["reviewed_pose_clips"] = sorted(reviewed)
    index_path.write_text(json.dumps(index, indent=2))


def load_frames(path: Path) -> tuple[list[np.ndarray], int]:
    """Decode all frames via ffmpeg pipe."""
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        num, den = probe.stdout.strip().split("/")
        fps = int(num) // max(int(den), 1)
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
        np.frombuffer(raw[off:off + size], dtype=np.uint8)
          .reshape(DISPLAY_H, DISPLAY_W, 3).copy()
        for off in range(0, len(raw) - size + 1, size)
    ]
    return frames, fps


def draw_overlay(frame: np.ndarray, i: int, total: int, stem: str,
                 src_label: str, fi: int, n_frames: int,
                 paused: bool, marked_delete: bool, n_deleted: int) -> np.ndarray:
    h, w = frame.shape[:2]
    out  = frame.copy()

    # Top bar
    cv2.rectangle(out, (0, 0), (w, 58), (20, 20, 20), -1)
    cv2.putText(out, f"[{i+1}/{total}]", (10, 36), FONT, 0.9, GRAY, 2)
    cv2.putText(out, stem, (110, 36), FONT, 0.6, WHITE, 1)
    if src_label:
        cv2.putText(out, src_label, (110, 54), FONT, 0.45, CYAN, 1)

    # Frame counter + pause indicator
    frame_txt = f"{'⏸ ' if paused else ''}frame {fi+1}/{n_frames}"
    cv2.putText(out, frame_txt, (w - 200, 36), FONT, 0.55, YELLOW, 1)

    # Delete badge + red border
    if marked_delete:
        badge = f" DELETE ({n_deleted}) "
        (bw, _), _ = cv2.getTextSize(badge, FONT, 0.65, 2)
        cv2.rectangle(out, (w - bw - 16, 8), (w - 6, 50), RED, -1)
        cv2.putText(out, badge, (w - bw - 12, 38), FONT, 0.65, WHITE, 2)
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), RED, 4)

    # Bottom bar
    cv2.rectangle(out, (0, h - 38), (w, h), (20, 20, 20), -1)
    cv2.putText(out, "space=pause  ←/→=frame  k=keep  d=delete  u=undo  r=replay  q=quit",
                (10, h - 10), FONT, 0.5, GRAY, 1)
    return out


def play_clip(frames: list[np.ndarray], fps: int, i: int, total: int,
              stem: str, src_label: str, to_delete: list[str]) -> str:
    n      = len(frames)
    delay  = max(1, int(1000 / fps))
    paused = False
    marked = stem in to_delete
    fi     = 0

    while True:
        frame = draw_overlay(frames[fi % n], i, total, stem, src_label,
                             fi % n, n, paused, marked, len(to_delete))
        cv2.imshow(WINDOW, frame)

        wait = 0 if paused else delay
        key  = cv2.waitKey(wait) & 0xFF

        # No key while playing → advance frame
        if key == 255 and not paused:
            fi = (fi + 1) % n
            continue

        if key == ord(' '):
            paused = not paused

        elif key in (83, 3, 0):    # → right arrow (macOS: 3 or 83)
            fi = min(fi + 1, n - 1)
            paused = True
        elif key in (81, 2):       # ← left arrow (macOS: 2 or 81)
            fi = max(fi - 1, 0)
            paused = True

        elif key == ord('k') or key == ord('K'):
            return 'k'
        elif key == ord('d') or key == ord('D'):
            return 'd'
        elif key == ord('u') or key == ord('U'):
            return 'u'
        elif key == ord('r') or key == ord('R'):
            fi = 0
            paused = False
        elif key in (ord('q'), ord('Q'), 27):
            return 'q'


def confirm_in_window(to_delete: list[str]) -> bool:
    bg = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    if not to_delete:
        cv2.putText(bg, "Review complete — nothing to delete.", (40, 200), FONT, 0.8, WHITE, 2)
        cv2.putText(bg, "Press any key to exit.", (40, 260), FONT, 0.7, GRAY, 1)
        cv2.imshow(WINDOW, bg)
        cv2.waitKey(0)
        return False

    lines = [f"Delete {len(to_delete)} clip(s)?", ""] + to_delete[:14]
    if len(to_delete) > 14:
        lines.append(f"  ... and {len(to_delete) - 14} more")
    lines += ["", "y = confirm delete    n / ESC = cancel"]
    for j, line in enumerate(lines):
        cv2.putText(bg, line, (40, 60 + j * 36), FONT, 0.6, WHITE, 1)
    cv2.imshow(WINDOW, bg)

    while True:
        key = cv2.waitKey(33) & 0xFF
        if key in (ord('y'), ord('Y')):
            return True
        if key in (ord('n'), ord('N'), 27):
            return False


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
        reviewed = set(index.get("reviewed_pose_clips", []))
    elif "reviewed_pose_clips" in index:
        del index["reviewed_pose_clips"]
        index_path.write_text(json.dumps(index, indent=2))

    all_stems = sorted(
        p.stem.replace("_pose", "")
        for p in clips_dir.glob("*_pose.json")
        if "summary" not in p.name
    )
    stems = [s for s in all_stems if s not in reviewed]

    skipped = len(all_stems) - len(stems)
    if skipped:
        print(f"Skipping {skipped} already-reviewed pose clips.")
    if not stems:
        print("All pose clips reviewed. Use --reset to start over.")
        return

    total = len(stems)
    print(f"\nPlayer: {Path(args.player).name}  |  {total} pose clips to review")
    print("Focus the video window:  space=pause  ←/→=frame  k=keep  d=delete  u=undo  q=quit\n")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, DISPLAY_W, DISPLAY_H + 20)

    to_delete: list[str] = []
    i = 0

    while i < len(stems):
        stem  = stems[i]
        web   = clips_dir / f"{stem}_pose_preview_web.mp4"
        raw   = clips_dir / f"{stem}_pose_preview.mp4"
        video = web if web.exists() else (raw if raw.exists() else None)

        if video is None:
            print(f"  [{i+1}/{total}] {stem} — no pose preview, skipping")
            reviewed.add(stem)
            save_reviewed(index, index_path, reviewed)
            i += 1
            continue

        src_label = ""
        tj = clips_dir / f"{stem}_tracking.json"
        if tj.exists():
            try:
                src_label = json.loads(tj.read_text()).get("source_video", "")[:55]
            except Exception:
                pass

        print(f"  loading [{i+1}/{total}] ...", end="\r", flush=True)
        frames, fps = load_frames(video)
        if not frames:
            print(f"  [{i+1}/{total}] {stem} — decode failed, skipping")
            reviewed.add(stem)
            save_reviewed(index, index_path, reviewed)
            i += 1
            continue

        action = play_clip(frames, fps, i, total, stem, src_label, to_delete)

        if action == 'k':
            if stem in to_delete:
                to_delete.remove(stem)
            reviewed.add(stem)
            save_reviewed(index, index_path, reviewed)
            print(f"[{i+1}/{total}]  {stem}  → kept")
            i += 1

        elif action == 'd':
            if stem not in to_delete:
                to_delete.append(stem)
            reviewed.add(stem)
            save_reviewed(index, index_path, reviewed)
            print(f"[{i+1}/{total}]  {stem}  → DELETE  ({len(to_delete)} marked)")
            i += 1

        elif action == 'u':
            if to_delete:
                undone = to_delete.pop()
                reviewed.discard(undone)
                save_reviewed(index, index_path, reviewed)
                undone_idx = next((j for j, s in enumerate(stems) if s == undone), None)
                if undone_idx is not None:
                    print(f"  ← undo  {undone}  (going back)")
                    i = undone_idx
                else:
                    print(f"  ← undo  {undone}")
            else:
                print("  nothing to undo")

        elif action == 'q':
            print(f"\nQuit early. Progress saved ({len(reviewed)} reviewed).")
            break

    confirmed = confirm_in_window(to_delete)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if not confirmed:
        if to_delete:
            print("Cancelled. Delete marks saved for next run.")
        else:
            print("\nNo clips marked for deletion. Done.")
        return

    for stem in to_delete:
        deleted = []
        for suffix in SUFFIXES:
            f = clips_dir / (stem + suffix)
            if f.exists():
                f.unlink()
                deleted.append(f.name)
        reviewed.discard(stem)
        print(f"  deleted {len(deleted)} files  ({stem})")

    if index_path.exists():
        index = json.loads(index_path.read_text())
        before = len(index.get("clips", []))
        index["clips"] = [c for c in index.get("clips", [])
                          if Path(c["clip"]).stem not in to_delete]
        index["total_clips"] = len(index["clips"])
        index["reviewed_pose_clips"] = sorted(reviewed)
        index_path.write_text(json.dumps(index, indent=2))
        print(f"\nindex.json: {before} → {index['total_clips']} clips")

    print(f"Done. Removed {len(to_delete)} clip(s).")


if __name__ == "__main__":
    main()
