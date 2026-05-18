"""
Run action classifier on all videos in a player folder, extract jump-shot clips.

Usage:
    python video/extract_jump_shots.py --player video/1_Cameron_Boozer

Output:
    video/<player_folder>/jump_shot_clips/<video_stem>_clip_001.mp4 ...
    video/<player_folder>/jump_shot_clips/detections.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

WEIGHTS      = Path("models/action_classifier/weights/best.pt")
JUMP_CLASS   = 6       # player-jump-shot
CONF_THRESH  = 0.35
SAMPLE_EVERY = 2       # process every 2nd frame (30fps from 60fps)
# A jump-shot action lasts ~1-2s. Merge detections within 2s of each other.
GAP_TOLERANCE_SEC = 2.0
MIN_SEG_SEC  = 0.3     # discard spurious detections shorter than this
CLIP_BUFFER_SEC = 0.5  # small buffer before/after the detected segment


def detect_jump_shots(video_path: Path, model: YOLO) -> tuple[list[dict], float]:
    """Return list of {frame, conf, bbox} for each jump-shot detection + fps."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detections = []

    frame_idx = 0
    print(f"\n  {video_path.name}")
    print(f"  {total} frames @ {fps:.0f}fps")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SAMPLE_EVERY == 0:
            results = model(frame, conf=CONF_THRESH, verbose=False)[0]
            for box in results.boxes:
                if int(box.cls[0]) == JUMP_CLASS:
                    detections.append({
                        "frame": frame_idx,
                        "conf": round(float(box.conf[0]), 3),
                        "bbox": list(map(int, box.xyxy[0])),
                    })

        frame_idx += 1
        if frame_idx % 1000 == 0:
            pct = frame_idx / total * 100
            print(f"  {frame_idx}/{total} ({pct:.0f}%)  detections: {len(detections)}")

    cap.release()
    print(f"  Total detections: {len(detections)}")
    return detections, fps


def group_segments(detections: list[dict], fps: float) -> list[tuple[int, int]]:
    """
    Merge detections within GAP_TOLERANCE_SEC into one segment.
    Consecutive frames of the same shot action stay as one clip.
    """
    if not detections:
        return []

    gap_frames = int(GAP_TOLERANCE_SEC * fps)
    min_frames = int(MIN_SEG_SEC * fps)

    frames = sorted(set(d["frame"] for d in detections))
    segments, seg_start, seg_end = [], frames[0], frames[0]

    for f in frames[1:]:
        if f - seg_end <= gap_frames:
            seg_end = f
        else:
            segments.append((seg_start, seg_end))
            seg_start = seg_end = f
    segments.append((seg_start, seg_end))

    segments = [(s, e) for s, e in segments if (e - s) >= min_frames]
    print(f"  → {len(segments)} shot segments")
    return segments


def extract_clips(video_path: Path, segments: list[tuple[int, int]],
                  fps: float, out_dir: Path, prefix: str) -> list[dict]:
    """Write one mp4 per segment with a small buffer."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = int(CLIP_BUFFER_SEC * fps)

    clip_info = []
    for i, (seg_start, seg_end) in enumerate(segments):
        clip_start = max(0, seg_start - buf)
        clip_end   = min(total_frames - 1, seg_end + buf)

        out_path = out_dir / f"{prefix}_clip_{i+1:03d}.mp4"
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        for _ in range(clip_end - clip_start + 1):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        writer.release()

        clip_info.append({
            "clip": out_path.name,
            "source_video": video_path.name,
            "seg_start_sec": round(seg_start / fps, 2),
            "seg_end_sec":   round(seg_end   / fps, 2),
            "clip_start_sec": round(clip_start / fps, 2),
            "clip_end_sec":   round(clip_end   / fps, 2),
        })
        print(f"  {out_path.name}: {clip_start/fps:.1f}s – {clip_end/fps:.1f}s")

    cap.release()
    return clip_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True,
                        help="Player folder, e.g. video/1_Cameron_Boozer")
    parser.add_argument("--weights", default=str(WEIGHTS))
    args = parser.parse_args()

    player_dir = Path(args.player)
    if not player_dir.exists():
        raise FileNotFoundError(f"Player folder not found: {player_dir}")

    out_dir = player_dir / "jump_shot_clips"
    # Clear existing clips
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    videos = sorted(v for v in player_dir.glob("*.mp4")
                    if not v.name.startswith("detected_"))
    if not videos:
        print("No mp4 files found.")
        return

    print(f"Player: {player_dir.name}")
    print(f"Videos: {len(videos)}")

    model = YOLO(args.weights)
    all_clips = []

    for video_path in videos:
        detections, fps = detect_jump_shots(video_path, model)
        segments = group_segments(detections, fps)
        if not segments:
            print("  No jump-shot segments found.")
            continue
        prefix = video_path.stem[:30].replace(" ", "_")
        clips = extract_clips(video_path, segments, fps, out_dir, prefix)
        all_clips.extend(clips)

    meta_path = out_dir / "detections.json"
    meta_path.write_text(json.dumps({"player": player_dir.name,
                                      "total_clips": len(all_clips),
                                      "clips": all_clips}, indent=2))
    print(f"\nDone. {len(all_clips)} clips total → {out_dir}")


if __name__ == "__main__":
    main()
