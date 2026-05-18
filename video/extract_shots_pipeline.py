"""
Complete pipeline: video → jump-shot clips + player tracking

Usage:
    python video/extract_shots_pipeline.py --player video/1_Cameron_Boozer

Output per clip:
    jump_shot_clips/clip_001.mp4              - clean clip (for MediaPipe)
    jump_shot_clips/clip_001_tracking.json    - frame-by-frame player bbox
    jump_shot_clips/clip_001_preview.mp4      - annotated with tracked bbox
    jump_shot_clips/index.json                - clip index
"""

from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

WEIGHTS           = Path("models/action_classifier/weights/best.pt")
JUMP_CLASS        = 6      # player-jump-shot
PLAYER_CLASS      = 3      # player
CONF_THRESH       = 0.25
SAMPLE_EVERY      = 2
GAP_TOLERANCE_SEC = 2.0    # detections within 2s → same shot event
MIN_SEG_SEC       = 0.15
CLIP_BEFORE_SEC   = 1.0    # 1s before first detection
CLIP_AFTER_SEC    = 2.0    # 2s after first detection
IOU_THRESH        = 0.25   # min IoU to keep tracking same player


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / area


def scan_video(video_path: Path, model: YOLO) -> tuple[list, float, int]:
    """Scan video (sampled) and return jump-shot detections, fps, total frames."""
    cap = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detections = []
    frame_idx = 0

    print(f"  Scanning: {video_path.name}")
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
                        "conf":  round(float(box.conf[0]), 3),
                        "bbox":  list(map(int, box.xyxy[0])),
                    })
        frame_idx += 1
        if frame_idx % 2000 == 0:
            print(f"    {frame_idx}/{total}  detections: {len(detections)}")

    cap.release()
    print(f"    {len(detections)} jump-shot detections")
    return detections, fps, total


def cluster_detections(detections: list, fps: float) -> list[dict]:
    """Group nearby detections into one shot event per cluster."""
    if not detections:
        return []

    gap_frames = int(GAP_TOLERANCE_SEC * fps)
    min_frames = int(MIN_SEG_SEC * fps)

    frames = sorted(set(d["frame"] for d in detections))
    seg_start = seg_end = frames[0]
    raw_clusters = []

    for f in frames[1:]:
        if f - seg_end <= gap_frames:
            seg_end = f
        else:
            raw_clusters.append((seg_start, seg_end))
            seg_start = seg_end = f
    raw_clusters.append((seg_start, seg_end))

    clusters = []
    for s, e in raw_clusters:
        if e - s < min_frames:
            continue
        dets = [d for d in detections if s <= d["frame"] <= e]
        best = max(dets, key=lambda d: d["conf"])
        clusters.append({
            "anchor_frame": s,
            "anchor_bbox":  best["bbox"],
            "anchor_conf":  best["conf"],
        })

    print(f"    {len(clusters)} shot clusters")
    return clusters


def track_player(frame_dets: dict, anchor: int, anchor_bbox: list) -> dict:
    """
    Return {frame_idx: [x1,y1,x2,y2]} for all sampled frames, then interpolate.
    Uses anchor_bbox to seed the tracker, IoU-matches forward and backward.
    """
    sampled = sorted(frame_dets.keys())
    tracked = {}

    # Seed at anchor
    candidates = frame_dets.get(anchor, {}).get("players", [])
    js = frame_dets.get(anchor, {}).get("jump_shot")
    if js:
        candidates = candidates + [js["bbox"]]
    if candidates:
        best = max(candidates, key=lambda b: iou(b, anchor_bbox))
        tracked[anchor] = best if iou(best, anchor_bbox) >= IOU_THRESH else anchor_bbox
    else:
        tracked[anchor] = anchor_bbox

    # Forward
    last = tracked[anchor]
    for f in (x for x in sampled if x > anchor):
        candidates = frame_dets[f].get("players", [])
        js = frame_dets[f].get("jump_shot")
        if js:
            candidates = candidates + [js["bbox"]]
        if candidates:
            best = max(candidates, key=lambda b: iou(b, last))
            if iou(best, last) >= IOU_THRESH:
                last = best
        tracked[f] = last

    # Backward
    last = tracked[anchor]
    for f in (x for x in reversed(sampled) if x < anchor):
        candidates = frame_dets[f].get("players", [])
        js = frame_dets[f].get("jump_shot")
        if js:
            candidates = candidates + [js["bbox"]]
        if candidates:
            best = max(candidates, key=lambda b: iou(b, last))
            if iou(best, last) >= IOU_THRESH:
                last = best
        tracked[f] = last

    # Linear interpolation for non-sampled frames
    full = {}
    keys = sorted(tracked)
    for i in range(len(keys) - 1):
        f1, f2 = keys[i], keys[i + 1]
        b1, b2 = tracked[f1], tracked[f2]
        for f in range(f1, f2 + 1):
            t = (f - f1) / (f2 - f1) if f2 > f1 else 0
            full[f] = [int(b1[j] + t * (b2[j] - b1[j])) for j in range(4)]
    if keys:
        full[keys[-1]] = tracked[keys[-1]]
    return full


def extract_clip(video_path: Path, cluster: dict, fps: float, total_frames: int,
                 model: YOLO, out_dir: Path) -> dict:
    """Run detection pass on clip, track player, write mp4 + json + preview."""
    anchor      = cluster["anchor_frame"]
    anchor_bbox = cluster["anchor_bbox"]
    clip_start  = max(0, anchor - int(CLIP_BEFORE_SEC * fps))
    clip_end    = min(total_frames - 1, anchor + int(CLIP_AFTER_SEC * fps))

    # --- Detection pass (sampled) ---
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
    frame_dets = {}
    f = clip_start

    while f <= clip_end:
        ret, frame = cap.read()
        if not ret:
            break
        if (f - clip_start) % SAMPLE_EVERY == 0:
            results = model(frame, conf=CONF_THRESH, verbose=False)[0]
            players, jump_shot = [], None
            for box in results.boxes:
                cls  = int(box.cls[0])
                bbox = list(map(int, box.xyxy[0]))
                conf = float(box.conf[0])
                if cls == PLAYER_CLASS:
                    players.append(bbox)
                elif cls == JUMP_CLASS:
                    if jump_shot is None or conf > jump_shot["conf"]:
                        jump_shot = {"bbox": bbox, "conf": round(conf, 3)}
            frame_dets[f] = {"players": players, "jump_shot": jump_shot}
        f += 1
    cap.release()

    # --- Track player ---
    full_tracking = track_player(frame_dets, anchor, anchor_bbox)

    # --- Write clip + preview ---
    ts           = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem         = f"clip_{ts}"
    clip_path    = out_dir / f"{stem}.mp4"
    preview_path = out_dir / f"{stem}_preview.mp4"

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w_clean   = cv2.VideoWriter(str(clip_path),    fourcc, fps, (W, H))
    w_preview = cv2.VideoWriter(str(preview_path), fourcc, fps, (W, H))

    frames_meta = []
    f = clip_start
    while f <= clip_end:
        ret, frame = cap.read()
        if not ret:
            break
        w_clean.write(frame)

        preview  = frame.copy()
        bbox     = full_tracking.get(f)
        is_shot  = f in frame_dets and frame_dets[f].get("jump_shot") is not None
        color    = (0, 255, 0) if is_shot else (0, 165, 255)

        if bbox:
            cv2.rectangle(preview, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
            label = "JUMP SHOT" if is_shot else "tracking"
            cv2.putText(preview, label, (bbox[0], max(bbox[1] - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        sec = (f - clip_start) / fps
        cv2.putText(preview, f"{stem}  {sec:.1f}s",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        w_preview.write(preview)

        frames_meta.append({
            "frame":             f,
            "sec":               round(sec, 3),
            "bbox":              bbox,
            "jump_shot_detected": is_shot,
        })
        f += 1

    cap.release()
    w_clean.release()
    w_preview.release()

    # --- Tracking JSON ---
    tracking = {
        "clip":             clip_path.name,
        "source_video":     video_path.name,
        "fps":              fps,
        "clip_start_frame": clip_start,
        "clip_end_frame":   clip_end,
        "anchor_frame":     anchor,
        "anchor_sec":       round((anchor - clip_start) / fps, 2),
        "frames":           frames_meta,
    }
    (out_dir / f"{stem}_tracking.json").write_text(json.dumps(tracking, indent=2))

    print(f"  {stem}  {clip_start/fps:.1f}s–{clip_end/fps:.1f}s  "
          f"({video_path.stem[:35]})")
    return {"clip": clip_path.name, "source_video": video_path.name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player",  required=True,
                        help="Player folder, e.g. video/1_Cameron_Boozer")
    parser.add_argument("--weights", default=str(WEIGHTS))
    args = parser.parse_args()

    player_dir = Path(args.player)
    out_dir    = player_dir / "jump_shot_clips"
    out_dir.mkdir(exist_ok=True)

    # Load existing index to append to
    index_path = out_dir / "index.json"
    if index_path.exists():
        existing = json.loads(index_path.read_text())
        all_info = existing.get("clips", [])
        # processed_sources tracks every video ever scanned, independent of review deletions
        already_processed = set(existing.get("processed_sources", []))
        if not already_processed:
            # backfill from clip entries for indexes written before this field existed
            already_processed = {c["source_video"] for c in all_info}
    else:
        all_info = []
        already_processed = set()

    videos = sorted(v for v in player_dir.glob("*.mp4")
                    if not v.stem.startswith(("detected_", "annotated_"))
                    and "jump_shot_clips" not in str(v))
    new_videos = [v for v in videos if v.name not in already_processed]
    print(f"Player: {player_dir.name} | {len(videos)} videos "
          f"({len(new_videos)} new, {len(videos)-len(new_videos)} already processed)\n")

    if not new_videos:
        print("No new videos to process.")
        return

    model    = YOLO(args.weights)
    new_info = []

    for video_path in new_videos:
        dets, fps, total = scan_video(video_path, model)
        clusters         = cluster_detections(dets, fps)
        for cluster in clusters:
            info = extract_clip(video_path, cluster, fps, total, model, out_dir)
            new_info.append(info)
        already_processed.add(video_path.name)

    all_info.extend(new_info)
    index_path.write_text(json.dumps({
        "player":            player_dir.name,
        "total_clips":       len(all_info),
        "processed_sources": sorted(already_processed),
        "clips":             all_info,
    }, indent=2))
    print(f"\nDone. +{len(new_info)} new clips ({len(all_info)} total) → {out_dir}")


if __name__ == "__main__":
    main()
