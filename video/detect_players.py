"""
Step 1: Run YOLO on Cameron Boozer videos and output annotated video
with numbered bounding boxes for all detected players.

Usage:
    python video/detect_players.py

Outputs:
    video/1_Cameron_Boozer/detected_<video_name>.mp4
"""

from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_DIR   = Path("video/1_Cameron_Boozer")
OUTPUT_DIR  = VIDEO_DIR
SAMPLE_EVERY = 2        # process every Nth frame (2 = 30fps from 60fps)
CONF_THRESH  = 0.4      # YOLO confidence threshold
PERSON_CLASS = 0        # COCO class 0 = person

model = YOLO("yolov8n.pt")  # nano model, auto-downloads ~6MB


def draw_boxes(frame, boxes):
    """Draw numbered bounding boxes on frame."""
    overlay = frame.copy()
    for i, (x1, y1, x2, y2, conf) in enumerate(boxes):
        color = (0, 200, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"#{i}  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return overlay


def process_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps_in  = cap.get(cv2.CAP_PROP_FPS)
    fps_out = fps_in / SAMPLE_EVERY
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = OUTPUT_DIR / f"detected_{video_path.stem}.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_out, (w, h)
    )

    frame_idx = 0
    written   = 0
    print(f"\nProcessing: {video_path.name}")
    print(f"  {total} frames @ {fps_in:.0f}fps → sampling every {SAMPLE_EVERY} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SAMPLE_EVERY == 0:
            results = model(frame, classes=[PERSON_CLASS],
                            conf=CONF_THRESH, verbose=False)[0]

            boxes = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                area = (x2 - x1) * (y2 - y1)
                if area > 2000:   # filter out tiny false positives
                    boxes.append((x1, y1, x2, y2, conf))

            # Sort by x-position so numbering is consistent left→right
            boxes.sort(key=lambda b: b[0])

            annotated = draw_boxes(frame, boxes)

            # Overlay frame timestamp
            ts = frame_idx / fps_in
            cv2.putText(annotated, f"{ts:.1f}s  frame {frame_idx}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)

            writer.write(annotated)
            written += 1

            if written % 100 == 0:
                print(f"  {written} frames written ({ts:.0f}s / {total/fps_in:.0f}s)")

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"  Done → {out_path}")
    return out_path


if __name__ == "__main__":
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    videos = [v for v in videos if not v.name.startswith("detected_")]

    if not videos:
        print("No mp4 files found in", VIDEO_DIR)
    else:
        for v in videos:
            process_video(v)
        print("\nAll done. Open the detected_*.mp4 files to review player boxes.")
