"""
RTMPose analysis on jump-shot clips.
Uses per-frame bbox from tracking.json; RTMPose runs top-down on the full frame
with the tracking bbox — it handles its own crop/padding (1.25×) internally.
Outputs per-frame metric trajectory + pose preview video.

Usage:
    python video/analyze_pose.py --player video/1_Cameron_Boozer
    python video/analyze_pose.py --player video/1_Cameron_Boozer --complexity 2
    python video/analyze_pose.py --player video/1_Cameron_Boozer --clips 14 19
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from rtmlib import RTMPose


class VideoReader:
    """Sequential frame reader via ffmpeg pipe — works on macOS with H.264."""

    def __init__(self, path: Path):
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "default=noprint_wrappers=1", str(path)],
            capture_output=True, text=True,
        )
        self.W = self.H = 0
        for line in probe.stdout.splitlines():
            if line.startswith("width="):
                self.W = int(line.split("=")[1])
            elif line.startswith("height="):
                self.H = int(line.split("=")[1])
        self._size = self.W * self.H * 3
        self._proc = subprocess.Popen(
            ["ffmpeg", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-loglevel", "quiet", "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        raw = self._proc.stdout.read(self._size)
        if len(raw) < self._size:
            return False, None
        return True, np.frombuffer(raw, dtype=np.uint8).reshape(self.H, self.W, 3).copy()

    def release(self):
        self._proc.kill()
        self._proc.wait()

CONF_THRESH       = 0.30   # RTMPose SimCC scores (lower scale than MediaPipe visibility)
WINDOW            = 30
CROP_PAD          = 0.25   # only used for preview rectangle
MIN_STABILITY_IOU = 0.30
MODEL_COMPLEXITY  = 1      # 0=lite, 1=balanced, 2=performance

_MODEL_URLS = {
    0: "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
    1: "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
    2: "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",
}
_MODEL_INPUT_SIZES = {0: (192, 256), 1: (192, 256), 2: (288, 384)}
_MODEL_LABELS      = {
    0: "rtmpose-s (lightweight)",
    1: "rtmpose-m (balanced)",
    2: "rtmpose-x (performance)",
}

# COCO-17 keypoint indices
KP = dict(
    nose=0,
    l_shoulder=5,  r_shoulder=6,
    l_elbow=7,     r_elbow=8,
    l_wrist=9,     r_wrist=10,
    l_hip=11,      r_hip=12,
    l_knee=13,     r_knee=14,
    l_ankle=15,    r_ankle=16,
)

SKELETON = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return super().default(o)

def _dump(obj, path, **kw):
    Path(path).write_text(json.dumps(obj, cls=_Enc, **kw))


def _get_model(complexity: int) -> RTMPose:
    print(f"Loading {_MODEL_LABELS[complexity]} (auto-download on first run)...")
    return RTMPose(
        onnx_model=_MODEL_URLS[complexity],
        model_input_size=_MODEL_INPUT_SIZES[complexity],
        backend="onnxruntime",
        device="cpu",
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def crop_region(bbox, W, H, pad=CROP_PAD):
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    return (max(0, int(x1 - bw*pad)), max(0, int(y1 - bh*pad)),
            min(W, int(x2 + bw*pad)), min(H, int(y2 + bh*pad)))


def bbox_iou(a, b) -> float:
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def angle_at(a, b, c) -> float | None:
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc = a - b, c - b
    n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
    if n1 < 1e-6 or n2 < 1e-6: return None
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (n1*n2), -1, 1))))


def body_lean_deg(sh, hp) -> float | None:
    dx, dy = sh[0] - hp[0], hp[1] - sh[1]
    return float(np.degrees(np.arctan2(dx, dy))) if abs(dy) > 1e-6 else None


def draw_skeleton(img, kps, vis, shoot_kps):
    for i, j in SKELETON:
        if vis[i] > CONF_THRESH and vis[j] > CONF_THRESH:
            color = (0, 255, 255) if {i, j} <= shoot_kps else (0, 200, 0)
            cv2.line(img, (int(kps[i][0]), int(kps[i][1])),
                         (int(kps[j][0]), int(kps[j][1])), color, 2)
    for i in range(len(kps)):
        if vis[i] > CONF_THRESH:
            cv2.circle(img, (int(kps[i][0]), int(kps[i][1])), 4,
                       (0, 0, 255) if i in shoot_kps else (0, 255, 0), -1)


# ── analysis ──────────────────────────────────────────────────────────────────

MIN_POSE_FRAMES = 10


def _delete_clip(clip_path: Path, tracking_json: Path) -> None:
    """Delete all files associated with a clip (original + pose outputs)."""
    stem   = clip_path.stem
    parent = clip_path.parent
    for f in [
        clip_path,
        tracking_json,
        parent / f"{stem}_preview.mp4",
        parent / f"{stem}_pose.json",
        parent / f"{stem}_pose_preview.mp4",
        parent / f"{stem}_pose_preview_web.mp4",
    ]:
        if f.exists():
            f.unlink()


def analyze_clip(clip_path: Path, tracking_json: Path,
                 hand: str, window: int, pad: float,
                 pose_estimator: RTMPose,
                 min_pose_frames: int = MIN_POSE_FRAMES) -> dict | None:

    meta       = json.loads(tracking_json.read_text())
    fps        = meta["fps"]
    clip_start = meta["clip_start_frame"]
    anchor_abs = meta["anchor_frame"]
    anchor_off = anchor_abs - clip_start

    if hand == "r":
        s_sh, s_el, s_wr  = KP["r_shoulder"], KP["r_elbow"], KP["r_wrist"]
        g_wr               = KP["l_wrist"]
        s_hip, s_kn, s_an = KP["r_hip"], KP["r_knee"], KP["r_ankle"]
    else:
        s_sh, s_el, s_wr  = KP["l_shoulder"], KP["l_elbow"], KP["l_wrist"]
        g_wr               = KP["r_wrist"]
        s_hip, s_kn, s_an = KP["l_hip"], KP["l_knee"], KP["l_ankle"]
    shoot_kps = {s_sh, s_el, s_wr}

    cap     = VideoReader(clip_path)
    W       = cap.W
    H       = cap.H
    fmeta   = meta["frames"]
    n_total = len(fmeta)

    win_start = max(0, anchor_off - window)
    win_end   = min(n_total - 1, anchor_off + window)

    print(f"  {clip_path.name}  anchor={anchor_off}({anchor_off/fps:.2f}s)  "
          f"window=[{win_start},{win_end}]")

    trajectory: list[dict] = []
    prev_bbox: list | None = None
    prev_kps:  np.ndarray | None = None
    prev_vis:  np.ndarray | None = None

    # ── Pass 1: inference only (no rendering) ────────────────────────────────
    # RTMPose top-down: pass full frame + tracking bbox.
    # RTMPose handles crop + 1.25× padding internally and returns keypoints
    # in full-frame pixel coordinates — no manual crop/uncrop needed.
    for f_off in range(n_total):
        ret, frame = cap.read()
        if not ret:
            break

        fm     = fmeta[f_off]
        in_win = win_start <= f_off <= win_end
        if not in_win:
            continue

        bbox = fm.get("bbox")
        if bbox is None:
            continue

        if prev_bbox is not None and bbox_iou(prev_bbox, bbox) < MIN_STABILITY_IOU:
            continue
        prev_bbox = bbox

        bx1, by1, bx2, by2 = [int(v) for v in bbox]

        try:
            kps_all, vis_all = pose_estimator(frame, bboxes=[[bx1, by1, bx2, by2]])
        except Exception:
            continue

        if len(kps_all) == 0:
            continue

        kps = kps_all[0].copy()  # (17, 2) full-frame pixel coords
        vis = vis_all[0].copy()  # (17,) confidence scores

        torso_idx = [KP["l_shoulder"], KP["r_shoulder"],
                     KP["l_hip"],      KP["r_hip"]]

        # Validation 1: torso center inside tracking bbox
        torso_pts = [kps[i] for i in torso_idx if vis[i] > CONF_THRESH]
        if torso_pts:
            tc_x = float(np.mean([p[0] for p in torso_pts]))
            tc_y = float(np.mean([p[1] for p in torso_pts]))
            mx = (bx2 - bx1) * 0.20
            my = (by2 - by1) * 0.20
            if not (bx1 - mx <= tc_x <= bx2 + mx and
                    by1 - my <= tc_y <= by2 + my):
                prev_kps = prev_vis = None
                continue

        # Validation 2: min visible shooting joints
        shoot_joints = [s_sh, s_el, s_wr, s_hip, s_kn]
        if sum(1 for i in shoot_joints if vis[i] > CONF_THRESH) < 3:
            prev_kps = prev_vis = None
            continue

        # Validation 3: anatomical ordering (hips below shoulders, knees below hips)
        anat_ok = True
        for sh_i, hp_i in [(KP["l_shoulder"], KP["l_hip"]),
                            (KP["r_shoulder"], KP["r_hip"])]:
            if (vis[hp_i] > CONF_THRESH and vis[sh_i] > CONF_THRESH and
                    kps[hp_i][1] < kps[sh_i][1] - 5):
                anat_ok = False
        if (vis[s_kn] > CONF_THRESH and vis[s_hip] > CONF_THRESH and
                kps[s_kn][1] < kps[s_hip][1] - 5):
            anat_ok = False
        if not anat_ok:
            prev_kps = prev_vis = None
            continue

        # Validation 4: keypoint continuity
        if prev_kps is not None and prev_vis is not None:
            max_jump = (bx2 - bx1) * 0.40
            jumps = [np.linalg.norm(kps[i] - prev_kps[i])
                     for i in torso_idx
                     if vis[i] > CONF_THRESH and prev_vis[i] > CONF_THRESH]
            if jumps and float(np.mean(jumps)) > max_jump:
                prev_kps = prev_vis = None
                continue

        prev_kps = kps.copy()
        prev_vis = vis.copy()

        def v(idx):
            return kps[idx].tolist() if vis[idx] > CONF_THRESH else None

        pts = {i: v(i) for i in [s_sh, s_el, s_wr, g_wr,
                                  s_hip, s_kn, s_an,
                                  KP["l_shoulder"], KP["r_shoulder"],
                                  KP["l_hip"],      KP["r_hip"]]}

        elbow = angle_at(pts[s_sh], pts[s_el], pts[s_wr]) \
                if all(pts[k] for k in [s_sh, s_el, s_wr]) else None
        knee  = angle_at(pts[s_hip], pts[s_kn], pts[s_an]) \
                if all(pts[k] for k in [s_hip, s_kn, s_an]) else None
        wrist_norm = (1.0 - float(kps[s_wr][1]) / H) \
                     if vis[s_wr] > CONF_THRESH else None
        sh_l, sh_r = pts[KP["l_shoulder"]], pts[KP["r_shoulder"]]
        hp_l, hp_r = pts[KP["l_hip"]],      pts[KP["r_hip"]]
        lean = body_lean_deg(
            [(sh_l[0]+sh_r[0])/2, (sh_l[1]+sh_r[1])/2],
            [(hp_l[0]+hp_r[0])/2, (hp_l[1]+hp_r[1])/2],
        ) if sh_l and sh_r and hp_l and hp_r else None
        g, s_pt = pts[g_wr], pts[s_wr]
        guide_sep = float(np.linalg.norm(np.array(s_pt) - np.array(g))) / W \
                    if g and s_pt else None

        trajectory.append({
            "frame_offset": f_off,
            "rel_frame":    f_off - anchor_off,
            "sec":          round(fm["sec"], 3),
            "elbow_angle":  round(elbow, 1)      if elbow      else None,
            "knee_angle":   round(knee, 1)       if knee       else None,
            "wrist_height": round(wrist_norm, 3) if wrist_norm else None,
            "body_lean":    round(lean, 1)       if lean       else None,
            "guide_sep":    round(guide_sep, 3)  if guide_sep  else None,
            "keypoints":    [[float(x), float(y)] for x, y in kps],
            "kp_conf":      [float(c) for c in vis],
        })

    cap.release()

    # ── Trajectory outlier removal (player-switch block detection) ───────────
    # A bbox-induced person switch creates a bounded block: large torso jump IN
    # at entry, large torso jump OUT at exit, bad frames in between.
    # Strategy: find every "large-step" transition, pair consecutive ones up,
    # and remove any block of ≤ MAX_BLOCK_FRAMES frames between them.
    OUTLIER_ABS_PX    = 35
    OUTLIER_STEP_MULT = 3.0
    MAX_BLOCK_FRAMES  = 25
    n_traj = len(trajectory)

    if n_traj >= 5:
        torso_track = [KP["l_shoulder"], KP["r_shoulder"],
                       KP["l_hip"],      KP["r_hip"]]
        kps_tmp = np.array([r["keypoints"] for r in trajectory])  # (n, 17, 2)
        steps = [
            float(np.mean(np.linalg.norm(
                kps_tmp[i][torso_track] - kps_tmp[i-1][torso_track], axis=1)))
            for i in range(1, n_traj)
        ]
        med_step = float(np.median(steps))
        thresh   = max(OUTLIER_ABS_PX, med_step * OUTLIER_STEP_MULT)

        jump_idxs = [i for i, s in enumerate(steps) if s > thresh]
        to_remove: set = set()
        for k in range(len(jump_idxs) - 1):
            j_in  = jump_idxs[k]
            j_out = jump_idxs[k + 1]
            block = range(j_in + 1, j_out + 1)
            if 0 < len(block) <= MAX_BLOCK_FRAMES:
                to_remove.update(block)
        removed = len(to_remove)
        if removed:
            print(f"    outlier removal: dropped {removed} frame(s)")
        if to_remove:
            trajectory = [r for i, r in enumerate(trajectory) if i not in to_remove]
            n_traj = len(trajectory)

    if n_traj < min_pose_frames:
        print(f"  → auto-deleted ({n_traj} pose frames < min {min_pose_frames})")
        return None

    # ── Smooth keypoints + metrics (3-frame median filter) ───────────────────
    if n_traj >= 3:
        kps_arr    = np.array([r["keypoints"] for r in trajectory])  # (n, 17, 2)
        kps_smooth = kps_arr.copy()
        for i in range(1, n_traj - 1):
            kps_smooth[i] = np.median(
                np.stack([kps_arr[i-1], kps_arr[i], kps_arr[i+1]]), axis=0)
        for i, r in enumerate(trajectory):
            r["keypoints"] = [[float(x), float(y)] for x, y in kps_smooth[i]]

        for key in ["elbow_angle", "knee_angle", "body_lean"]:
            idxs = [i for i, r in enumerate(trajectory) if r.get(key) is not None]
            if len(idxs) < 3:
                continue
            raw = [trajectory[i][key] for i in idxs]
            for j in range(1, len(idxs) - 1):
                trajectory[idxs[j]][key] = round(
                    float(np.median([raw[j-1], raw[j], raw[j+1]])), 1)

    # ── Pass 2: render preview with smoothed keypoints ────────────────────────
    traj_by_frame = {r["frame_offset"]: r for r in trajectory}
    preview_path  = clip_path.parent / (clip_path.stem + "_pose_preview.mp4")
    cap2   = VideoReader(clip_path)
    writer = cv2.VideoWriter(str(preview_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for f_off in range(n_total):
        ret, frame = cap2.read()
        if not ret:
            break
        fm      = fmeta[f_off]
        preview = frame.copy()
        in_win  = win_start <= f_off <= win_end

        r = traj_by_frame.get(f_off)
        if r:
            kps = np.array(r["keypoints"])
            vis = np.array(r["kp_conf"])
            bbox = fm.get("bbox")
            if bbox:
                cx1, cy1, cx2, cy2 = crop_region(bbox, W, H, pad)
                cv2.rectangle(preview, (cx1, cy1), (cx2, cy2), (200, 200, 0), 1)
            draw_skeleton(preview, kps, vis, shoot_kps)
            y0 = 50
            for txt in filter(None, [
                f"Elbow: {r['elbow_angle']:.0f}" if r.get("elbow_angle") else None,
                f"Knee:  {r['knee_angle']:.0f}"  if r.get("knee_angle")  else None,
                f"Lean:  {r['body_lean']:+.1f}"  if r.get("body_lean")   else None,
            ]):
                cv2.putText(preview, txt, (20, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
                y0 += 35

        label = "* " if in_win else "  "
        cv2.putText(preview, f"{label}{clip_path.stem}  {fm['sec']:.2f}s",
                    (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
        writer.write(preview)

    cap2.release()
    writer.release()

    def vals(k): return [r[k] for r in trajectory if r.get(k) is not None]

    ev, kv, lv, wv = vals("elbow_angle"), vals("knee_angle"), vals("body_lean"), vals("wrist_height")
    summary = {
        "clip":                    clip_path.name,
        "fps":                     fps,
        "anchor_offset":           anchor_off,
        "anchor_sec":              round(anchor_off / fps, 3),
        "window":                  [win_start, win_end],
        "window_frames_with_pose": len(trajectory),
        "trajectory":              trajectory,
        "stats": {
            "elbow":     {"min": round(float(min(ev)), 1), "max": round(float(max(ev)), 1),
                          "mean": round(float(np.mean(ev)), 1)} if ev else None,
            "knee_min":  round(float(min(kv)), 1)  if kv else None,
            "wrist_max": round(float(max(wv)), 3)  if wv else None,
            "lean":      {"mean": round(float(np.mean(lv)), 1),
                          "std":  round(float(np.std(lv)),  1)} if lv else None,
        },
    }

    _dump(summary, clip_path.parent / (clip_path.stem + "_pose.json"), indent=2)

    # Re-encode pose preview to H.264 for browser playback
    web_path = clip_path.parent / (clip_path.stem + "_pose_preview_web.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(preview_path),
             "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
             "-movflags", "+faststart", "-an", str(web_path)],
            check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    print(f"  {clip_path.stem}  {len(trajectory)} pose frames  "
          f"elbow={summary['stats']['elbow']}  lean={summary['stats']['lean']}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player",     required=True)
    parser.add_argument("--hand",       default="r", choices=["r", "l"])
    parser.add_argument("--window",     type=int,   default=WINDOW)
    parser.add_argument("--pad",        type=float, default=CROP_PAD)
    parser.add_argument("--complexity", type=int,   default=MODEL_COMPLEXITY,
                        choices=[0, 1, 2],
                        help="0=rtmpose-s, 1=rtmpose-m (default), 2=rtmpose-x")
    parser.add_argument("--clips",      nargs="*",  type=int)
    parser.add_argument("--min-frames", type=int,   default=MIN_POSE_FRAMES,
                        help="auto-delete clips with fewer pose frames than this after cleanup")
    args = parser.parse_args()

    pose_estimator = _get_model(args.complexity)
    clips_dir      = Path(args.player) / "jump_shot_clips"

    if args.clips:
        paths = [clips_dir / f"clip_{n:03d}.mp4" for n in args.clips]
    else:
        paths = sorted(p for p in clips_dir.glob("clip_*.mp4")
                       if "preview" not in p.stem and "pose" not in p.stem
                       and "zoom" not in p.stem)

    print(f"Player: {Path(args.player).name}  |  {len(paths)} clips  "
          f"|  hand={args.hand}  |  window=+/-{args.window}  "
          f"|  complexity={args.complexity} ({_MODEL_LABELS[args.complexity]})\n")

    summaries = []
    for p in paths:
        tj = p.parent / (p.stem + "_tracking.json")
        if not p.exists() or not tj.exists():
            print(f"  missing {p.name}, skipping")
            continue
        if (p.parent / (p.stem + "_pose_preview.mp4")).exists():
            print(f"  {p.name}  already done, skipping")
            continue
        result = analyze_clip(p, tj, args.hand, args.window, args.pad,
                              pose_estimator, args.min_frames)
        if result is None:
            _delete_clip(p, tj)
        else:
            summaries.append(result)

    if not summaries:
        return

    _dump({"player": Path(args.player).name,
           "clips_analyzed": len(summaries),
           "clips": [s["clip"] for s in summaries]},
          clips_dir / "pose_summary.json", indent=2)
    print(f"\nDone. {len(summaries)} clips → {clips_dir}")


if __name__ == "__main__":
    main()
