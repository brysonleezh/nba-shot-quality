"""
Build a Mechanics Fingerprint per prospect.

For each player directory with ≥ MIN_CLIPS pose JSON files, loads all
clip_*_pose.json files, aligns trajectories to rel_frame=0 (release),
computes mean ± std for knee_angle, elbow_angle, and body_lean across
frames [-30, +10], and saves mechanics_fingerprint.json in the player dir.

Usage:
    python video/compute_fingerprint.py
    python video/compute_fingerprint.py --min-clips 5
    python video/compute_fingerprint.py --player video/1_Cameron_Boozer
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


MIN_CLIPS   = 10
WIN_START   = -30
WIN_END     = 10
METRICS     = ("knee_angle", "elbow_angle", "body_lean")
VIDEO_ROOT  = Path(__file__).parent


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return super().default(o)


def _load_clips(player_dir: Path) -> list[dict]:
    clips_dir = player_dir / "jump_shot_clips"
    if not clips_dir.exists():
        return []
    return [
        json.loads(p.read_text())
        for p in sorted(clips_dir.glob("clip_*_pose.json"))
    ]


def compute_fingerprint(player_dir: Path, min_clips: int = MIN_CLIPS) -> dict | None:
    clips = _load_clips(player_dir)
    if len(clips) < min_clips:
        return None

    frames = list(range(WIN_START, WIN_END + 1))  # -30 … +10 inclusive
    n_frames = len(frames)
    frame_idx = {f: i for i, f in enumerate(frames)}

    # buckets[metric][frame_index] = list of float values across clips
    buckets: dict[str, list[list[float]]] = {
        m: [[] for _ in range(n_frames)] for m in METRICS
    }

    for clip in clips:
        for entry in clip.get("trajectory", []):
            rf = entry.get("rel_frame")
            if rf is None or rf not in frame_idx:
                continue
            fi = frame_idx[rf]
            for m in METRICS:
                v = entry.get(m)
                if v is not None:
                    buckets[m][fi].append(float(v))

    result: dict[str, dict] = {}
    for m in METRICS:
        means, stds, ns = [], [], []
        for fi in range(n_frames):
            vals = buckets[m][fi]
            if vals:
                means.append(round(float(np.mean(vals)), 3))
                stds.append(round(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0), 3))
                ns.append(len(vals))
            else:
                means.append(None)
                stds.append(None)
                ns.append(0)
        result[m] = {"mean": means, "std": stds, "n": ns}

    return {
        "player":  player_dir.name,
        "n_clips": len(clips),
        "window":  [WIN_START, WIN_END],
        "frames":  frames,
        **result,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute mechanics fingerprint per prospect")
    parser.add_argument("--player", type=Path, default=None,
                        help="Single player dir (default: all players under video/)")
    parser.add_argument("--min-clips", type=int, default=MIN_CLIPS,
                        help=f"Minimum pose JSON files required (default: {MIN_CLIPS})")
    args = parser.parse_args()

    if args.player:
        dirs = [args.player]
    else:
        dirs = sorted(p for p in VIDEO_ROOT.iterdir() if p.is_dir() and not p.name.startswith("."))

    processed = 0
    for player_dir in dirs:
        fp = compute_fingerprint(player_dir, args.min_clips)
        if fp is None:
            n = len(_load_clips(player_dir))
            print(f"  skip  {player_dir.name}  ({n} clips < {args.min_clips})")
            continue

        out = player_dir / "mechanics_fingerprint.json"
        out.write_text(json.dumps(fp, cls=_Enc, indent=2))
        print(f"  wrote {out}  (n_clips={fp['n_clips']})")
        processed += 1

    print(f"\nDone — {processed} fingerprint(s) written.")


if __name__ == "__main__":
    main()
