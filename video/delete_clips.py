"""
Delete specific jump-shot clips by stem name and update index.json.

Usage:
    python video/delete_clips.py --player video/1_Cameron_Boozer \
        --clips clip_20250513_143022_456789 clip_20250513_144512_123456
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True,
                        help="Player folder, e.g. video/1_Cameron_Boozer")
    parser.add_argument("--clips", nargs="+", required=True,
                        help="Clip stems to delete, e.g. clip_20250513_143022_456789")
    args = parser.parse_args()

    clips_dir = Path(args.player) / "jump_shot_clips"
    if not clips_dir.exists():
        raise FileNotFoundError(f"No jump_shot_clips folder found at {clips_dir}")

    stems = list(dict.fromkeys(args.clips))  # deduplicate, preserve order
    for stem in stems:
        for suffix in [".mp4", "_preview.mp4", "_tracking.json",
                       "_pose.json", "_pose_preview.mp4"]:
            p = clips_dir / f"{stem}{suffix}"
            if p.exists():
                p.unlink()
                print(f"  deleted  {p.name}")
            else:
                print(f"  missing  {p.name}")

    index_path = clips_dir / "index.json"
    if index_path.exists():
        index  = json.loads(index_path.read_text())
        before = len(index.get("clips", []))
        index["clips"] = [c for c in index.get("clips", [])
                          if Path(c["clip"]).stem not in stems]
        index["total_clips"] = len(index["clips"])
        index_path.write_text(json.dumps(index, indent=2))
        print(f"\nindex.json: {before} → {index['total_clips']} clips")

    print(f"\nDone. Removed {len(stems)} clip(s).")


if __name__ == "__main__":
    main()
