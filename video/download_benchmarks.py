"""
Download elite NBA shooter highlight clips for benchmark fingerprinting.

Each benchmark player maps to a curated YouTube URL with clean jump-shot
footage. yt-dlp downloads the best ≤720p mp4; clips are saved under
video/benchmarks/<player_id>/raw/<title>.mp4.

After downloading, run analyze_pose.py on each benchmark player dir, then
compute_fingerprint.py --min-clips 1 to generate benchmark fingerprints.

Usage:
    python video/download_benchmarks.py
    python video/download_benchmarks.py --player curry
    python video/download_benchmarks.py --dry-run
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path

VIDEO_ROOT = Path(__file__).parent

# Curated single-shooter highlight reels — one URL per benchmark player.
# Each URL should contain many isolated jump shots of that player only.
BENCHMARKS: list[dict] = [
    {
        "id":   "curry",
        "name": "Stephen Curry",
        "urls": [
            "https://www.youtube.com/watch?v=GrinCPtLbzg",  # Chase Center jump shot workout
        ],
    },
    {
        "id":   "thompson",
        "name": "Klay Thompson",
        "urls": [
            "https://www.youtube.com/watch?v=t_AlDkBJuR4",  # shooting workout footage
        ],
    },
    {
        "id":   "durant",
        "name": "Kevin Durant",
        "urls": [
            "https://www.youtube.com/watch?v=ixJOV-QAaO4",  # full shooting workout
        ],
    },
    {
        "id":   "tatum",
        "name": "Jayson Tatum",
        "urls": [
            "https://www.youtube.com/watch?v=939F_rr4DUE",  # Team USA shooting workout
        ],
    },
    {
        "id":   "lillard",
        "name": "Damian Lillard",
        "urls": [
            "https://www.youtube.com/watch?v=c26O1sFNyQg",  # pregame jump shot workout
        ],
    },
]


def player_dir(bench_id: str) -> Path:
    return VIDEO_ROOT / "benchmarks" / bench_id


def _node_path() -> str | None:
    import shutil
    return shutil.which("node")


def download(bench: dict, dry_run: bool = False) -> None:
    pdir = player_dir(bench["id"])
    raw  = pdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    node = _node_path()
    for url in bench["urls"]:
        cmd = ["yt-dlp"]
        if node:
            cmd += ["--js-runtimes", f"node:{node}"]
        cmd += [
            "--cookies-from-browser", "safari",
            "--format", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
            "--merge-output-format", "mp4",
            "--output", str(raw / "%(title)s.%(ext)s"),
            "--no-playlist",
            url,
        ]
        print(f"  {'[dry-run] ' if dry_run else ''}download → {raw.relative_to(VIDEO_ROOT.parent)}")
        print(f"    {url}")
        if not dry_run:
            subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Download NBA benchmark shooting highlights")
    parser.add_argument("--player", default=None,
                        help="Single benchmark id (curry/thompson/durant/tatum/lillard)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running them")
    args = parser.parse_args()

    targets = (
        [b for b in BENCHMARKS if b["id"] == args.player]
        if args.player else BENCHMARKS
    )
    if not targets:
        parser.error(f"Unknown player id '{args.player}'. "
                     f"Choose from: {[b['id'] for b in BENCHMARKS]}")

    for bench in targets:
        print(f"\n── {bench['name']} ({bench['id']}) ──")
        download(bench, dry_run=args.dry_run)

    print("\nDone.")
    if not args.dry_run:
        print("\nNext steps:")
        print("  1. python video/analyze_pose.py --player video/benchmarks/<id>")
        print("     (repeat for each benchmark player)")
        print("  2. python video/compute_fingerprint.py --min-clips 1")
        print("     (generates fingerprints for benchmarks too)")


if __name__ == "__main__":
    main()
