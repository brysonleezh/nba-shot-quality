"""
Compute DTW-based similarity between prospect mechanics fingerprints
and elite NBA shooter benchmark fingerprints.

Weights: knee_angle 30%, elbow_angle 40%, body_lean 30%.

Reads  : video/<player>/mechanics_fingerprint.json  (prospects + benchmarks)
Writes : data/similarity_scores.json

Usage:
    pip install dtaidistance
    python video/compute_similarity.py
    python video/compute_similarity.py --prospect video/1_Cameron_Boozer
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

try:
    from dtaidistance import dtw
    _HAS_DTW = True
except ImportError:
    _HAS_DTW = False
    print("WARNING: dtaidistance not installed — falling back to Euclidean distance.\n"
          "         pip install dtaidistance  for proper DTW similarity.")


VIDEO_ROOT   = Path(__file__).parent
DATA_ROOT    = VIDEO_ROOT.parent / "data"
BENCHMARK_ROOT = VIDEO_ROOT / "benchmarks"

WEIGHTS = {"knee_angle": 0.30, "elbow_angle": 0.40, "body_lean": 0.30}
METRICS = list(WEIGHTS.keys())


def _load_fingerprint(fp_path: Path) -> dict | None:
    if not fp_path.exists():
        return None
    return json.loads(fp_path.read_text())


def _mean_series(fp: dict, metric: str) -> np.ndarray:
    """Return mean trajectory as a float array; replace None with linear interp."""
    vals = np.array([v if v is not None else np.nan for v in fp[metric]["mean"]], dtype=float)
    # linear interpolation over NaN gaps; extrapolate edges with nearest
    nans = np.isnan(vals)
    if nans.all():
        return np.zeros(len(vals))
    if nans.any():
        idx = np.arange(len(vals))
        vals = np.interp(idx, idx[~nans], vals[~nans])
    return vals


def _dtw_dist(a: np.ndarray, b: np.ndarray) -> float:
    if _HAS_DTW:
        return float(dtw.distance_fast(a, b))
    # fallback: root mean squared difference (same frames, no warping)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _normalize(distances: dict[str, float]) -> dict[str, float]:
    """Min-max normalize per metric across all benchmark distances."""
    all_vals = list(distances.values())
    lo, hi = min(all_vals), max(all_vals)
    if hi == lo:
        return {k: 0.0 for k in distances}
    return {k: (v - lo) / (hi - lo) for k, v in distances.items()}


def score_prospect(prospect_fp: dict, benchmark_fps: list[dict]) -> list[dict]:
    """
    For each benchmark, compute weighted normalized DTW similarity [0, 100].
    100 = identical mechanics, 0 = most different among the benchmark set.
    """
    results = []
    for bench_fp in benchmark_fps:
        per_metric_dist: dict[str, float] = {}
        for m in METRICS:
            a = _mean_series(prospect_fp, m)
            b = _mean_series(bench_fp, m)
            per_metric_dist[m] = _dtw_dist(a, b)
        results.append({
            "benchmark":       bench_fp["player"],
            "per_metric_dist": per_metric_dist,
        })

    # normalize each metric across all benchmarks
    for m in METRICS:
        dists = {r["benchmark"]: r["per_metric_dist"][m] for r in results}
        normed = _normalize(dists)
        for r in results:
            r.setdefault("per_metric_norm", {})[m] = round(normed[r["benchmark"]], 4)

    # weighted composite similarity (inverted: 1 - normed_dist)
    for r in results:
        composite = sum(
            WEIGHTS[m] * (1.0 - r["per_metric_norm"][m])
            for m in METRICS
        )
        r["similarity"] = round(composite * 100, 1)  # 0–100

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results


def _find_fingerprints(root: Path) -> list[Path]:
    return sorted(root.glob("*/mechanics_fingerprint.json"))


def main():
    parser = argparse.ArgumentParser(description="Compute prospect-to-benchmark DTW similarity")
    parser.add_argument("--prospect", type=Path, default=None,
                        help="Single prospect dir (default: all under video/)")
    args = parser.parse_args()

    # Load benchmark fingerprints
    bench_fps = [_load_fingerprint(p) for p in _find_fingerprints(BENCHMARK_ROOT)]
    bench_fps = [fp for fp in bench_fps if fp is not None]
    if not bench_fps:
        print("No benchmark fingerprints found.")
        print("Run: python video/download_benchmarks.py")
        print("Then: python video/analyze_pose.py --player video/benchmarks/<id>")
        print("Then: python video/compute_fingerprint.py --min-clips 1")
        return

    print(f"Loaded {len(bench_fps)} benchmark(s): {[b['player'] for b in bench_fps]}")

    # Load prospect fingerprints
    if args.prospect:
        fp_path = args.prospect / "mechanics_fingerprint.json"
        prospect_fps = [_load_fingerprint(fp_path)]
    else:
        prospect_fps = [_load_fingerprint(p) for p in _find_fingerprints(VIDEO_ROOT)
                        if "benchmarks" not in str(p)]

    prospect_fps = [fp for fp in prospect_fps if fp is not None]
    if not prospect_fps:
        print("No prospect fingerprints found. Run compute_fingerprint.py first.")
        return

    output: list[dict] = []
    for pfp in prospect_fps:
        scores = score_prospect(pfp, bench_fps)
        most_similar = scores[0] if scores else None
        entry = {
            "prospect":      pfp["player"],
            "n_clips":       pfp["n_clips"],
            "most_similar":  most_similar["benchmark"] if most_similar else None,
            "top_score":     most_similar["similarity"] if most_similar else None,
            "benchmarks":    scores,
        }
        output.append(entry)
        print(f"  {pfp['player']:30s} → most similar: "
              f"{entry['most_similar']} ({entry['top_score']})")

    DATA_ROOT.mkdir(exist_ok=True)
    out_path = DATA_ROOT / "similarity_scores.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
