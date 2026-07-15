"""
p1_mstate_tracking/visualization/random_aggregate.py

Aggregates the multi-seed random-control runs under results/p1_random/
into one (mean, std) summary per (model, prompt), so the rest of the
package can plot a "random" line as a band over N seeds instead of
whichever single seed happened to be discovered.

This script does no new parsing — it reuses the same per-layer series
extractors series.py already has for trained runs, and just calls them
once per seed and folds the results together. If you add a new per-layer
metric to series.py and want it aggregated too, add its extractor to
SERIES_EXTRACTORS below; nothing else needs to change.

This module is normally invoked automatically as part of the single
`python -m p1_mstate_tracking.visualization` CLI entry point (see cli.py /
pipeline.py) — every --random_seed_dirs entry gets aggregated as part of
that one run. The CLI below is for inspecting or precomputing the
aggregate on its own:

    python -m p1_mstate_tracking.visualization.random_aggregate \\
        --random_dir results/p1_random \\
        --out         results/p1_random/aggregate.json

    python -m p1_mstate_tracking.visualization.random_aggregate \\
        --random_dir results/p1_random --list
        (prints discovered groups and seed counts, writes nothing)

Or imported directly, to aggregate on the fly instead of reading a
precomputed file:
    from p1_mstate_tracking.visualization.random_aggregate import build_aggregate
    agg = build_aggregate(Path("results/p1_random"))
    agg[("albert-base-v2-random@24iter", "wiki_paragraph")]["cluster_count"]["mean"]

Expected layout under --random_dir: either run subdirs directly (each with
its own geometry.json), or a parent containing several timestamped seed
directories that each hold run subdirs — same two layouts
pipeline._discover_random_dir already handles for a single seed.

Aggregate dict shape, per (model, prompt):
    {
      "model": str, "prompt": str, "n_runs": int, "run_dirs": [str, ...],
      "mass_near_1":         {"mean": [...], "std": [...], "n": int},
      "effective_rank":      {"mean": [...], "std": [...], "n": int},
      "cluster_membership":  {"mean": [...], "std": [...], "n": int},
      "cluster_count":       {"mean": [...], "std": [...], "n": int},
      "cka_prev":             {"mean": [...], "std": [...], "n": int},
      "fiedler_mean":         {"mean": [...], "std": [...], "n": int},
      "token_in_cluster_fraction_pooled": [...],  # every seed's per-token
                                                   # fractions concatenated
    }
Each (model, prompt) entry only has the keys for series at least one seed
actually produced — a metric missing from every seed's artifacts is
omitted rather than filled with NaNs.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .loaders import discover_runs
from .series import (
    _mass_near_1_series,
    _effective_rank_series,
    _cluster_membership_series,
    _cluster_count_series,
    _cka_prev_series,
    _fiedler_mean_series,
    _token_in_cluster_fraction,
)

# Per-layer scalar series aggregated as (mean, std) across seeds. Key here
# is the name written into the aggregate dict; value is the extractor
# already defined in visualization.py — same function trained runs use.
SERIES_EXTRACTORS = {
    "mass_near_1":        _mass_near_1_series,
    "effective_rank":     _effective_rank_series,
    "cluster_membership": _cluster_membership_series,
    "cluster_count":      _cluster_count_series,
    "cka_prev":           _cka_prev_series,
    "fiedler_mean":       _fiedler_mean_series,
}

AggKey = Tuple[str, str]  # (model, prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Discovery — same two layouts visualization._discover_random_dir handles,
# except nothing is overwritten: every seed contributes its own entry.
# ─────────────────────────────────────────────────────────────────────────────

def discover_seed_runs(random_dir: Path) -> Dict[AggKey, List[Path]]:
    """
    Scan results/p1_random/ and group every seed's runs by (model, prompt).
    Unlike visualization.discover_runs (one dir per key, last-wins), every
    seed directory that produced a given (model, prompt) contributes one
    entry to that key's list here — nothing is dropped.
    """
    groups: Dict[AggKey, List[Path]] = {}

    def _add(found: Dict[AggKey, Path]) -> None:
        for key, run_dir in found.items():
            groups.setdefault(key, []).append(run_dir)

    # Layout 1: random_dir holds run subdirs directly (single seed).
    direct = discover_runs(random_dir)
    if direct:
        _add(direct)
        return groups

    # Layout 2: random_dir holds several timestamped seed directories.
    if not random_dir.is_dir():
        return groups
    for seed_dir in sorted(random_dir.iterdir()):
        if seed_dir.is_dir():
            _add(discover_runs(seed_dir))
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _stack_truncated(series_list: List[List[float]]) -> np.ndarray:
    """
    Stack per-seed series into (n_seeds, n_layers), truncating every seed
    to the shortest one present. Seeds at the same (model, prompt) should
    already share a layer count — this just guards against a partial run
    rather than silently failing the whole group.
    """
    arrs = [np.asarray(s, dtype=float) for s in series_list if s]
    if not arrs:
        return np.empty((0, 0))
    min_len = min(a.size for a in arrs)
    if min_len == 0:
        return np.empty((0, 0))
    return np.stack([a[:min_len] for a in arrs], axis=0)


def aggregate_one(model: str, prompt: str, run_dirs: List[Path]) -> Optional[dict]:
    """Build the (mean, std, pooled-token) aggregate for one (model, prompt)."""
    if not run_dirs:
        return None

    out: dict = {
        "model": model, "prompt": prompt,
        "n_runs": len(run_dirs), "run_dirs": [str(rd) for rd in run_dirs],
    }

    for agg_key, extractor in SERIES_EXTRACTORS.items():
        per_seed = []
        for rd in run_dirs:
            try:
                vals = extractor(rd)
            except Exception:
                vals = None
            if vals:
                per_seed.append(vals)
        stacked = _stack_truncated(per_seed)
        if stacked.size == 0:
            continue
        out[agg_key] = {
            "mean": np.nanmean(stacked, axis=0).tolist(),
            "std":  np.nanstd(stacked, axis=0).tolist(),
            "n":    int(stacked.shape[0]),
        }

    pooled: List[float] = []
    for rd in run_dirs:
        try:
            frac = _token_in_cluster_fraction(rd)
        except Exception:
            frac = None
        if frac is not None:
            pooled.extend(frac.tolist())
    if pooled:
        out["token_in_cluster_fraction_pooled"] = pooled

    return out


def build_aggregate(random_dir: Path) -> Dict[AggKey, dict]:
    """Aggregate every (model, prompt) group found under random_dir."""
    groups = discover_seed_runs(random_dir)
    agg: Dict[AggKey, dict] = {}
    for (model, prompt), run_dirs in groups.items():
        entry = aggregate_one(model, prompt, run_dirs)
        if entry is not None:
            agg[(model, prompt)] = entry
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# JSON (de)serialization — tuple keys aren't JSON-safe, so flatten to a
# "model||prompt" string key on the way out and split it on the way back.
# ─────────────────────────────────────────────────────────────────────────────

_SEP = "||"

def save_aggregate(agg: Dict[AggKey, dict], out_path: Path) -> None:
    flat = {f"{model}{_SEP}{prompt}": entry for (model, prompt), entry in agg.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(flat, f, indent=2)


def load_aggregate(path: Path) -> Dict[AggKey, dict]:
    with open(path) as f:
        flat = json.load(f)
    agg: Dict[AggKey, dict] = {}
    for key, entry in flat.items():
        model, prompt = key.split(_SEP, 1)
        agg[(model, prompt)] = entry
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the multi-seed random-control runs in results/p1_random/ "
            "into a per-(model, prompt) mean/std summary for visualization.py."
        ),
    )
    parser.add_argument(
        "--random_dir", type=Path, default=Path("results/p1_random"),
        help="Directory containing the random-control seed runs (default: results/p1_random)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/p1_random/aggregate.json"),
        help="Output JSON path (default: results/p1_random/aggregate.json)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print discovered (model, prompt) groups and seed counts, then exit without writing.",
    )
    args = parser.parse_args()

    if not args.random_dir.exists():
        print(f"ERROR: random_dir not found: {args.random_dir}", file=sys.stderr)
        sys.exit(1)

    groups = discover_seed_runs(args.random_dir)
    if not groups:
        print(f"No runs found under {args.random_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Discovered {len(groups)} (model, prompt) group(s) under {args.random_dir}:")
    for (model, prompt), run_dirs in sorted(groups.items()):
        flag = "" if len(run_dirs) > 1 else "  (single seed — no variance)"
        print(f"  {model:<40} {prompt:<20} n={len(run_dirs)}{flag}")

    if args.list:
        return

    agg = build_aggregate(args.random_dir)
    save_aggregate(agg, args.out)
    print(f"\nWrote aggregate for {len(agg)} group(s) -> {args.out}")


if __name__ == "__main__":
    main()
