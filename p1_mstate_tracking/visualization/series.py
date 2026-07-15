"""
visualization/series.py

Every per-layer scalar/array extractor that turns a saved run_dir into a
plain list/array: mass-near-1, effective rank, cluster membership/count,
CKA(prev), Fiedler mean, attention entropy, per-token cluster persistence.

These are deliberately kept in one leaf module with no plotting code and
no dependency on generate_all/pipeline: random_aggregate.py (multi-seed)
and energy_attribution_aggregate.py (multi-prompt) both import straight
from here, and overview.py / pair_comparisons.py / energy_decomposition.py
call the same functions when plotting a single run. One extractor, every
caller — nothing recomputes its own copy.

_series_or_aggregate is the one piece of plotting-adjacent logic that
lives here rather than in overview.py: every "-random" line in every
figure module routes through it so a multi-seed mean ± std band is used
whenever random_aggregate.py has produced one, instead of silently
falling back to whichever single seed happened to be discovered.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .naming import _is_untrained
from .loaders import _geo, _clustering, _sinkhorn, _hdbscan_labels

# ─────────────────────────────────────────────────────────────────────────────
# Random-aggregate lookup — every "random" line below goes through this so
# a multi-seed mean ± std band is used whenever one is available, instead
# of whichever single seed happened to be discovered.
# ─────────────────────────────────────────────────────────────────────────────

def _series_or_aggregate(
    model: str, prompt: str, run_dir: Optional[Path], value_fn,
    agg_key: Optional[str], random_agg: Optional[dict],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Returns (mean_vals, std_vals, n_seeds) for a per-layer series.

    Prefers random_agg[(model, prompt)][agg_key] when model is an
    untrained/"-random" variant and that entry exists — std_vals is then
    the across-seed std at each layer and n_seeds > 1. Otherwise falls
    back to value_fn(run_dir) for a single run: std_vals is None, n_seeds
    is 1 (or 0 if nothing could be loaded, in which case mean_vals is also
    None and the caller should skip this model).
    """
    if agg_key and random_agg and _is_untrained(model):
        entry = random_agg.get((model, prompt))
        if entry and agg_key in entry and entry[agg_key].get("mean"):
            d = entry[agg_key]
            return (
                np.asarray(d["mean"], dtype=float),
                np.asarray(d["std"], dtype=float) if d.get("std") else None,
                entry.get("n_runs", 1),
            )
    if run_dir is None:
        return None, None, 0
    try:
        vals = value_fn(run_dir)
    except Exception:
        vals = None
    if not vals:
        return None, None, 0
    return np.asarray(vals, dtype=float), None, 1


# ═════════════════════════════════════════════════════════════════════════════
# Overview figures — the only cross-model charts (one line per model variant)
# ═════════════════════════════════════════════════════════════════════════════

def _mass_near_1_series(run_dir: Path) -> Optional[List[float]]:
    layers = _geo(run_dir).get("layers", [])
    if not layers:
        return None
    out = [lr.get("ip_mass_near_1", np.nan) for lr in layers]
    return [np.nan if v is None else v for v in out]

def _effective_rank_series(run_dir: Path) -> Optional[List[float]]:
    layers = _geo(run_dir).get("layers", [])
    if not layers:
        return None
    out = [lr.get("effective_rank", np.nan) for lr in layers]
    return [np.nan if v is None else v for v in out]

def _cluster_membership_series(run_dir: Path) -> Optional[List[float]]:
    layers = _clustering(run_dir).get("layers", [])
    if not layers:
        return None
    out = []
    for lr in layers:
        nf = lr.get("clustering", {}).get("hdbscan", {}).get("noise_fraction")
        out.append(1.0 - nf if nf is not None else np.nan)
    return out

def _cluster_count_series(run_dir: Path) -> Optional[List[float]]:
    layers = _clustering(run_dir).get("layers", [])
    if not layers:
        return None
    out = []
    for lr in layers:
        k = lr.get("clustering", {}).get("hdbscan", {}).get("n_clusters")
        out.append(k if k is not None else np.nan)
    return out

def _token_in_cluster_fraction(run_dir: Path) -> Optional[np.ndarray]:
    """
    (n_tokens,) — for each token (fixed position across layers, same
    prompt), the fraction of layers in which HDBSCAN assigned it to a real
    cluster rather than noise. 1.0 = clustered at every layer that has
    labels; 0.0 = noise everywhere.
    """
    hdb = _hdbscan_labels(run_dir)
    if not hdb:
        return None
    n_tok = None
    rows = []
    for li in sorted(hdb.keys()):
        arr = np.asarray(hdb[li])
        if arr.size == 0:
            continue
        if n_tok is None:
            n_tok = arr.size
        if arr.size != n_tok:
            continue
        rows.append(arr != -1)
    if not rows:
        return None
    return np.stack(rows, axis=0).mean(axis=0).astype(float)


def _cka_prev_series(run_dir: Path) -> Optional[List[float]]:
    layers = _geo(run_dir).get("layers", [])
    if not layers:
        return None
    out = [lr.get("cka_prev", np.nan) for lr in layers]
    return [np.nan if v is None else v for v in out]


def _fiedler_mean_series(run_dir: Path) -> Optional[List[float]]:
    layers = _sinkhorn(run_dir).get("layers", [])
    if not layers:
        return None
    out = [lr.get("fiedler_mean", np.nan) for lr in layers]
    return [np.nan if v is None else v for v in out]

def _attention_entropy_mean_series(run_dir: Path) -> Optional[List[float]]:
    layers = _sinkhorn(run_dir).get("layers", [])
    if not layers:
        return None
    out = [lr.get("attention_entropy_mean", np.nan) for lr in layers]
    return [np.nan if v is None else v for v in out]
