"""
core/seeds.py — Seed policy + stability reporting (transition plan v2,
core infrastructure item 6).

Two things, both formalizing what Phase 1's old 10-seed random-weight
sweep already did ad hoc (see p1_mstate_tracking's design-1.md /
random_aggregate.py): a single place to pin every seed a run uses, and a
standing check on clustering stability across seeds — "run HDBSCAN at k
seeds, report label stability (ARI across seeds) per layer per
checkpoint. Low stability at a checkpoint mechanically flags every
cluster-conditioned result there."

torch is optional here (duck-typed, never imported at module level),
matching core.metrics's pattern, so this module works in a pure
numpy/scipy/sklearn environment.
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score

DEFAULT_STABILITY_THRESHOLD = 0.5  # plan doesn't pin an exact number; this
                                    # is a reasonable default — override
                                    # per call if a phase has a different
                                    # tolerance for "flag this checkpoint."


# ---------------------------------------------------------------------------
# Seed pinning
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int) -> Dict[str, bool]:
    """
    Set python's `random`, numpy, and (if importable) torch's RNGs to
    `seed`. Returns which of them were actually set, so a caller building
    a manifest's `seeds` dict can record exactly what was pinned rather
    than assuming torch is present.

    Recording {"numpy": seed, "torch": seed} in every run's manifest.json
    (core.io.write_manifest's `seeds` parameter) is what the plan's seed
    policy actually requires; this function is the single call site that
    performs the pinning those numbers describe, so the two can't drift
    apart (manifest claims a seed was set that never actually was).
    """
    random.seed(seed)
    np.random.seed(seed)
    set_flags = {"python": True, "numpy": True, "torch": False}

    try:
        import torch  # noqa: F401 (optional dependency, duck-typed below)
    except ImportError:
        return set_flags

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_flags["torch"] = True
    return set_flags


# ---------------------------------------------------------------------------
# Stability reporting
# ---------------------------------------------------------------------------

def cluster_stability_across_seeds(
    labels_per_seed: Sequence[np.ndarray],
    threshold: float = DEFAULT_STABILITY_THRESHOLD,
) -> dict:
    """
    Pairwise Adjusted Rand Index across every pair of seeds' label
    assignments for the same layer (and, by construction of the caller,
    the same checkpoint). ARI is the standard label-stability measure
    here because it's invariant to arbitrary cluster-id relabeling across
    seeds (HDBSCAN doesn't guarantee cluster 3 in one run means the same
    thing as cluster 3 in another) and corrects for chance agreement.

    Parameters
    ----------
    labels_per_seed : list of (n_tokens,) int arrays, one per seed. Must
                       all have the same length (same tokens, same layer).
    threshold        : mean pairwise ARI below this flags `low_stability`.

    Returns
    -------
    dict:
      n_seeds          : int
      pairwise_ari      : list[float], one per unordered seed pair
      mean_ari          : float
      min_ari           : float
      low_stability     : bool  — mean_ari < threshold
      threshold         : float — echoed back for provenance
    """
    n = len(labels_per_seed)
    if n < 2:
        raise ValueError(
            f"cluster_stability_across_seeds needs >= 2 seeds to compare, got {n}"
        )
    lengths = {len(l) for l in labels_per_seed}
    if len(lengths) > 1:
        raise ValueError(
            f"All seeds' label arrays must have the same length (same tokens, "
            f"same layer); got lengths {sorted(lengths)}"
        )

    pairwise = [
        float(adjusted_rand_score(labels_per_seed[i], labels_per_seed[j]))
        for i, j in combinations(range(n), 2)
    ]
    mean_ari = float(np.mean(pairwise))
    min_ari = float(np.min(pairwise))

    return {
        "n_seeds": n,
        "pairwise_ari": pairwise,
        "mean_ari": mean_ari,
        "min_ari": min_ari,
        "low_stability": mean_ari < threshold,
        "threshold": threshold,
    }


def stability_report_per_layer(
    labels_per_seed_per_layer: Dict[int, Sequence[np.ndarray]],
    threshold: float = DEFAULT_STABILITY_THRESHOLD,
) -> Dict[int, dict]:
    """
    `cluster_stability_across_seeds` applied independently at every
    layer. Returns {layer: stability_dict}, so a caller can flag exactly
    which layers (not just which checkpoint) have unreliable clustering —
    "low stability at a checkpoint mechanically flags every
    cluster-conditioned result there" is a per-layer claim in practice,
    since clustering quality routinely varies by depth.
    """
    return {
        layer: cluster_stability_across_seeds(labels_list, threshold=threshold)
        for layer, labels_list in labels_per_seed_per_layer.items()
    }


def run_clustering_over_seeds(
    activations: np.ndarray,
    seeds: Sequence[int],
    cluster_fn: Callable[..., np.ndarray],
    threshold: float = DEFAULT_STABILITY_THRESHOLD,
    **cluster_kwargs,
) -> dict:
    """
    Convenience wrapper: run `cluster_fn(activations, seed=seed,
    **cluster_kwargs) -> labels` once per seed, then report stability.
    Deliberately takes `cluster_fn` as a parameter rather than importing
    a specific HDBSCAN wrapper — the actual clustering implementation
    lives in p1_mstate_tracking/clustering.py (or wherever a given phase's
    clustering call lives), and this module has no opinion on which one;
    it only needs something that accepts activations + a seed and returns
    integer labels.

    Returns the stability dict from cluster_stability_across_seeds, with
    an added "labels_per_seed" key (the raw label arrays, for anything
    downstream that wants them, e.g. picking a representative seed).
    """
    labels_per_seed = [
        np.asarray(cluster_fn(activations, seed=seed, **cluster_kwargs))
        for seed in seeds
    ]
    result = cluster_stability_across_seeds(labels_per_seed, threshold=threshold)
    result["labels_per_seed"] = labels_per_seed
    return result
