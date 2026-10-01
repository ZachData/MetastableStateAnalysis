"""
tools/hdbscan_tie_repro.py — HDBSCAN's clusters change with row order (tied edges).

Standalone (numpy, scipy, scikit-learn, hdbscan; nothing from this repo), so it
can be pasted into an upstream issue (`docs/upstream/hdbscan_ties.md`). The same
points are clustered in 30 row orders, labels mapped back, and compared by ARI
to the original order. Repeats on one order are identical, so a change is the
row order, not randomness.

Why: with ``min_samples`` k, a point's core distance is its k-th neighbour
distance, and that one number is the weight of several of its mutual-reachability
edges, so tied edges are the rule. The hierarchy's definition (components of the
graph thresholded at each level) is order-free; the binary single-linkage tree
joins tied edges one at a time in processing order. 1d's fix merges every edge of
one weight at once (`p1d_cluster_ensemble.admit.level_set_hdbscan`;
`status-1d.md` "Admission"). Settings are matched: scikit-learn counts the point
itself in ``min_samples``, so its 3 is hdbscan's 2.

Run: ``python tools/hdbscan_tie_repro.py`` (seconds). Tier 1, not a result.
"""
from __future__ import annotations

import numpy as np


def cosine_cloud(n: int = 200, d: int = 1024, seed: int = 0) -> np.ndarray:
    """Unit rows, one loose group of 30 in a Gaussian background; cosine distances."""
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((n, d))
    Y[:30] += 0.6 * rng.standard_normal(d)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    D = np.clip(1.0 - Y @ Y.T, 0.0, None)
    np.fill_diagonal(D, 0.0)
    return D


def changed_by_order(fit, D: np.ndarray, n_orders: int = 30) -> int:
    """How many random row orders give a different clustering (ARI < 1)."""
    from sklearn.metrics import adjusted_rand_score
    base = fit(D)
    changed = 0
    for s in range(n_orders):
        p = np.random.default_rng(s).permutation(len(D))
        lab = np.empty_like(base)
        lab[p] = fit(D[np.ix_(p, p)])
        changed += adjusted_rand_score(base, lab) < 1.0
    return int(changed)


def main() -> None:
    import hdbscan
    import sklearn
    from sklearn.cluster import HDBSCAN

    D = cosine_cloud()
    fits = {
        f"hdbscan {getattr(hdbscan, '__version__', '')} (min_samples=2)":
            lambda M: hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=2,
                                      min_samples=2).fit(M).labels_,
        f"scikit-learn {sklearn.__version__} HDBSCAN (min_samples=3)":
            lambda M: HDBSCAN(metric="precomputed", min_cluster_size=2, min_samples=3,
                              copy=True).fit(M).labels_,
    }
    for name, fit in fits.items():
        same = all((fit(D) == fit(D)).all() for _ in range(5))
        print(f"{name}: repeat on one order identical: {same}; "
              f"{changed_by_order(fit, D)} of 30 row orders change the clustering")


if __name__ == "__main__":
    main()
