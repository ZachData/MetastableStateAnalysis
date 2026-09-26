"""
p1d_cluster_ensemble/merge_tree.py — cluster count against scale, and
clusters linked across neighbouring layers by shared tokens.

This is option D from the literature scan (`lit-1d.md` §5): instead of
picking one scale per family (what `selection.py`'s stability ranking
was shown to do, always at an extreme — `status-1d.md` "First real
run"), read the full agglomerative hierarchy as a step function of
cluster count against distance threshold `delta`, and call a scale
"robust" when it survives an interval of `delta` rather than existing at
one grid point (Fred & Jain's lifetime; ToMATo's persistence;
`lit-1d.md` finding 4). **Not done here:** marking the theory's own
`delta = c * beta_eff^-1/2` on this curve (option C) waits on beta's
scale convention (STATE.md "Blocked" item 9); a robust plateau is
reported by its own (delta_lo, delta_hi), with no claim about where the
theory would place it.

The second half links each layer's chosen partition to its neighbour's
by shared token membership and classifies each linked group as stable,
a merge, a split, or (when both happen in the same group) a tangle —
counting merges against splits as a function of depth. This is a fresh,
symmetric implementation rather than a call into
`p1_mstate_tracking.cluster_tracking`, for a concrete reason: that
tracker (`match_layer_pair`) does Hungarian one-to-one matching to pick
a single "primary" trajectory per cluster, so an unmatched later-layer
cluster that overlaps an already-matched one is silently a "birth" —
split information the Hungarian match structurally cannot report. This
module reads connected components of the overlap graph instead, which
has no primary/secondary asymmetry and reports splits as a first-class
outcome alongside merges. It is a different reading of the same kind of
question, not a copy of that module's code, and the two agree on which
groups are births/deaths (`tests/test_phase1d_merge_tree.py`
`TestAgreesWithClusterTracking`).

Both halves are tier 1: descriptive readouts of one v1 run, not
adjudications. Nothing here is registered.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from .methods import LayerData

#: Linkages comparable at a shared distance-threshold scale. Ward is
#: excluded for the same reason methods.py's agglomerative family fits
#: it on coordinates instead of the precomputed matrix: its merge cost
#: is a variance increase, not a cosine distance, so a Ward height is
#: not on the same axis as an average/complete/single height and a
#: merge tree mixing them would plot two different units on one curve.
MERGE_TREE_LINKAGES = ("average", "complete", "single")


# ---------------------------------------------------------------------------
# Part 1 — the merge tree over scales
# ---------------------------------------------------------------------------

def layer_merge_tree(data: LayerData, linkage: str = "average", top_n: int = 5) -> Dict:
    """
    One layer's full agglomerative hierarchy, read as a step function of
    cluster count `k` against distance threshold `delta`.

    Every plateau (a run of `delta` over which `k` does not change) is
    reported, ordered by `delta_lo`. The two ends are always trivial and
    marked as such rather than dropped: `k == n_tokens` is "no merge has
    happened yet" (the resolution floor, not a cluster reading), and
    `k == 1` is "everything has merged" (`delta_hi` and `lifetime` are
    +inf, which `p1d_io._sanitize` writes as JSON `null`). `robust` is
    the non-trivial plateaus ranked by lifetime, descending, truncated
    to `top_n` — the reading `lit-1d.md` recommends in place of one
    family's stability-ranked pick.

    Returns a dict with `n_tokens`, `linkage`, `branch`, `plateaus`,
    `robust`, and `_Z` — the raw scipy linkage matrix, kept only for
    `labels_at_delta` within the same process and stripped by the
    caller before anything is written to `p1d_results.json` (it is
    reproducible from `activations.npz` plus this module, the same
    "re-run rather than store" convention the rest of this phase's
    per-run artifacts follow).
    """
    if linkage not in MERGE_TREE_LINKAGES:
        raise ValueError(f"unknown merge-tree linkage {linkage!r}; use one of "
                          f"{MERGE_TREE_LINKAGES}")
    n = data.n
    empty = {"n_tokens": n, "linkage": linkage, "n_merges": 0,
              "plateaus": [], "robust": [], "_Z": None}
    if n < 2:
        return {**empty, "branch": "n<2"}
    if n == 2:
        # scipy refuses a 1x1 condensed distance matrix; the one possible
        # merge is exact, not a fit.
        d01 = float(data.cos_dist[0, 1])
        Z = np.array([[0.0, 1.0, max(d01, 0.0), 2.0]])
        plateaus = _plateaus_from_heights(np.array([max(d01, 0.0)]), n)
        return {"n_tokens": n, "linkage": linkage, "branch": "two_tokens",
                "n_merges": 1, "plateaus": plateaus,
                "robust": _robust(plateaus, top_n), "_Z": Z}

    try:
        from scipy.cluster.hierarchy import linkage as _linkage
        from scipy.spatial.distance import squareform
    except ImportError:
        return {**empty, "branch": "scipy_unavailable"}

    D = np.asarray(data.cos_dist, dtype=np.float64)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, None)
    try:
        Z = _linkage(squareform(D, checks=False), method=linkage)
    except (ValueError, RuntimeError):
        return {**empty, "branch": "linkage_failed"}

    heights = np.asarray(Z[:, 2], dtype=float)
    # average/complete/single are monotone by construction; a small
    # negative step is float noise in the distance matrix, not a real
    # inversion, and is clipped rather than silently accepted or raising
    # on data this phase otherwise treats as fine (status-1d.md "the
    # float32 defect" is exactly the kind of thing a raise here would
    # have caught earlier).
    if np.any(np.diff(heights) < -1e-9):
        return {**empty, "branch": "non_monotone_heights", "_Z": Z}
    heights = np.maximum.accumulate(heights)

    plateaus = _plateaus_from_heights(heights, n)
    return {"n_tokens": n, "linkage": linkage, "branch": "plateaus",
            "n_merges": int(heights.size), "plateaus": plateaus,
            "robust": _robust(plateaus, top_n), "_Z": Z}


def _plateaus_from_heights(heights: np.ndarray, n: int) -> List[Dict]:
    """heights: sorted, nondecreasing merge heights, length n-1."""
    plateaus: List[Dict] = []
    boundaries = np.concatenate([[0.0], heights])
    for i in range(len(boundaries)):
        k = n - i
        delta_lo = float(boundaries[i])
        delta_hi = float(boundaries[i + 1]) if i + 1 < len(boundaries) else math.inf
        lifetime = delta_hi - delta_lo
        plateaus.append({
            "k": int(k), "delta_lo": delta_lo, "delta_hi": delta_hi,
            "lifetime": lifetime, "trivial": bool(k == n or k == 1),
        })
    return plateaus


def _robust(plateaus: Sequence[Dict], top_n: int) -> List[Dict]:
    candidates = [p for p in plateaus if not p["trivial"]]
    candidates.sort(key=lambda p: p["lifetime"], reverse=True)
    return candidates[:max(0, int(top_n))]


def labels_at_delta(Z: np.ndarray, n_tokens: int, delta: float) -> np.ndarray:
    """
    (n_tokens,) int32 labels from cutting a merge tree at `delta`.

    `delta` should be a plateau's own `delta_lo` (a merge height, or 0.0
    for the all-singletons plateau): `fcluster`'s distance criterion
    keeps every merge at height <= t, so cutting at a plateau's own
    lower edge reproduces exactly the `k` that plateau reports, with no
    off-by-one from cutting mid-interval.
    """
    if n_tokens < 2:
        return np.zeros(n_tokens, dtype=np.int32)
    from scipy.cluster.hierarchy import fcluster
    return (fcluster(Z, t=float(delta), criterion="distance").astype(np.int32) - 1)


# ---------------------------------------------------------------------------
# Part 2 — clusters linked across neighbouring layers
# ---------------------------------------------------------------------------

_KINDS = ("stable", "merge", "split", "tangle", "birth", "death")


def link_layer_pair(labels_a: np.ndarray, labels_b: np.ndarray,
                     min_overlap: float = 0.1) -> Dict:
    """
    Link layer `a`'s clusters to layer `b`'s by Jaccard overlap of token
    membership, read as connected components of the (cluster_a,
    cluster_b) overlap graph rather than a one-to-one match.

    A component with `n_prev` layer-`a` clusters and `n_curr` layer-`b`
    clusters is:

        stable  n_prev == 1 and n_curr == 1   (same tokens, both sides)
        merge   n_prev >  1 and n_curr == 1   (several a-clusters -> one)
        split   n_prev == 1 and n_curr >  1   (one a-cluster -> several)
        tangle  n_prev >  1 and n_curr >  1   (both at once)

    A tangle is reported as its own kind, not forced into a merge or a
    split count: several a-clusters partially recombining into several
    b-clusters is neither operation and counting it as one would be a
    modelling choice this module does not make silently.

    An a-cluster (b-cluster) with no edge above `min_overlap` is a
    death (birth) — no noise label enters this reading, unlike
    `p1_mstate_tracking.cluster_tracking`'s HDBSCAN-labels case, because
    the input here is an agglomerative cut, which partitions every
    token; a birth or death means the scale itself does not persist to
    the next layer, not that tokens left structure.
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if labels_a.shape != labels_b.shape:
        raise ValueError(f"layer pair must share token count; got "
                          f"{labels_a.shape} vs {labels_b.shape}")

    ids_a = sorted(int(c) for c in set(labels_a.tolist()))
    ids_b = sorted(int(c) for c in set(labels_b.tolist()))
    sets_a = {c: set(np.flatnonzero(labels_a == c).tolist()) for c in ids_a}
    sets_b = {c: set(np.flatnonzero(labels_b == c).tolist()) for c in ids_b}

    def key_a(c: int) -> str: return f"a{c}"
    def key_b(c: int) -> str: return f"b{c}"

    adj: Dict[str, set] = {key_a(c): set() for c in ids_a}
    adj.update({key_b(c): set() for c in ids_b})
    edges: List[List] = []
    for a in ids_a:
        sa = sets_a[a]
        for b in ids_b:
            sb = sets_b[b]
            inter = len(sa & sb)
            if inter == 0:
                continue
            union = len(sa) + len(sb) - inter
            jac = inter / union if union else 0.0
            if jac >= min_overlap:
                edges.append([a, b, float(jac)])
                adj[key_a(a)].add(key_b(b))
                adj[key_b(b)].add(key_a(a))

    components: List[Dict] = []
    seen: set = set()
    for start in list(adj):
        if start in seen:
            continue
        comp, stack = set(), [start]
        while stack:
            node = stack.pop()
            if node in comp:
                continue
            comp.add(node)
            stack.extend(adj[node] - comp)
        seen |= comp
        prev = sorted(int(x[1:]) for x in comp if x[0] == "a")
        curr = sorted(int(x[1:]) for x in comp if x[0] == "b")
        if not prev:
            kind = "birth"
        elif not curr:
            kind = "death"
        elif len(prev) == 1 and len(curr) == 1:
            kind = "stable"
        elif len(prev) > 1 and len(curr) == 1:
            kind = "merge"
        elif len(prev) == 1 and len(curr) > 1:
            kind = "split"
        else:
            kind = "tangle"
        components.append({"prev": prev, "curr": curr, "kind": kind})

    counts = {kind: sum(1 for c in components if c["kind"] == kind) for kind in _KINDS}
    return {
        "n_clusters_a": len(ids_a), "n_clusters_b": len(ids_b),
        "min_overlap": float(min_overlap),
        "edges": edges, "components": components, "counts": counts,
    }


def layer_link_chain(labels_by_layer: Dict[int, np.ndarray],
                      min_overlap: float = 0.1) -> Dict:
    """
    `link_layer_pair` over every neighbouring pair in the layers that
    have a partition — "neighbouring" in the sorted key set actually
    supplied (e.g. a smoke run's L0/L12/L18), not necessarily adjacent
    transformer layers, since 1d is routinely run at a stride or an
    explicit layer list. The `layer_from` on each boundary is the depth
    "counting merges against splits with depth" reads against; whether
    merges start to dominate splits (or vice versa) at greater depth is
    left to be read off this table; not computed as a trend statistic
    here, since a stride run has too few boundaries for one to mean
    anything (`status-1d.md` "Merge tree and layer links").
    """
    layers = sorted(labels_by_layer)
    boundaries: List[Dict] = []
    for here, nxt in zip(layers, layers[1:]):
        link = link_layer_pair(labels_by_layer[here], labels_by_layer[nxt],
                               min_overlap=min_overlap)
        boundaries.append({"layer_from": here, "layer_to": nxt, **link})

    totals = {kind: sum(b["counts"][kind] for b in boundaries) for kind in _KINDS}
    return {"layers": layers, "min_overlap": float(min_overlap),
            "boundaries": boundaries, "totals": totals}
