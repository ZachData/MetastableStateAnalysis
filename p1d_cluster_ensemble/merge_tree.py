"""
p1d_cluster_ensemble/merge_tree.py — cluster count against scale, and
clusters linked across neighbouring layers by shared tokens.

This is option D from the literature scan (`lit-1d.md` §5): read the full
agglomerative hierarchy of one layer as a step function of cluster count
against distance threshold `delta`, and call a scale robust when it
survives an interval of `delta` (Fred & Jain's lifetime; ToMATo's
persistence; `lit-1d.md` finding 4). **Not done here:** marking the
theory's `delta = c * beta_eff^-1/2` on this curve (option C), which waits
on beta's scale convention (STATE.md "Blocked" item 9).

Two properties of the real data shaped the reading (`status-1d.md` "Merge
tree over scales..."):

- **Longest lifetime alone picks an extreme.** On Pythia layers >= 2 the
  longest-lived plateau is one cluster holding 91-96 % of tokens plus a few
  outliers (often token 0, the attention sink); at L0 it is the exact
  token-identity partition. Counting clusters there counts outliers. So a
  plateau is only a candidate when at least two of its clusters have
  `SUBSTANTIAL_CLUSTER_SIZE` tokens, and every plateau reports how many
  substantial clusters and how many outlier tokens it has. The plain
  longest-lived plateau is still reported (`longest_any`), since "one blob
  plus outliers" is itself the finding at that scale.
- **Jaccard cannot see a small piece leave a large cluster.** One token
  leaving 460 has Jaccard 1/460, so a Jaccard-thresholded link records a
  birth, never a split. Links are classified on containment (overlap over
  the smaller cluster's size) by default; Jaccard is kept as an option and
  stored on every edge.

The second half links each layer's partition to its neighbour's and
classifies each connected group of the overlap graph as stable, merge,
split or tangle (both at once). It is a fresh implementation rather than a
call into `p1_mstate_tracking.cluster_tracking`, whose Hungarian one-to-one
match picks a primary trajectory per cluster and so cannot report a split:
an unmatched later cluster overlapping a matched one becomes a birth. On
Jaccard inputs with no split, the two agree on births, deaths and merge
groups (`tests/test_phase1d_merge_tree.py` `TestAgreesWithClusterTracking`).

Both halves are tier 1: descriptive readouts, not adjudications.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from .constants import SUBSTANTIAL_CLUSTER_SIZE
from .methods import LayerData

#: Linkages whose merge heights are cosine distances. Ward is excluded for
#: the reason methods.py fits it on coordinates: its height is a variance
#: increase, not a distance, so it would put a second unit on the same axis.
MERGE_TREE_LINKAGES = ("average", "complete", "single")

#: Merge heights closer than this are one tie group (identical token
#: vectors merge at ~1e-16, not exactly 0). PLACED: far above float64
#: rounding of a clipped cosine distance, far below any real gap.
HEIGHT_TIE_TOL = 1e-12

LINK_MEASURES = ("containment", "jaccard")

#: PLACED defaults. Containment 0.5: most of the smaller cluster lies in
#: the other. Jaccard 0.1: p1_mstate_tracking.cluster_tracking's default.
DEFAULT_MIN_OVERLAP = {"containment": 0.5, "jaccard": 0.1}


# ---------------------------------------------------------------------------
# Part 1 — the merge tree over scales
# ---------------------------------------------------------------------------

def layer_merge_tree(data: LayerData, linkage: str = "average", top_n: int = 5,
                     min_size: int = SUBSTANTIAL_CLUSTER_SIZE) -> Dict:
    """
    One layer's full agglomerative hierarchy as plateaus of constant `k`.

    Merges whose heights tie (within `HEIGHT_TIE_TOL`) are applied together,
    so every plateau has positive width and cutting at its `delta_lo`
    reproduces its `k`. Each plateau carries `k_substantial` (clusters with
    at least `min_size` tokens) and `n_outliers` (tokens in smaller ones).

    Trivial, and never robust: the first plateau (`delta_lo` in the tie
    group at 0: singletons, or exact duplicates merged — at L0 that is token
    identity) and `k == 1`. `robust` is the non-trivial plateaus with
    `k_substantial >= 2`, by lifetime, descending, at most `top_n`.
    `longest_any` is the longest non-trivial plateau with no size condition.

    `_Z` is the scipy linkage matrix for `labels_at_delta` in the same
    process; callers strip it before writing JSON (it is reproducible from
    `activations.npz`).
    """
    if linkage not in MERGE_TREE_LINKAGES:
        raise ValueError(f"unknown merge-tree linkage {linkage!r}; use one of "
                         f"{MERGE_TREE_LINKAGES}")
    n = data.n
    empty = {"n_tokens": n, "linkage": linkage, "min_size": int(min_size),
             "n_merges": 0, "plateaus": [], "robust": [], "longest_any": None,
             "_Z": None}
    if n < 2:
        return {**empty, "branch": "n<2"}

    if n == 2:
        # scipy refuses a 1x1 condensed matrix; the one merge is exact.
        Z = np.array([[0.0, 1.0, max(float(data.cos_dist[0, 1]), 0.0), 2.0]])
        branch = "two_tokens"
    else:
        try:
            from scipy.cluster.hierarchy import linkage as _linkage
            from scipy.spatial.distance import squareform
        except ImportError:
            return {**empty, "branch": "scipy_unavailable"}
        D = np.asarray(data.cos_dist, dtype=np.float64)
        D = np.clip(0.5 * (D + D.T), 0.0, None)
        np.fill_diagonal(D, 0.0)
        try:
            Z = _linkage(squareform(D, checks=False), method=linkage)
        except (ValueError, RuntimeError):
            return {**empty, "branch": "linkage_failed"}
        if np.any(np.diff(Z[:, 2]) < -HEIGHT_TIE_TOL):
            return {**empty, "branch": "non_monotone_heights"}
        branch = "plateaus"

    plateaus = _plateaus(Z, n, int(min_size))
    nontrivial = [p for p in plateaus if not p["trivial"]]
    robust = sorted((p for p in nontrivial if p["k_substantial"] >= 2),
                    key=lambda p: p["lifetime"], reverse=True)[:max(0, int(top_n))]
    longest = max(nontrivial, key=lambda p: p["lifetime"]) if nontrivial else None
    return {"n_tokens": n, "linkage": linkage, "min_size": int(min_size),
            "branch": branch, "n_merges": int(Z.shape[0]), "plateaus": plateaus,
            "robust": robust, "longest_any": longest, "_Z": Z}


def _plateaus(Z: np.ndarray, n: int, min_size: int) -> List[Dict]:
    """Walk the merges in order, one tie group at a time, tracking sizes."""
    heights = np.asarray(Z[:, 2], dtype=float)
    size = {i: 1 for i in range(n)}
    n_sub = n if min_size <= 1 else 0
    n_out = 0 if min_size <= 1 else n

    def _drop(s: int) -> None:
        nonlocal n_sub, n_out
        if s >= min_size:
            n_sub -= 1
        else:
            n_out -= s

    def _add(s: int) -> None:
        nonlocal n_sub, n_out
        if s >= min_size:
            n_sub += 1
        else:
            n_out += s

    # tie groups: [start, end) row ranges; the first group is the one at ~0
    groups: List[List[int]] = []
    i, m = 0, heights.size
    while i < m:
        j = i + 1
        while j < m and heights[j] - heights[i] <= HEIGHT_TIE_TOL:
            j += 1
        groups.append([i, j])
        i = j
    if not groups or heights[0] > HEIGHT_TIE_TOL:
        groups.insert(0, [0, 0])  # no zero-height merges: floor is all singletons

    plateaus: List[Dict] = []
    done = 0
    for g, (start, end) in enumerate(groups):
        for row in range(start, end):
            a, b = int(Z[row, 0]), int(Z[row, 1])
            sa, sb = size.pop(a), size.pop(b)
            _drop(sa)
            _drop(sb)
            size[n + row] = sa + sb
            _add(sa + sb)
        done = end
        delta_lo = float(heights[end - 1]) if end > start else 0.0
        delta_hi = (float(heights[groups[g + 1][0]]) if g + 1 < len(groups)
                    else math.inf)
        k = n - done
        plateaus.append({
            "k": int(k), "k_substantial": int(n_sub), "n_outliers": int(n_out),
            "delta_lo": delta_lo, "delta_hi": delta_hi,
            "lifetime": delta_hi - delta_lo,
            "trivial": bool(g == 0 or k == 1),
        })
    return plateaus


def labels_at_delta(Z: np.ndarray, n_tokens: int, delta: float) -> np.ndarray:
    """
    (n_tokens,) int32 labels from cutting the tree at `delta`. Use a
    plateau's own `delta_lo`: `fcluster` keeps every merge at height <= t,
    and `delta_lo` is the top of that plateau's tie group.
    """
    if n_tokens < 2:
        return np.zeros(n_tokens, dtype=np.int32)
    from scipy.cluster.hierarchy import fcluster
    return fcluster(Z, t=float(delta), criterion="distance").astype(np.int32) - 1


def substantial_labels(labels: np.ndarray, min_size: int = SUBSTANTIAL_CLUSTER_SIZE
                       ) -> np.ndarray:
    """Labels with every cluster smaller than `min_size` set to -1 (outlier)."""
    labels = np.asarray(labels, dtype=np.int32).copy()
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    small = ids[counts < min_size]
    labels[np.isin(labels, small)] = -1
    return labels


# ---------------------------------------------------------------------------
# Part 2 — clusters linked across neighbouring layers
# ---------------------------------------------------------------------------

_KINDS = ("stable", "merge", "split", "tangle", "birth", "death")


def link_layer_pair(labels_a: np.ndarray, labels_b: np.ndarray,
                    min_overlap: Optional[float] = None,
                    measure: str = "containment") -> Dict:
    """
    Link layer `a`'s clusters to layer `b`'s by token overlap, read as
    connected components of the overlap graph. Tokens labelled -1 are
    outliers and belong to no cluster on that side.

    An edge needs `measure >= min_overlap`, where containment is
    |A & B| / min(|A|, |B|) and Jaccard is |A & B| / |A | B|. Both are stored
    on every edge as [a, b, jaccard, containment]. A component with
    `n_prev` a-clusters and `n_curr` b-clusters is

        stable  1 and 1        merge   >1 and 1
        split   1 and >1       tangle  >1 and >1  (not forced into either)

    and a cluster with no edge is a death (a side) or birth (b side): its
    scale does not persist to the neighbouring layer.
    """
    if measure not in LINK_MEASURES:
        raise ValueError(f"unknown link measure {measure!r}; use one of {LINK_MEASURES}")
    if min_overlap is None:
        min_overlap = DEFAULT_MIN_OVERLAP[measure]
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if labels_a.shape != labels_b.shape:
        raise ValueError(f"layer pair must share token count; got "
                         f"{labels_a.shape} vs {labels_b.shape}")

    ids_a = sorted(int(c) for c in set(labels_a.tolist()) if c >= 0)
    ids_b = sorted(int(c) for c in set(labels_b.tolist()) if c >= 0)
    sets_a = {c: set(np.flatnonzero(labels_a == c).tolist()) for c in ids_a}
    sets_b = {c: set(np.flatnonzero(labels_b == c).tolist()) for c in ids_b}

    adj: Dict[str, set] = {f"a{c}": set() for c in ids_a}
    adj.update({f"b{c}": set() for c in ids_b})
    edges: List[List] = []
    for a in ids_a:
        sa = sets_a[a]
        for b in ids_b:
            sb = sets_b[b]
            inter = len(sa & sb)
            if inter == 0:
                continue
            jac = inter / (len(sa) + len(sb) - inter)
            con = inter / min(len(sa), len(sb))
            if (con if measure == "containment" else jac) >= min_overlap:
                edges.append([a, b, float(jac), float(con)])
                adj[f"a{a}"].add(f"b{b}")
                adj[f"b{b}"].add(f"a{a}")

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
        elif len(prev) == 1:
            kind = "stable" if len(curr) == 1 else "split"
        else:
            kind = "merge" if len(curr) == 1 else "tangle"
        components.append({"prev": prev, "curr": curr, "kind": kind})

    return {
        "n_clusters_a": len(ids_a), "n_clusters_b": len(ids_b),
        "n_outliers_a": int((labels_a < 0).sum()),
        "n_outliers_b": int((labels_b < 0).sum()),
        "measure": measure, "min_overlap": float(min_overlap),
        "edges": edges, "components": components,
        "counts": {kind: sum(c["kind"] == kind for c in components) for kind in _KINDS},
    }


def layer_link_chain(labels_by_layer: Dict[int, np.ndarray],
                     layers: Optional[Sequence[int]] = None,
                     min_overlap: Optional[float] = None,
                     measure: str = "containment") -> Dict:
    """
    `link_layer_pair` over consecutive entries of `layers` (default: the
    sorted keys). A boundary where either side has no partition is recorded
    as skipped, not bridged: linking L5 to L7 across a missing L6 would read
    two layers of change as one.
    """
    layers = sorted(labels_by_layer) if layers is None else list(layers)
    if min_overlap is None:
        min_overlap = DEFAULT_MIN_OVERLAP.get(measure, 0.0)
    boundaries: List[Dict] = []
    for here, nxt in zip(layers, layers[1:]):
        if here not in labels_by_layer or nxt not in labels_by_layer:
            boundaries.append({"layer_from": here, "layer_to": nxt,
                               "skipped": "no robust plateau on one side"})
            continue
        link = link_layer_pair(labels_by_layer[here], labels_by_layer[nxt],
                               min_overlap=min_overlap, measure=measure)
        boundaries.append({"layer_from": here, "layer_to": nxt, **link})

    linked = [b for b in boundaries if "skipped" not in b]
    totals = {kind: sum(b["counts"][kind] for b in linked) for kind in _KINDS}
    return {"layers": layers, "measure": measure, "min_overlap": float(min_overlap),
            "n_skipped": len(boundaries) - len(linked),
            "boundaries": boundaries, "totals": totals}
