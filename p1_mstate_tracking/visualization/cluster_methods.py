"""
visualization/cluster_methods.py

The non-plotting half of the multi-method cluster comparison: it turns a
run_dir into the four partitions of the same tokens that Phase 1 already
persists, and computes the agreement statistics the figures render.

Phase 1 runs four cluster definitions per layer and saves all four, but
only HDBSCAN was ever plotted:

  hdbscan        density clustering on cosine distance; the only one with
                 a noise label (-1)
  kmeans         k chosen by silhouette over K_RANGE; every token assigned
  agglomerative  average linkage at the *middle* cosine-distance threshold
                 (the only threshold whose labels are persisted)
  fiedler        sign of the second normalized-Laplacian eigenvector — a
                 forced bipartition, k=2 by construction

Nothing here reads a file directly; every path goes through loaders.py.
Nothing here draws; method_comparison.py and spectral_structure.py do that.
The split exists so the agreement arithmetic is testable without a figure
backend or a run directory.

Two conventions worth stating once, because every metric below depends on
them:

Noise policy. HDBSCAN's -1 is not a cluster, it is a refusal to assign.
Comparing it against KMeans (which never refuses) needs a decision, and
the two defensible ones disagree, so both are available:
  "singleton"  each noise token becomes its own cluster. Measures "do the
               methods agree about the whole token set", and penalizes
               HDBSCAN for declining to assign.
  "exclude"    noise tokens are dropped from both partitions. Measures "do
               the methods agree about the tokens HDBSCAN was willing to
               commit on", which is the fairer question about the clusters
               HDBSCAN does find.
Every ARI/NMI figure should report which one it used.

KMeans trust gate. K_RANGE starts at 2, so best_k=2 is a floor, not a
finding. In the collapsed regime all tokens are near-collinear and any
2-way split scores a silhouette of ~0.1-0.3 from geometry alone. The gate
here (silhouette >= 0.1 AND effective_rank >= 10) is the same one
reporting_p1._method_agreement applies, kept identical on purpose so the
figure and the text report never disagree about which layers count.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .loaders import (
    _geo, _clustering, _spectral, _sinkhorn,
    _hdbscan_labels, _kmeans_labels, _agglom_labels, _fiedler_bipartition,
    _agglom_threshold_counts,
)

# ─────────────────────────────────────────────────────────────────────────────
# Method registry
# ─────────────────────────────────────────────────────────────────────────────

METHOD_ORDER: Tuple[str, ...] = ("hdbscan", "kmeans", "agglomerative", "fiedler")

METHOD_LABELS: Dict[str, str] = {
    "hdbscan":       "HDBSCAN (density)",
    "kmeans":        "KMeans (silhouette k)",
    "agglomerative": "Agglomerative (mid threshold)",
    "fiedler":       "Fiedler sign (k=2)",
}

# Deliberately not CLUSTER_PAL — these color *methods*, not cluster ids,
# and must never be confused with a cluster identity in a shared legend.
METHOD_COLORS: Dict[str, str] = {
    "hdbscan":       "#DC2626",
    "kmeans":        "#2563EB",
    "agglomerative": "#059669",
    "fiedler":       "#7C3AED",
    "spectral_k":    "#D97706",   # count-only series, no label array
    "sinkhorn_k":    "#6B7280",   # count-only series, attention-graph
}

# KMeans trust gate — mirrors reporting_p1._method_agreement.
KMEANS_SIL_MIN = 0.1
KMEANS_RANK_MIN = 10.0


def method_labels(
    run_dir: Path, methods: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    {method: {layer_idx: (n_tokens,) int array}} for every requested method
    that has data on disk. Methods with no persisted labels are omitted
    entirely rather than returned empty, so callers can branch on presence.
    """
    wanted = tuple(methods) if methods is not None else METHOD_ORDER
    readers = {
        "hdbscan":       lambda rd: {k: np.asarray(v, dtype=np.int32)
                                     for k, v in _hdbscan_labels(rd).items()},
        "kmeans":        lambda rd: {k: np.asarray(v, dtype=np.int32)
                                     for k, v in _kmeans_labels(rd).items()},
        "agglomerative": lambda rd: {k: np.asarray(v, dtype=np.int32)
                                     for k, v in _agglom_labels(rd).items()},
        "fiedler":       lambda rd: {k: np.asarray(v, dtype=np.int32)
                                     for k, v in _fiedler_bipartition(rd).items()},
    }
    out: Dict[str, Dict[int, np.ndarray]] = {}
    for name in wanted:
        reader = readers.get(name)
        if reader is None:
            continue
        labels = reader(run_dir)
        if labels:
            out[name] = labels
    return out


def common_layers(per_method: Dict[str, Dict[int, np.ndarray]]) -> List[int]:
    """
    Ascending layer indices present in *every* method, and where every
    method agrees on n_tokens. A layer where one method wrote a
    different-length array (truncated run, mismatched prompt) is dropped
    rather than compared against a padded or sliced counterpart.
    """
    if not per_method:
        return []
    shared = set.intersection(*(set(d.keys()) for d in per_method.values()))
    out = []
    for li in sorted(shared):
        sizes = {int(per_method[m][li].size) for m in per_method}
        if len(sizes) == 1 and sizes != {0}:
            out.append(li)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Noise handling
# ─────────────────────────────────────────────────────────────────────────────

def noise_as_singletons(labels: np.ndarray) -> np.ndarray:
    """
    Replace every -1 with a distinct new cluster id. Ids start above the
    existing maximum so no noise token can collide with a real cluster.
    """
    labels = np.asarray(labels, dtype=np.int64)
    is_noise = labels == -1
    if not is_noise.any():
        return labels.copy()
    out = labels.copy()
    start = int(labels.max()) + 1 if (~is_noise).any() else 0
    out[is_noise] = np.arange(start, start + int(is_noise.sum()))
    return out


def _apply_noise_policy(
    a: np.ndarray, b: np.ndarray, policy: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if policy == "singleton":
        return noise_as_singletons(a), noise_as_singletons(b)
    if policy == "exclude":
        keep = (np.asarray(a) != -1) & (np.asarray(b) != -1)
        return np.asarray(a)[keep], np.asarray(b)[keep]
    raise ValueError(f"unknown noise policy: {policy!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise agreement
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_agreement(
    a: np.ndarray, b: np.ndarray, noise_policy: str = "singleton",
) -> Dict[str, float]:
    """
    ARI and NMI between two label arrays over the same tokens.

    Returns n_used alongside the scores: under "exclude" it can fall to
    near zero in the collapsed regime, at which point an ARI of 1.0 means
    "three tokens agreed", not "the methods agree". Callers should mask on
    n_used before drawing.
    """
    a, b = _apply_noise_policy(np.asarray(a), np.asarray(b), noise_policy)
    n = int(a.size)
    if n < 2:
        return {"ari": np.nan, "nmi": np.nan, "n_used": n}
    return {
        "ari":    float(adjusted_rand_score(a, b)),
        "nmi":    float(normalized_mutual_info_score(a, b)),
        "n_used": n,
    }


def agreement_trajectory(
    per_method: Dict[str, Dict[int, np.ndarray]],
    noise_policy: str = "singleton",
    metric: str = "ari",
) -> Tuple[List[int], Dict[Tuple[str, str], np.ndarray], Dict[Tuple[str, str], np.ndarray]]:
    """
    Agreement between every method pair, layer by layer.

    Returns (layers, {(m1, m2): (n_layers,) scores}, {(m1, m2): n_used}).
    Pairs are ordered by METHOD_ORDER so the legend is stable across runs
    regardless of which methods happen to be present.
    """
    layers = common_layers(per_method)
    present = [m for m in METHOD_ORDER if m in per_method]
    scores: Dict[Tuple[str, str], np.ndarray] = {}
    counts: Dict[Tuple[str, str], np.ndarray] = {}
    for i, m1 in enumerate(present):
        for m2 in present[i + 1:]:
            s = np.full(len(layers), np.nan)
            c = np.zeros(len(layers), dtype=int)
            for row, li in enumerate(layers):
                res = pairwise_agreement(
                    per_method[m1][li], per_method[m2][li], noise_policy
                )
                s[row] = res[metric]
                c[row] = res["n_used"]
            scores[(m1, m2)] = s
            counts[(m1, m2)] = c
    return layers, scores, counts


# ─────────────────────────────────────────────────────────────────────────────
# Consensus
# ─────────────────────────────────────────────────────────────────────────────

def co_association(
    label_arrays: Sequence[np.ndarray], noise_policy: str = "singleton",
) -> np.ndarray:
    """
    (n_tokens, n_tokens) co-association matrix: entry (i, j) is the
    fraction of the supplied partitions that place i and j together.

    This is the algorithm-agnostic cluster definition. Block structure
    here cannot be blamed on any single method's inductive bias, which is
    what makes it the strongest available evidence that the clusters are a
    property of the geometry.

    Under "exclude", a noise token contributes nothing to that partition's
    vote — the denominator is per-pair, counting only partitions where
    both tokens were assigned. Pairs with no valid partition come back 0.
    """
    if not label_arrays:
        return np.zeros((0, 0))
    n = int(np.asarray(label_arrays[0]).size)
    total = np.zeros((n, n), dtype=float)
    denom = np.zeros((n, n), dtype=float)

    for raw in label_arrays:
        lab = np.asarray(raw)
        if lab.size != n:
            continue
        if noise_policy == "singleton":
            lab = noise_as_singletons(lab)
            valid = np.ones(n, dtype=bool)
        elif noise_policy == "exclude":
            valid = lab != -1
        else:
            raise ValueError(f"unknown noise policy: {noise_policy!r}")
        same = (lab[:, None] == lab[None, :]).astype(float)
        pair_valid = (valid[:, None] & valid[None, :]).astype(float)
        total += same * pair_valid
        denom += pair_valid

    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(denom > 0, total / denom, 0.0)
    np.fill_diagonal(out, 1.0)
    return out


def consensus_order(C: np.ndarray) -> np.ndarray:
    """
    Row/column ordering that puts the co-association blocks on the
    diagonal, via average-linkage on (1 - C). Falls back to identity
    ordering when scipy is unavailable or the matrix is degenerate — the
    heatmap is still correct, just unsorted.
    """
    n = C.shape[0]
    if n < 3:
        return np.arange(n)
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
    except ImportError:
        return np.arange(n)
    D = 1.0 - C
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, None)
    try:
        Z = linkage(squareform(D, checks=False), method="average")
        return np.asarray(leaves_list(Z), dtype=int)
    except (ValueError, RuntimeError):
        return np.arange(n)


def consensus_strength(C: np.ndarray) -> float:
    """
    Fraction of off-diagonal pairs on which the methods are unanimous
    (co-association exactly 0 or 1). High values mean the partition is
    method-independent; values near 0.5 mean the cluster assignment is
    largely an artifact of which algorithm was run.
    """
    n = C.shape[0]
    if n < 2:
        return np.nan
    iu = np.triu_indices(n, k=1)
    vals = C[iu]
    if vals.size == 0:
        return np.nan
    unanimous = np.isclose(vals, 0.0) | np.isclose(vals, 1.0)
    return float(unanimous.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Cluster-count series (for the agreement figure)
# ─────────────────────────────────────────────────────────────────────────────

def _mid_threshold(thresholds: np.ndarray) -> Optional[float]:
    """The middle sweep threshold — the one whose labels run_1 persists."""
    if thresholds.size == 0:
        return None
    return float(thresholds[len(thresholds) // 2])


def kmeans_trust(run_dir: Path) -> Dict[int, bool]:
    """
    {layer_idx: bool} — whether this layer's KMeans best_k is meaningful,
    per the silhouette + effective-rank gate. See the module docstring.
    """
    ranks = {int(lr["layer"]): lr.get("effective_rank")
             for lr in _geo(run_dir).get("layers", [])}
    out: Dict[int, bool] = {}
    for lr in _clustering(run_dir).get("layers", []):
        li = int(lr["layer"])
        sil = lr.get("clustering", {}).get("kmeans", {}).get("best_silhouette")
        rank = ranks.get(li)
        out[li] = bool(
            sil is not None and rank is not None
            and sil >= KMEANS_SIL_MIN and rank >= KMEANS_RANK_MIN
        )
    return out


def cluster_count_table(run_dir: Path) -> Dict[str, object]:
    """
    Every per-layer cluster-count estimate this run holds, aligned on a
    single layer axis.

    Returns
    -------
    dict with keys:
      layers          (n_layers,) ascending layer indices
      counts          {series_name: (n_layers,) float, NaN where missing}
                      over hdbscan, kmeans, agglomerative, spectral_k,
                      sinkhorn_k
      kmeans_trusted  (n_layers,) bool — the silhouette/rank gate
      mid_threshold   float | None — which agglomerative threshold was used
      agreement       (n_layers,) bool — all trusted series within +/-1,
                      the same criterion as reporting_p1._method_agreement

    The Fiedler bipartition is deliberately absent: it is k=2 by
    construction and would plot as a flat line carrying no information.
    """
    clustering_layers = _clustering(run_dir).get("layers", [])
    spectral_layers = {int(lr["layer"]): lr for lr in _spectral(run_dir).get("layers", [])}
    sinkhorn_layers = {int(lr["layer"]): lr for lr in _sinkhorn(run_dir).get("layers", [])}
    thresholds, _, _ = _agglom_threshold_counts(run_dir)
    mid_t = _mid_threshold(thresholds)

    layers = [int(lr["layer"]) for lr in clustering_layers]
    n = len(layers)
    counts = {k: np.full(n, np.nan)
              for k in ("hdbscan", "kmeans", "agglomerative", "spectral_k", "sinkhorn_k")}
    trust_map = kmeans_trust(run_dir)
    trusted = np.array([trust_map.get(li, False) for li in layers], dtype=bool)

    for row, lr in enumerate(clustering_layers):
        li = layers[row]
        cl = lr.get("clustering", {})

        hdb_k = cl.get("hdbscan", {}).get("n_clusters")
        if hdb_k is not None:
            counts["hdbscan"][row] = float(hdb_k)

        km_k = cl.get("kmeans", {}).get("best_k")
        if km_k is not None:
            counts["kmeans"][row] = float(km_k)

        if mid_t is not None:
            agg = cl.get("agglomerative", {})
            for key, val in agg.items():
                if key == "mid_labels":
                    continue
                try:
                    if np.isclose(float(key), mid_t):
                        counts["agglomerative"][row] = float(val)
                        break
                except (TypeError, ValueError):
                    continue

        sp = spectral_layers.get(li, {}).get("k_eigengap")
        if sp is not None:
            counts["spectral_k"][row] = float(sp)

        sk = sinkhorn_layers.get(li, {}).get("sinkhorn_cluster_count_mean")
        if sk is not None:
            counts["sinkhorn_k"][row] = float(round(sk))

    agreement = np.zeros(n, dtype=bool)
    for row in range(n):
        series = ["hdbscan", "agglomerative", "spectral_k", "sinkhorn_k"]
        if trusted[row]:
            series.append("kmeans")
        vals = [counts[s][row] for s in series]
        vals = [v for v in vals if np.isfinite(v) and v > 0]
        agreement[row] = bool(vals) and (max(vals) - min(vals)) <= 1

    return {
        "layers":         layers,
        "counts":         counts,
        "kmeans_trusted": trusted,
        "mid_threshold":  mid_t,
        "agreement":      agreement,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Threshold-sweep structure
# ─────────────────────────────────────────────────────────────────────────────

def scale_plateau_width(counts_column: np.ndarray) -> int:
    """
    Longest run of consecutive thresholds returning the same cluster
    count, for one layer's column of the agglomerative sweep.

    This is the scale-separation statistic HDBSCAN structurally hides: a
    wide plateau means the partition is stable across a range of cut
    distances (real, well-separated clusters), while a width of 1
    everywhere means the count is entirely a function of where you cut.
    NaN entries break a run rather than extending it.
    """
    col = np.asarray(counts_column, dtype=float)
    best = run = 0
    prev = None
    for v in col:
        if not np.isfinite(v):
            prev, run = None, 0
            continue
        if prev is not None and v == prev:
            run += 1
        else:
            run = 1
        prev = v
        best = max(best, run)
    return int(best)


def plateau_widths(counts: np.ndarray) -> np.ndarray:
    """scale_plateau_width applied down each layer column of the sweep."""
    if counts.size == 0:
        return np.array([])
    return np.array([scale_plateau_width(counts[:, c]) for c in range(counts.shape[1])])


# ─────────────────────────────────────────────────────────────────────────────
# Noise audit
# ─────────────────────────────────────────────────────────────────────────────

def noise_audit(
    hdb: np.ndarray, other: np.ndarray, min_cluster_size: int = 4,
) -> Dict[str, float]:
    """
    Of the tokens HDBSCAN declined to assign, how many does another method
    place inside a substantial cluster?

    "Substantial" means the receiving cluster has at least
    min_cluster_size members — a KMeans cluster of one is the same refusal
    HDBSCAN made, spelled differently.

    Returns
    -------
    n_noise               how many tokens HDBSCAN called noise
    noise_fraction        that as a fraction of all tokens
    rescued_fraction      of the noise tokens, the fraction the other
                          method puts in a cluster of size >= min_cluster_size
    rescued_into_shared   of the noise tokens, the fraction landing in a
                          cluster that also contains at least one token
                          HDBSCAN *did* assign — i.e. absorbed into
                          existing structure rather than pooled together
                          into a separate "leftovers" cluster

    A high rescued_fraction with a low rescued_into_shared is the
    interesting case: it means the noise tokens form their own coherent
    group that HDBSCAN's density criterion split off, which is a different
    claim from "noise is unstructured."
    """
    hdb = np.asarray(hdb)
    other = np.asarray(other)
    n = int(hdb.size)
    is_noise = hdb == -1
    n_noise = int(is_noise.sum())
    out = {
        "n_noise":             n_noise,
        "noise_fraction":      n_noise / n if n else np.nan,
        "rescued_fraction":    np.nan,
        "rescued_into_shared": np.nan,
    }
    if n_noise == 0 or other.size != n:
        return out

    ids, sizes = np.unique(other, return_counts=True)
    size_of = dict(zip(ids.tolist(), sizes.tolist()))
    big = {cid for cid, sz in size_of.items() if sz >= min_cluster_size and cid != -1}
    # Clusters that contain at least one HDBSCAN-assigned token.
    shared = set(np.unique(other[~is_noise]).tolist()) if (~is_noise).any() else set()

    noise_ids = other[is_noise]
    out["rescued_fraction"] = float(np.mean([cid in big for cid in noise_ids.tolist()]))
    out["rescued_into_shared"] = float(
        np.mean([(cid in big and cid in shared) for cid in noise_ids.tolist()])
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Fiedler bipartition sharpness
# ─────────────────────────────────────────────────────────────────────────────

def bipartition_separation(fiedler_vec: np.ndarray) -> Dict[str, float]:
    """
    How sharp the Fiedler split is, from the raw eigenvector.

    A sign bipartition always produces two groups; the question is whether
    the underlying values are bimodal (two genuine lobes) or unimodal with
    the sign cut falling arbitrarily near zero. Reported as:

    separation   |mean(+) - mean(-)| / pooled std. Large = two lobes.
    balance      size of the smaller side / n. Near 0 means the "split"
                 peeled off a handful of outliers.
    near_zero    fraction of |v| below 10% of max|v| — tokens the cut is
                 essentially assigning at random.
    """
    v = np.asarray(fiedler_vec, dtype=float)
    n = int(v.size)
    out = {"separation": np.nan, "balance": np.nan, "near_zero": np.nan}
    if n < 2:
        return out

    pos, neg = v[v > 0], v[v <= 0]
    if pos.size and neg.size:
        pooled = np.sqrt(0.5 * (pos.var() + neg.var()))
        out["separation"] = (
            float(abs(pos.mean() - neg.mean()) / pooled) if pooled > 1e-12 else np.inf
        )
    out["balance"] = float(min(pos.size, neg.size) / n)

    vmax = np.abs(v).max()
    if vmax > 0:
        out["near_zero"] = float((np.abs(v) < 0.1 * vmax).mean())
    return out
