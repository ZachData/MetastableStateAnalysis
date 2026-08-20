"""
p1d_cluster_ensemble/methods.py — the method registry.

Seven clustering families, each with a hyperparameter grid, behind one
`fit(family, params, data, seed) -> labels` call. Phase 1 runs four
partitions at library defaults; this module is what makes "tuned" mean
something for the other six, and for HDBSCAN itself.

Families and why each is here
-----------------------------
Every entry is a *different inductive bias*, not a different
implementation of the same one. A consensus built from six centroid
methods would be a consensus about centroids.

  hdbscan          density, with a refusal label. The shipped method.
                    Tuned here for the first time — min_cluster_size=2 is
                    a library-minimum, not a choice anyone made.
  kmeans           Euclidean centroids at fixed k. Phase 1's second
                    partition. On L2-normed rows squared Euclidean
                    distance is 2 - 2cos, so the *assignment* step is
                    cosine; the centroid update is not (it leaves the
                    sphere), which is exactly what spherical_kmeans fixes.
  spherical_kmeans centroids renormalized onto S^{d-1} each iteration —
                    the k-means whose objective is actually cosine. New
                    here. If it and kmeans disagree, the difference is
                    off-sphere centroid drift and nothing else, which
                    makes the pair a controlled comparison rather than
                    two votes for one bias.
  agglomerative    linkage at a distance threshold. Phase 1 sweeps 12
                    thresholds but persists labels at one; here all 12
                    are re-fit under three linkages, plus Ward on k.
                    Single/complete/average bracket the chaining-vs-
                    compactness axis that one linkage choice hides.
  spectral         graph cut on an affinity built from the same cosine
                    distances. Phase 1 has the eigengap k but never the
                    partition.
  gmm              a generative model with a likelihood — the only family
                    whose fit can be scored without reference to a
                    distance at all. Fit on PCA coordinates because a
                    full-covariance mixture in d=1024 with n<=512 tokens
                    is not estimable, and that reduction is recorded in
                    the params rather than hidden.
  graph_modularity greedy modularity on a mutual-kNN cosine graph. The
                    one family with no notion of a centre, a radius, or a
                    number of clusters: it is the strongest available
                    check that a partition is not an artifact of assuming
                    clusters are blobs. Implemented here (no networkx /
                    igraph in this project's dependency set) — see
                    _greedy_modularity.

Deliberately absent: UMAP-then-cluster. It is available (Phase 1 already
optionally imports umap-learn) and it is excluded on purpose — clustering
a 2-D embedding measures the embedding's inductive bias, which is
neighbour-preserving by construction, and would enter the consensus as a
vote for whatever the density methods already say. Recorded here so the
absence reads as a decision rather than an oversight.

Backends
--------
HDBSCAN has two: the `hdbscan` package (what Phase 1 imports) and
sklearn's `sklearn.cluster.HDBSCAN` (sklearn >= 1.3). They are not
bit-identical. `hdbscan_backend()` reports which one is in use and every
artifact written by this phase carries it, because a P-C2 verdict
computed against a different implementation than Phase 1 ran is not a
comparison of settings.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture

from .constants import DISTANCE_THRESHOLDS, K_VALUES

FAMILIES = (
    "hdbscan",
    "kmeans",
    "spherical_kmeans",
    "agglomerative",
    "spectral",
    "gmm",
    "graph_modularity",
)

#: Families whose labels depend on an RNG seed. Everything else returns
#: the same partition every time, which is why stability across seeds is
#: only a meaningful statistic for these (core.seeds.run_clustering_over_seeds
#: would report a perfect 1.0 for the rest and mean nothing by it).
STOCHASTIC = frozenset({"kmeans", "spherical_kmeans", "spectral", "gmm"})

#: Only HDBSCAN can decline to assign a token. Every other family here
#: partitions the whole set, so a -1 in their output would be a bug, not
#: a refusal — asserted in fit().
CAN_REFUSE = frozenset({"hdbscan"})


# ---------------------------------------------------------------------------
# Layer data
# ---------------------------------------------------------------------------

@dataclass
class LayerData:
    """
    One layer's tokens in the two forms the families need: L2-normed rows
    and the cosine distance matrix between them.

    Held together rather than passed separately because every subsample
    has to slice both consistently — a distance matrix sliced on one axis
    only is the kind of error that produces a plausible wrong number.
    """
    normed: np.ndarray      # (n, d) float32, rows unit-norm
    cos_dist: np.ndarray    # (n, n) float64, symmetric, zero diagonal

    @classmethod
    def from_normed(cls, normed: np.ndarray) -> "LayerData":
        X = np.asarray(normed, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"LayerData needs (n_tokens, d); got shape {X.shape}")
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        if not np.allclose(norms, 1.0, atol=1e-3):
            X = X / np.maximum(norms, 1e-12)
        D = np.clip(pairwise_distances(X, metric="cosine"), 0.0, None).astype(np.float64)
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)
        return cls(normed=X, cos_dist=D)

    @property
    def n(self) -> int:
        return int(self.normed.shape[0])

    def subset(self, idx: Sequence[int]) -> "LayerData":
        """Both views restricted to the same token subset."""
        idx = np.asarray(idx, dtype=int)
        return LayerData(normed=self.normed[idx],
                         cos_dist=self.cos_dist[np.ix_(idx, idx)])


# ---------------------------------------------------------------------------
# HDBSCAN backend
# ---------------------------------------------------------------------------

_BACKEND_CACHE: Optional[dict] = None


def hdbscan_backend() -> dict:
    """
    {name, version, available} for whichever HDBSCAN implementation this
    environment provides. `hdbscan` is preferred over sklearn's because
    Phase 1 imports `hdbscan`, and P-C2 compares a tuned setting against
    the shipped one — across implementations that comparison would carry
    an implementation difference inside it.
    """
    global _BACKEND_CACHE
    if _BACKEND_CACHE is not None:
        return _BACKEND_CACHE
    try:
        import hdbscan as _h
        _BACKEND_CACHE = {"name": "hdbscan", "available": True,
                          "version": getattr(_h, "__version__", "unknown")}
        return _BACKEND_CACHE
    except ImportError:
        pass
    try:
        from sklearn.cluster import HDBSCAN as _probe  # availability probe
        assert _probe is not None
        import sklearn
        _BACKEND_CACHE = {"name": "sklearn", "available": True,
                          "version": sklearn.__version__}
    except ImportError:
        _BACKEND_CACHE = {"name": "none", "available": False, "version": None}
    return _BACKEND_CACHE


def available_families() -> List[str]:
    """Families this environment can actually run, in registry order."""
    if hdbscan_backend()["available"]:
        return list(FAMILIES)
    return [f for f in FAMILIES if f != "hdbscan"]


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

def param_grid(family: str, n_tokens: int, grid: str = "full") -> List[Dict]:
    """
    Candidate settings for one family at a given token count, as an
    explicit list of param dicts rather than a Cartesian product — the
    families do not have commensurable axes (Ward takes a k, average
    linkage takes a distance) and pretending they do produces grid points
    that are not fits.

    `grid="quick"` is a strict subset, for smoke runs. It is never the
    default: a selection made over a coarse grid is a different claim
    from one made over the full grid, and which was used is written into
    every artifact.

    Every grid is clipped to n_tokens — asking for 9 clusters from 6
    tokens is not a setting, and letting sklearn raise there would make
    a small prompt look like a code failure.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; known: {list(FAMILIES)}")
    if grid not in ("full", "quick"):
        raise ValueError(f"unknown grid {grid!r}; use 'full' or 'quick'")

    ks = [k for k in K_VALUES if k < max(n_tokens, 2)]
    if grid == "quick":
        ks = ks[:3]
    if not ks:
        ks = [2]

    if family == "hdbscan":
        sizes = [2, 3, 5, 8, 12] if grid == "full" else [2, 5]
        sizes = [s for s in sizes if s <= max(2, n_tokens // 3)] or [2]
        min_samples = [None, 1, 5] if grid == "full" else [None]
        methods = ["eom", "leaf"] if grid == "full" else ["eom"]
        return [
            {"min_cluster_size": s, "min_samples": ms,
             "cluster_selection_method": m, "cluster_selection_epsilon": 0.0}
            for s in sizes for ms in min_samples for m in methods
            if (ms is None or ms <= s * 2)
        ]

    if family in ("kmeans", "spherical_kmeans"):
        return [{"k": k} for k in ks]

    if family == "agglomerative":
        linkages = ["average", "complete", "single"] if grid == "full" else ["average"]
        thresholds = (DISTANCE_THRESHOLDS if grid == "full"
                      else DISTANCE_THRESHOLDS[::4])
        out = [{"linkage": lk, "threshold": float(t)}
               for lk in linkages for t in thresholds]
        # Ward is not threshold-comparable to the others: its merge cost is
        # a variance increase in the embedding space, not a cosine
        # distance, so it takes a k instead. Included because Ward is the
        # linkage most people reach for, and excluding it would leave the
        # tuned agglomerative arm unrepresentative of what a reader would
        # have tried.
        out += [{"linkage": "ward", "k": k} for k in ks]
        return out

    if family == "spectral":
        affinities = ["rbf_median", "knn"] if grid == "full" else ["rbf_median"]
        return [{"k": k, "affinity": a} for k in ks for a in affinities]

    if family == "gmm":
        covs = ["spherical", "diag"] if grid == "full" else ["spherical"]
        n_comp = int(min(20, max(2, n_tokens - 1)))
        return [{"k": k, "covariance_type": c, "pca_components": n_comp}
                for k in ks for c in covs]

    if family == "graph_modularity":
        neighbours = [3, 5, 8, 12, 20] if grid == "full" else [5, 12]
        neighbours = [k for k in neighbours if k < max(2, n_tokens)] or [2]
        return [{"n_neighbors": int(k)} for k in neighbours]

    raise AssertionError(f"grid missing for family {family!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------

def fit(family: str, params: Dict, data: LayerData, seed: int = 0) -> np.ndarray:
    """
    (n_tokens,) int32 labels. -1 means "not assigned" and may only appear
    for families in CAN_REFUSE; the check is enforced rather than assumed,
    because downstream every -1 is treated as a refusal and a family that
    used it as an ordinary id would corrupt every noise statistic in the
    phase.
    """
    if data.n < 2:
        return np.zeros(data.n, dtype=np.int32)

    fitter = _FITTERS.get(family)
    if fitter is None:
        raise ValueError(f"unknown family {family!r}; known: {list(FAMILIES)}")

    labels = np.asarray(fitter(params, data, seed), dtype=np.int32)
    if labels.shape != (data.n,):
        raise AssertionError(
            f"{family} returned {labels.shape} labels for {data.n} tokens"
        )
    if family not in CAN_REFUSE and (labels == -1).any():
        raise AssertionError(
            f"{family} emitted -1, which this phase reads as a refusal to "
            f"assign. Only {sorted(CAN_REFUSE)} may do that."
        )
    return labels


def _fit_hdbscan(params: Dict, data: LayerData, seed: int) -> np.ndarray:
    backend = hdbscan_backend()
    if not backend["available"]:
        raise RuntimeError(
            "no HDBSCAN backend available (neither the `hdbscan` package nor "
            "sklearn.cluster.HDBSCAN). Phase 1d refuses to substitute another "
            "density method for the one Phase 1 ran."
        )
    D = np.ascontiguousarray(data.cos_dist, dtype=np.float64)
    kwargs = {
        "min_cluster_size": int(params.get("min_cluster_size", 2)),
        "metric": "precomputed",
        "cluster_selection_method": params.get("cluster_selection_method", "eom"),
    }
    if params.get("min_samples") is not None:
        kwargs["min_samples"] = int(params["min_samples"])
    eps = float(params.get("cluster_selection_epsilon", 0.0) or 0.0)
    if eps > 0:
        kwargs["cluster_selection_epsilon"] = eps

    if backend["name"] == "hdbscan":
        import hdbscan as _h
        return _h.HDBSCAN(**kwargs).fit_predict(D)
    from sklearn.cluster import HDBSCAN as _SkHDBSCAN
    # copy=True is not cosmetic: with metric="precomputed" sklearn's
    # HDBSCAN mutates the distance matrix it is handed unless told not to,
    # and this phase re-fits many settings against one cached LayerData.
    # The default is scheduled to flip in sklearn 1.10, so it is passed
    # explicitly rather than relied on in either direction.
    return _SkHDBSCAN(copy=True, **kwargs).fit_predict(D)


def _fit_kmeans(params: Dict, data: LayerData, seed: int) -> np.ndarray:
    k = _clip_k(params["k"], data.n)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(data.normed)


def _fit_spherical_kmeans(params: Dict, data: LayerData, seed: int) -> np.ndarray:
    return spherical_kmeans(data.normed, _clip_k(params["k"], data.n), seed=seed)


def _fit_agglomerative(params: Dict, data: LayerData, seed: int) -> np.ndarray:
    linkage = params.get("linkage", "average")
    if linkage == "ward":
        # Ward is defined for Euclidean geometry only, so it runs on the
        # normed coordinates rather than the precomputed cosine matrix. On
        # unit-norm rows Euclidean distance is a monotone function of
        # cosine distance, so the neighbourhood ordering is identical; the
        # merge *cost* is not, which is the whole point of including it.
        k = _clip_k(params["k"], data.n)
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        return model.fit_predict(np.asarray(data.normed, dtype=np.float64))
    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=float(params["threshold"]),
        linkage=linkage, metric="precomputed",
    )
    return model.fit_predict(data.cos_dist)


def _fit_spectral(params: Dict, data: LayerData, seed: int) -> np.ndarray:
    k = _clip_k(params["k"], data.n)
    A = affinity_matrix(data, params.get("affinity", "rbf_median"))
    model = SpectralClustering(
        n_clusters=k, affinity="precomputed", random_state=seed,
        assign_labels="kmeans", n_init=10,
    )
    with warnings.catch_warnings():
        # A disconnected kNN affinity is a *finding* here, not a problem:
        # it means the graph family found components, which is the answer
        # spectral clustering then partitions further. The warning fires
        # once per fit and would bury everything else in a grid sweep.
        warnings.filterwarnings(
            "ignore", message="Graph is not fully connected",
            category=UserWarning,
        )
        return model.fit_predict(A)


def _fit_gmm(params: Dict, data: LayerData, seed: int) -> np.ndarray:
    k = _clip_k(params["k"], data.n)
    n_comp = int(min(params.get("pca_components", 20),
                     data.normed.shape[1], data.n - 1))
    X = PCA(n_components=max(1, n_comp), random_state=seed).fit_transform(
        np.asarray(data.normed, dtype=np.float64)
    )
    model = GaussianMixture(
        n_components=k, covariance_type=params.get("covariance_type", "spherical"),
        random_state=seed, reg_covar=1e-6, n_init=3,
    )
    return model.fit_predict(X)


def _fit_graph_modularity(params: Dict, data: LayerData, seed: int) -> np.ndarray:
    W = mutual_knn_graph(data, int(params.get("n_neighbors", 5)))
    return _greedy_modularity(W)


_FITTERS = {
    "hdbscan": _fit_hdbscan,
    "kmeans": _fit_kmeans,
    "spherical_kmeans": _fit_spherical_kmeans,
    "agglomerative": _fit_agglomerative,
    "spectral": _fit_spectral,
    "gmm": _fit_gmm,
    "graph_modularity": _fit_graph_modularity,
}


def _clip_k(k, n: int) -> int:
    """A k that exceeds the token count is not a setting; clip and move on."""
    return int(max(1, min(int(k), n - 1 if n > 1 else 1)))


# ---------------------------------------------------------------------------
# Spherical k-means
# ---------------------------------------------------------------------------

def spherical_kmeans(
    X: np.ndarray, k: int, seed: int = 0, n_init: int = 5,
    max_iter: int = 100, tol: float = 1e-6,
) -> np.ndarray:
    """
    k-means with centroids renormalized onto the unit sphere each
    iteration — i.e. the algorithm whose objective is sum_i max_c
    <x_i, mu_c> with ||mu_c|| = 1.

    Written out rather than pulled in because this is the one family
    whose absence from sklearn is the only reason Phase 1 used Euclidean
    k-means on spherical data. Empty clusters are re-seeded from a random
    point (the standard repair); the branch is not recorded per-fit
    because the selection layer above re-fits many times and the
    aggregate that matters — stability — already reflects it.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    k = int(max(1, min(k, n)))
    rng = np.random.default_rng(seed)

    best_labels, best_obj = np.zeros(n, dtype=np.int32), -np.inf
    for _ in range(max(1, n_init)):
        centers = _kmeanspp_cosine(X, k, rng)
        labels = np.zeros(n, dtype=np.int32)
        for _ in range(max_iter):
            labels = np.argmax(X @ centers.T, axis=1).astype(np.int32)
            new = np.empty_like(centers)
            for c in range(k):
                mask = labels == c
                if not mask.any():
                    new[c] = X[rng.integers(n)]
                    continue
                v = X[mask].sum(axis=0)
                nv = np.linalg.norm(v)
                new[c] = v / nv if nv > 1e-12 else X[rng.integers(n)]
            shift = float(np.abs(new - centers).max())
            centers = new
            if shift < tol:
                break
        obj = float(np.max(X @ centers.T, axis=1).sum())
        if obj > best_obj:
            best_obj, best_labels = obj, labels.copy()
    return best_labels


def _kmeanspp_cosine(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding with cosine distance (1 - <x, mu>) as the cost."""
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=np.float64)
    centers[0] = X[rng.integers(n)]
    if k == 1:
        return centers
    closest = 1.0 - X @ centers[0]
    for c in range(1, k):
        weights = np.clip(closest, 0.0, None) ** 2
        total = weights.sum()
        idx = (rng.integers(n) if total <= 1e-12
               else int(rng.choice(n, p=weights / total)))
        centers[c] = X[idx]
        closest = np.minimum(closest, 1.0 - X @ centers[c])
    return centers


# ---------------------------------------------------------------------------
# Graph constructions
# ---------------------------------------------------------------------------

def affinity_matrix(data: LayerData, kind: str = "rbf_median") -> np.ndarray:
    """
    Symmetric non-negative affinity for spectral clustering.

    "rbf_median" uses the median off-diagonal cosine distance as the
    bandwidth — the standard self-tuning choice, and the reason it is
    self-tuning matters here: under Pythia's anisotropy the whole
    distance distribution shifts with depth, and a fixed bandwidth would
    turn that shift into a spurious change in cluster count.

    "knn" is a mutual-kNN adjacency weighted by cosine similarity, which
    keeps only local structure. The two disagree exactly when a partition
    depends on long-range similarity, which is worth being able to see.
    """
    if kind == "knn":
        return mutual_knn_graph(data, n_neighbors=min(10, max(2, data.n - 1)))
    if kind != "rbf_median":
        raise ValueError(f"unknown affinity {kind!r}")
    D = data.cos_dist
    off = D[~np.eye(data.n, dtype=bool)]
    sigma = float(np.median(off)) if off.size else 1.0
    if sigma <= 1e-12:
        sigma = 1.0
    A = np.exp(-(D ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(A, 1.0)
    return A


def mutual_knn_graph(data: LayerData, n_neighbors: int) -> np.ndarray:
    """
    (n, n) symmetric weight matrix of the mutual-kNN cosine graph: an
    edge exists only where each token is among the other's n_neighbors
    nearest, weighted by max(cosine similarity, 0).

    Mutual rather than plain kNN because a plain kNN graph forces every
    token to have degree >= n_neighbors, which hands an isolated token a
    community regardless of the geometry — the exact thing HDBSCAN's -1
    exists to avoid, and this phase needs the graph family to be able to
    make that same refusal in its own idiom (an isolated node ends up a
    singleton community).
    """
    n = data.n
    k = int(max(1, min(n_neighbors, n - 1)))
    D = data.cos_dist
    order = np.argsort(D, axis=1)
    nn = np.zeros((n, n), dtype=bool)
    for i in range(n):
        neigh = [j for j in order[i] if j != i][:k]
        nn[i, neigh] = True
    mutual = nn & nn.T
    W = np.where(mutual, np.clip(1.0 - D, 0.0, None), 0.0)
    np.fill_diagonal(W, 0.0)
    return 0.5 * (W + W.T)


def _greedy_modularity(W: np.ndarray) -> np.ndarray:
    """
    Greedy agglomerative modularity maximization (Clauset-Newman-Moore) on
    a weighted graph.

    Communities start as singletons and the pair of *connected*
    communities with the largest positive modularity gain is merged until
    no positive gain remains. With e[c1,c2] the fraction of total edge
    weight between two communities and a[c] the fraction incident to c,
    the gain from merging is dQ = 2*(e[c1,c2] - a[c1]*a[c2]).

    Dense throughout: n_tokens is at most a few hundred in this project,
    and a dense O(n^3) worst case there is cheaper than the bookkeeping a
    sparse version needs. Isolated nodes stay singletons — no edge, no
    positive gain, no merge.

    Returns consecutively-numbered labels, ordered by community size
    descending, so that repeated runs on the same graph are comparable
    without a relabeling step.
    """
    W = np.asarray(W, dtype=np.float64)
    n = W.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    total = W.sum()
    if total <= 1e-12:
        return np.arange(n, dtype=np.int32)

    e = W / total                      # (n, n), sums to 1 over all ordered pairs
    a = e.sum(axis=1)                  # (n,)
    members = [[i] for i in range(n)]
    alive = np.ones(n, dtype=bool)

    while True:
        gain = 2.0 * (e - np.outer(a, a))
        np.fill_diagonal(gain, -np.inf)
        gain[~alive, :] = -np.inf
        gain[:, ~alive] = -np.inf
        # Only merge communities that actually share an edge; merging
        # disconnected communities can still show a positive gain when
        # a[c1]*a[c2] is tiny, and that produces communities with no
        # internal connectivity at all.
        gain[e <= 0.0] = -np.inf

        best = int(np.argmax(gain))
        i, j = divmod(best, n)
        if not np.isfinite(gain[i, j]) or gain[i, j] <= 1e-12:
            break

        e[i, :] += e[j, :]
        e[:, i] += e[:, j]
        a[i] = a[i] + a[j]
        e[j, :] = 0.0
        e[:, j] = 0.0
        a[j] = 0.0
        alive[j] = False
        members[i].extend(members[j])
        members[j] = []

    comms = [m for m in members if m]
    comms.sort(key=len, reverse=True)
    labels = np.empty(n, dtype=np.int32)
    for cid, mem in enumerate(comms):
        labels[np.asarray(mem, dtype=int)] = cid
    return labels


def modularity(W: np.ndarray, labels: np.ndarray) -> float:
    """
    Newman modularity Q of a labelling on a weighted graph. Not used by
    the fit (the greedy merge tracks its own gains); it exists so the
    result can be checked against an independent implementation of the
    quantity being maximized, which is the only way to catch a merge
    bookkeeping error that still returns plausible communities.
    """
    W = np.asarray(W, dtype=np.float64)
    total = W.sum()
    if total <= 1e-12:
        return 0.0
    k = W.sum(axis=1)
    labels = np.asarray(labels)
    same = labels[:, None] == labels[None, :]
    return float((((W - np.outer(k, k) / total) * same).sum()) / total)
