"""
clustering.py — Standard clustering algorithms + PCA/UMAP projections.

All functions operate on a single-layer (n_tokens, d_model) activation
tensor.  Tokens are L2-normed before any distance computation.

If the caller already holds an L2-normed ndarray (e.g. from the analysis
loop where normed is pre-computed once per layer), it can be passed directly
to avoid redundant normalization — all three public functions accept either a
torch.Tensor (which will be normalised internally) or a pre-normalised
np.ndarray (which is used as-is).

Functions
---------
cluster_count_sweep  : agglomerative threshold sweep + KMeans + HDBSCAN
pca_projection       : PCA onto S^{d-1}-normed activations
umap_projection      : UMAP (optional — requires umap-learn)
"""

import warnings
import numpy as np
import torch

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.decomposition import PCA

from core.models import layernorm_to_sphere
from core.config import DISTANCE_THRESHOLDS, K_RANGE

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("hdbscan not available — skipping HDBSCAN")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("umap-learn not available — skipping UMAP (pip install umap-learn)")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _to_normed(activations_or_normed) -> np.ndarray:
    """
    Accept either a torch.Tensor (raw activations) or a pre-normalised
    np.ndarray and return an L2-normed float32 ndarray.
    """
    if isinstance(activations_or_normed, np.ndarray):
        return activations_or_normed.astype(np.float32, copy=False)
    # torch.Tensor path
    return layernorm_to_sphere(activations_or_normed).numpy()


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_count_sweep(
    activations,
    thresholds: np.ndarray = DISTANCE_THRESHOLDS,
) -> dict:
    """
    Estimate cluster count at each distance threshold via agglomerative
    clustering, and find the best k via KMeans silhouette score.

    Also runs HDBSCAN if available.

    Parameters
    ----------
    activations : torch.Tensor  (n_tokens, d_model)  OR
                  np.ndarray    (n_tokens, d_model) already L2-normed

    Returns
    -------
    dict with keys:
      agglomerative  : {threshold -> cluster_count}
                       plus "mid_labels" — (n_tokens,) int list at the
                       middle threshold, for Phase 5 spatial analysis
      kmeans         : {best_k, best_silhouette, labels}
                       labels is a (n_tokens,) int list for the winning k
      hdbscan        : {n_clusters, labels}  (only if hdbscan is installed)
                       labels uses -1 for noise tokens
    """
    normed   = _to_normed(activations)
    n        = normed.shape[0]
    results  = {"agglomerative": {}, "kmeans": {}}

    cos_dist   = np.clip(pairwise_distances(normed, metric="cosine"), 0, None)
    thresholds = list(thresholds)
    mid_thresh = float(thresholds[len(thresholds) // 2])

    for t in thresholds:
        agg = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(t),
            linkage="average",
            metric="precomputed",
        )
        agg_labels = agg.fit_predict(cos_dist)
        results["agglomerative"][float(t)] = int(len(set(agg_labels)))
        # Save token assignments at the mid threshold for Phase 5
        if float(t) == mid_thresh:
            results["agglomerative"]["mid_labels"] = agg_labels.tolist()

    best_k, best_sil, best_labels = 1, -1.0, np.zeros(n, dtype=np.int32)
    if n > 3:
        for k in K_RANGE:
            if k >= n:
                break
            km     = KMeans(n_clusters=k, n_init=3, random_state=42)
            labels = km.fit_predict(normed)
            if len(set(labels)) < 2:
                continue
            sil = silhouette_score(normed, labels, metric="cosine")
            if sil > best_sil:
                best_sil   = sil
                best_k     = k
                best_labels = labels.copy()

    results["kmeans"]["best_k"]          = best_k
    results["kmeans"]["best_silhouette"] = best_sil
    results["kmeans"]["labels"]          = best_labels.tolist()

    if HAS_HDBSCAN:
        hdb        = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean")
        hdb_labels = hdb.fit_predict(normed)
        n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
        results["hdbscan"] = {
            "n_clusters": n_clusters,
            "labels":     hdb_labels.tolist(),
        }

    return results


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------

def pca_projection(
    activations,
    n_components: int = 3,
):
    """
    Project L2-normed activations onto their top PCA components.

    Parameters
    ----------
    activations : torch.Tensor  (n_tokens, d_model)  OR
                  np.ndarray    (n_tokens, d_model) already L2-normed

    Returns
    -------
    projected                : (n_tokens, n_components) array
    explained_variance_ratio : (n_components,) array
    """
    normed = _to_normed(activations)
    n_comp = min(n_components, normed.shape[1], normed.shape[0] - 1)
    pca    = PCA(n_components=n_comp)
    return pca.fit_transform(normed), pca.explained_variance_ratio_


def umap_projection(
    activations,
    n_components: int = 2,
):
    """
    Project L2-normed activations with UMAP.

    Parameters
    ----------
    activations : torch.Tensor  (n_tokens, d_model)  OR
                  np.ndarray    (n_tokens, d_model) already L2-normed

    Returns None if umap-learn is not installed or n_tokens is too small.
    """
    if not HAS_UMAP:
        return None
    normed = _to_normed(activations)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="n_jobs value.*overridden",
            category=UserWarning,
        )
        reducer = umap.UMAP(
            n_components=n_components,
            metric="cosine",
            random_state=42,
            n_neighbors=min(15, normed.shape[0] - 1),
            min_dist=0.1,
        )
        return reducer.fit_transform(normed)
