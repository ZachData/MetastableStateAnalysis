"""
clustering.py — Standard clustering algorithms + PCA/UMAP projections.

All functions operate on a single-layer (n_tokens, d_model) activation
tensor.  Tokens are L2-normed before any distance computation.

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

from models import layernorm_to_sphere
from config import DISTANCE_THRESHOLDS, K_RANGE

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


def cluster_count_sweep(
    activations: torch.Tensor,
    thresholds: np.ndarray = DISTANCE_THRESHOLDS,
) -> dict:
    """
    Estimate cluster count at each distance threshold via agglomerative
    clustering, and find the best k via KMeans silhouette score.

    Also runs HDBSCAN if available.

    Returns
    -------
    dict with keys:
      agglomerative  : {threshold -> cluster_count}
      kmeans         : {best_k, best_silhouette}
      hdbscan        : {n_clusters}  (only if hdbscan is installed)
    """
    normed   = layernorm_to_sphere(activations).numpy()
    n        = normed.shape[0]
    results  = {"agglomerative": {}, "kmeans": {}}

    cos_dist = np.clip(pairwise_distances(normed, metric="cosine"), 0, None)

    for t in thresholds:
        agg    = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(t),
            linkage="average",
            metric="precomputed",
        )
        labels = agg.fit_predict(cos_dist)
        results["agglomerative"][float(t)] = int(len(set(labels)))

    best_k, best_sil = 1, -1.0
    if n > 3:
        for k in K_RANGE:
            if k >= n:
                break
            km     = KMeans(n_clusters=k, n_init=5, random_state=42)
            labels = km.fit_predict(normed)
            if len(set(labels)) < 2:
                continue
            sil    = silhouette_score(normed, labels, metric="cosine")
            if sil > best_sil:
                best_sil = sil
                best_k   = k
    results["kmeans"]["best_k"]          = best_k
    results["kmeans"]["best_silhouette"] = best_sil

    if HAS_HDBSCAN:
        hdb        = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean")
        labels     = hdb.fit_predict(normed)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        results["hdbscan"] = {"n_clusters": n_clusters}

    return results


def pca_projection(
    activations: torch.Tensor,
    n_components: int = 3,
):
    """
    Project L2-normed activations onto their top PCA components.

    Returns
    -------
    projected            : (n_tokens, n_components) array
    explained_variance_ratio : (n_components,) array
    """
    normed = layernorm_to_sphere(activations).numpy()
    n_comp = min(n_components, normed.shape[1], normed.shape[0] - 1)
    pca    = PCA(n_components=n_comp)
    return pca.fit_transform(normed), pca.explained_variance_ratio_


def umap_projection(
    activations: torch.Tensor,
    n_components: int = 2,
):
    """
    Project L2-normed activations with UMAP.

    Returns None if umap-learn is not installed or n_tokens is too small.
    """
    if not HAS_UMAP:
        return None
    normed = layernorm_to_sphere(activations).numpy()
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
