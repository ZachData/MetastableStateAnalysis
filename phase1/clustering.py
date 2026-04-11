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
cluster_count_sweep       : agglomerative threshold sweep + KMeans + HDBSCAN
pca_projection            : PCA onto S^{d-1}-normed activations
umap_projection           : UMAP (optional — requires umap-learn)
multiscale_nesting        : spectral eigengap within each HDBSCAN cluster (P1-3)
pair_hdbscan_agreement    : tag mutual-NN pairs as semantic vs attention artifact (P1-4)
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
    mid_idx    = len(thresholds) // 2
    mid_thresh = float(thresholds[mid_idx])

    for idx, t in enumerate(thresholds):
        agg = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(t),
            linkage="average",
            metric="precomputed",
        )
        agg_labels = agg.fit_predict(cos_dist)
        results["agglomerative"][float(t)] = int(len(set(agg_labels)))
        # Save token assignments at the mid threshold for Phase 5
        if idx == mid_idx:
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
        hdb        = hdbscan.HDBSCAN(min_cluster_size=2, metric="precomputed")
        hdb_labels = hdb.fit_predict(cos_dist)
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


# ---------------------------------------------------------------------------
# Multi-scale cluster nesting (P1-3)
# ---------------------------------------------------------------------------

def multiscale_nesting(
    normed: np.ndarray,
    hdbscan_labels: np.ndarray,
    max_k: int = 10,
) -> dict:
    """
    Run spectral eigengap within each HDBSCAN cluster to detect hierarchical
    organization: a global bipartition (spectral k=2) nesting inside local
    density structure (HDBSCAN k=30-60).

    Parameters
    ----------
    normed         : (n_tokens, d) L2-normed activations
    hdbscan_labels : (n_tokens,) HDBSCAN labels (-1 = noise)
    max_k          : maximum eigenvalues to inspect per sub-cluster

    Returns
    -------
    dict with keys:
      global_spectral_k   : int — spectral eigengap k on the full token set
      per_cluster          : dict mapping cluster_id -> {
                               n_tokens, spectral_k, eigenvalues, eigengaps
                             }
      has_nesting          : bool — True if global_k <= 3 AND at least one
                             sub-cluster has spectral_k > 1
      nesting_summary      : str — human-readable description
    """
    from .spectral import spectral_eigengap_k

    normed = normed.astype(np.float32, copy=False)

    # Global spectral k
    G_full = normed @ normed.T
    global_spec = spectral_eigengap_k(G_full, max_k=max_k)
    global_k = global_spec["k_eigengap"]

    # Per-cluster spectral analysis
    cluster_ids = sorted(set(hdbscan_labels) - {-1})
    per_cluster = {}

    for cid in cluster_ids:
        mask = hdbscan_labels == cid
        n_c = int(mask.sum())
        if n_c < 4:
            per_cluster[cid] = {
                "n_tokens": n_c,
                "spectral_k": 1,
                "eigenvalues": [],
                "eigengaps": [],
            }
            continue

        sub_normed = normed[mask]
        G_sub = sub_normed @ sub_normed.T
        sub_spec = spectral_eigengap_k(G_sub, max_k=min(max_k, n_c - 2))
        per_cluster[cid] = {
            "n_tokens": n_c,
            "spectral_k": sub_spec["k_eigengap"],
            "eigenvalues": sub_spec["eigenvalues"],
            "eigengaps": sub_spec["eigengaps"],
        }

    # Nesting detection
    subclusters_with_structure = [
        cid for cid, info in per_cluster.items()
        if info["spectral_k"] > 1
    ]
    has_nesting = global_k <= 3 and len(subclusters_with_structure) > 0

    if has_nesting:
        summary = (
            f"Global spectral k={global_k} (macro-bipartition) with "
            f"{len(subclusters_with_structure)}/{len(cluster_ids)} "
            f"HDBSCAN clusters showing internal sub-structure"
        )
    elif global_k <= 3:
        summary = f"Global spectral k={global_k}, no sub-structure within HDBSCAN clusters"
    else:
        summary = f"Global spectral k={global_k} (>3), nesting analysis not applicable"

    return {
        "global_spectral_k": global_k,
        "per_cluster": per_cluster,
        "has_nesting": has_nesting,
        "nesting_summary": summary,
        "n_clusters_with_substructure": len(subclusters_with_structure),
    }


# ---------------------------------------------------------------------------
# Per-pair HDBSCAN agreement for induction head filtering (P1-4)
# ---------------------------------------------------------------------------

def pair_hdbscan_agreement(
    nn_indices: np.ndarray,
    hdbscan_labels: np.ndarray,
    tokens: list,
) -> dict:
    """
    Tag mutual nearest-neighbour pairs as semantic vs attention artifact.

    A mutual-NN pair (i, j) where nn[i]=j AND nn[j]=i is tagged:
      - "semantic"  if both tokens share the same HDBSCAN cluster
      - "artifact"  if they are mutual NNs but in different clusters
      - "noise"     if either token is HDBSCAN noise (-1)

    Cross-position subword completions (e.g. he↔ger for "Heger") that are
    driven by induction heads will typically appear as mutual NNs in different
    clusters — they are locally attracted by attention but not embedded in
    the same dense region.

    Parameters
    ----------
    nn_indices     : (n_tokens,) int array — nearest-neighbour indices
    hdbscan_labels : (n_tokens,) int array — HDBSCAN cluster labels (-1 = noise)
    tokens         : list of str — decoded token strings

    Returns
    -------
    dict with keys:
      mutual_pairs : list of dicts, each with:
                       i, j           : token indices
                       tok_i, tok_j   : token strings
                       cluster_i, cluster_j : HDBSCAN cluster IDs
                       tag            : "semantic" | "artifact" | "noise"
      n_semantic   : int
      n_artifact   : int
      n_noise      : int
      artifact_fraction : float — fraction of mutual-NN pairs that are artifacts
    """
    n = len(nn_indices)
    nn = np.asarray(nn_indices, dtype=np.int32)
    labels = np.asarray(hdbscan_labels, dtype=np.int32)

    # Find mutual-NN pairs (i < j to avoid double counting)
    mutual_pairs = []
    for i in range(n):
        j = int(nn[i])
        if j > i and int(nn[j]) == i:
            ci = int(labels[i])
            cj = int(labels[j])
            if ci == -1 or cj == -1:
                tag = "noise"
            elif ci == cj:
                tag = "semantic"
            else:
                tag = "artifact"
            mutual_pairs.append({
                "i": i,
                "j": j,
                "tok_i": tokens[i] if i < len(tokens) else "?",
                "tok_j": tokens[j] if j < len(tokens) else "?",
                "cluster_i": ci,
                "cluster_j": cj,
                "tag": tag,
            })

    n_semantic = sum(1 for p in mutual_pairs if p["tag"] == "semantic")
    n_artifact = sum(1 for p in mutual_pairs if p["tag"] == "artifact")
    n_noise = sum(1 for p in mutual_pairs if p["tag"] == "noise")
    total = len(mutual_pairs)

    return {
        "mutual_pairs": mutual_pairs,
        "n_semantic": n_semantic,
        "n_artifact": n_artifact,
        "n_noise": n_noise,
        "artifact_fraction": n_artifact / total if total > 0 else 0.0,
    }
