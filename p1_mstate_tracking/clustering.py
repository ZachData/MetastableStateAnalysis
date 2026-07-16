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
        hdb_labels = hdb.fit_predict(cos_dist.astype(np.float64))
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
    from core.metrics import fiedler_and_eigengap as spectral_eigengap_k

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
    emb_gram: np.ndarray = None,
    ext_sem_threshold: float = 0.5,
    ) -> dict:
    """
    Tag mutual nearest-neighbour pairs as semantic vs attention artifact.

    Axis 1 — cross-method agreement (HDBSCAN vs mutual-NN):
      "same_cluster"  both tokens share the same HDBSCAN cluster
      "diff_cluster"  mutual NNs in different clusters (likely induction/artifact)
      "noise"         either token is HDBSCAN noise (-1)

    Axis 2 — external semantic signal (Fix 1):
      Uses emb_gram (cosine-similarity Gram matrix of the layer-0 / embedding
      activations) as a model-independent semantic reference.
      "ext_semantic"     emb_gram[i,j] >  ext_sem_threshold  → similar at embedding
      "ext_non_semantic" emb_gram[i,j] <= ext_sem_threshold  → dissimilar at embedding
      "unknown"          emb_gram is None

    Cross metric:
      ext_sem_same_cluster_frac — among ext_semantic pairs, fraction in same cluster.
      High value ⇒ semantically close tokens also end up in same HDBSCAN cluster.

    Parameters
    ----------
    nn_indices        : (n_tokens,) int array
    hdbscan_labels    : (n_tokens,) int array  (-1 = noise)
    tokens            : list of str
    emb_gram          : (n_tokens, n_tokens) float32 or None  (Fix 1)
    ext_sem_threshold : cosine-sim cutoff for ext_semantic label (default 0.5)

    Returns
    -------
    dict with keys:
      mutual_pairs              list of per-pair dicts (see below)
      n_same_cluster            int
      n_diff_cluster            int
      n_noise                   int
      artifact_fraction         float  (= n_diff_cluster / total)
      n_ext_semantic            int
      n_ext_non_semantic        int
      n_ext_unknown             int
      ext_semantic_fraction     float | None
      ext_sem_same_cluster_frac float | None
      n_semantic                int  (alias for n_same_cluster, backward-compat)
      n_artifact                int  (alias for n_diff_cluster, backward-compat)

    Each pair dict contains:
      i, j                  token indices
      tok_i, tok_j          token strings
      cluster_i, cluster_j  HDBSCAN cluster IDs
      cross_method_tag      "same_cluster" | "diff_cluster" | "noise"
      ext_semantic_tag      "ext_semantic" | "ext_non_semantic" | "unknown"
      tag                   alias for cross_method_tag (backward-compat)
    """
    n      = len(nn_indices)
    nn     = np.asarray(nn_indices,   dtype=np.int32)
    labels = np.asarray(hdbscan_labels, dtype=np.int32)

    mutual_pairs = []
    for i in range(n):
        j = int(nn[i])
        if j <= i or int(nn[j]) != i:
            continue

        ci = int(labels[i])
        cj = int(labels[j])

        if ci == -1 or cj == -1:
            cross_tag = "noise"
        elif ci == cj:
            cross_tag = "same_cluster"
        else:
            cross_tag = "diff_cluster"

        if emb_gram is not None:
            sim = float(emb_gram[i, j])
            ext_tag = "ext_semantic" if sim > ext_sem_threshold else "ext_non_semantic"
        else:
            ext_tag = "unknown"

        mutual_pairs.append({
            "i":               i,
            "j":               j,
            "tok_i":           tokens[i] if i < len(tokens) else "?",
            "tok_j":           tokens[j] if j < len(tokens) else "?",
            "cluster_i":       ci,
            "cluster_j":       cj,
            "cross_method_tag": cross_tag,
            "ext_semantic_tag": ext_tag,
            "tag":             cross_tag,   # backward-compat alias
        })

    n_same  = sum(1 for p in mutual_pairs if p["cross_method_tag"] == "same_cluster")
    n_diff  = sum(1 for p in mutual_pairs if p["cross_method_tag"] == "diff_cluster")
    n_noise = sum(1 for p in mutual_pairs if p["cross_method_tag"] == "noise")
    total   = len(mutual_pairs)

    n_ext_sem     = sum(1 for p in mutual_pairs if p["ext_semantic_tag"] == "ext_semantic")
    n_ext_non_sem = sum(1 for p in mutual_pairs if p["ext_semantic_tag"] == "ext_non_semantic")
    n_ext_unk     = sum(1 for p in mutual_pairs if p["ext_semantic_tag"] == "unknown")

    ext_denom = n_ext_sem + n_ext_non_sem
    ext_sem_frac = (n_ext_sem / ext_denom) if ext_denom > 0 else None

    # Among ext_semantic pairs, what fraction share a HDBSCAN cluster?
    ext_sem_pairs = [p for p in mutual_pairs if p["ext_semantic_tag"] == "ext_semantic"]
    if ext_sem_pairs:
        ext_sem_same_frac = sum(
            1 for p in ext_sem_pairs if p["cross_method_tag"] == "same_cluster"
        ) / len(ext_sem_pairs)
    else:
        ext_sem_same_frac = None

    return {
        "mutual_pairs":              mutual_pairs,
        "n_same_cluster":            n_same,
        "n_diff_cluster":            n_diff,
        "n_noise":                   n_noise,
        "artifact_fraction":         n_diff / total if total > 0 else 0.0,
        "n_ext_semantic":            n_ext_sem,
        "n_ext_non_semantic":        n_ext_non_sem,
        "n_ext_unknown":             n_ext_unk,
        "ext_semantic_fraction":     ext_sem_frac,
        "ext_sem_same_cluster_frac": ext_sem_same_frac,
        # backward-compat aliases
        "n_semantic":                n_same,
        "n_artifact":                n_diff,
    }