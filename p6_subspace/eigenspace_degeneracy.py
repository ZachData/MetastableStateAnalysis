"""
eigenspace_degeneracy.py — Track B: Eigenspace degeneracy and LDA alignment.

Two closely related tests on whether cluster structure lives in the real (S)
subspace:

B.2 — Eigenspace degeneracy ratio
  Project tokens onto the top-k attractive eigenvectors of S (U_pos).
  Measure within-cluster vs between-cluster variance in that projection.
  High ratio = tokens in the same cluster are nearly degenerate in S's
  eigenspace, i.e., they look identical to the Geshkovski dynamics.

B.3 — LDA alignment with S repulsive subspace
  The LDA direction separating two clusters should align with U_neg
  (repulsive subspace of S) more than with U_A (imaginary subspace).
  Intuition: clusters are separated along the directions S repels tokens —
  which is exactly where merge events reduce separation.

Both tests are run per-layer for per-layer models and per-iteration for ALBERT.

Falsifiable predictions tested
-------------------------------
P6-R1 : degeneracy ratio R >= 5 at plateau layers; near 1 for random projection.
P6-R2 : LDA direction aligns more with U_neg than with U_A.

Functions
---------
project_to_subspace       : project token matrix onto a basis
degeneracy_ratio          : within/between cluster variance in a subspace
lda_direction             : Fisher LDA direction for two clusters
subspace_alignment        : cosine alignment between a direction and a subspace
run_eigenspace_degeneracy : full pipeline → SubResult

Fixes:
  1a/1b : Weighted degeneracy_ratio (cluster sizes, not equal weight)
  2     : Averaged LDA alignment across all pairs (not cherry-picked max)
"""

import numpy as np
from scipy.stats import spearmanr

N_LDA_PAIRS_MAX = 50


# -----------
# Projection helpers
# -----------

def project_to_subspace(
    X:     np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """
    Project rows of X (n, d) onto an orthonormal basis (d, r).

    Returns
    -------
    Z : (n, r)
    """
    return X @ basis   # (n, r)


# -----------
# Within/between cluster variance
# -----------

def degeneracy_ratio(
    Z:      np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Compute within-cluster and between-cluster variance in projection space Z.

    Noise tokens (label == -1) are excluded.

    FIX 1a/1b: Weighted variance using cluster_sizes (not equal weight).
    Small clusters have artificially low within-variance (few points close to mean).
    Both var_within and var_between now use np.average(..., weights=cluster_sizes).

    Returns
    -------
    dict with:
      var_within   : float — mean within-cluster variance (weighted by cluster size)
      var_between  : float — between-cluster variance (centroid spread, weighted)
      ratio        : float — var_between / var_within  (R in the spec)
      n_clusters   : int
      n_tokens     : int — non-noise tokens used
    """
    valid = labels >= 0
    Z_v   = Z[valid]
    L_v   = labels[valid]

    cluster_ids = np.unique(L_v)
    n_clusters  = len(cluster_ids)

    if n_clusters < 2 or len(Z_v) < 4:
        return {"var_within": None, "var_between": None, "ratio": None,
                "n_clusters": n_clusters, "n_tokens": int(valid.sum())}

    cluster_sizes = np.array(
        [int((L_v == c).sum()) for c in cluster_ids], dtype=float
    )
    centroids = np.stack([Z_v[L_v == c].mean(axis=0) for c in cluster_ids])
    global_mean = np.average(centroids, axis=0, weights=cluster_sizes)

    # FIX 1a: var_between weighted by cluster size
    var_between = float(
        np.average(
            np.sum((centroids - global_mean) ** 2, axis=1),
            weights=cluster_sizes,
        )
    )

    # FIX 1b: var_within weighted by cluster size (was: equal weight)
    within_vars = np.array([
        float(np.mean(np.sum((Z_v[L_v == c] - centroids[i]) ** 2, axis=1)))
        for i, c in enumerate(cluster_ids)
    ])
    var_within = float(np.average(within_vars, weights=cluster_sizes))

    ratio = var_between / max(var_within, 1e-12)

    return {
        "var_within":  var_within,
        "var_between": var_between,
        "ratio":       ratio,
        "n_clusters":  n_clusters,
        "n_tokens":    int(valid.sum()),
    }


# -----------
# Degeneracy sweep over k
# -----------

def degeneracy_sweep(
    X:          np.ndarray,
    labels:     np.ndarray,
    U_pos:      np.ndarray,
    U_neg:      np.ndarray,
    U_A:        np.ndarray,
    k_values:   list[int] | None = None,
    n_random:   int = 5,
) -> dict:
    """
    Sweep k (number of basis vectors) and compute degeneracy ratio for:
      - Top-k attractive eigenvectors of S (U_pos[:, :k])
      - Top-k repulsive eigenvectors of S (U_neg[:, :k])
      - Top-k imaginary planes (U_A[:, :k])
      - Random orthonormal subspaces of same dimension (baseline)

    Parameters
    ----------
    X        : (n, d) L2-normed activations
    labels   : (n,)  HDBSCAN cluster labels
    U_pos    : (d, r_pos) attractive basis
    U_neg    : (d, r_neg) repulsive basis
    U_A      : (d, r_A)  imaginary basis
    k_values : list of k to sweep (default [1,2,4,8,16,32])
    n_random : number of random subspace baselines per k

    Returns
    -------
    dict keyed by k, each containing ratio for 'pos', 'neg', 'imag', 'random_mean'
    """
    if k_values is None:
        k_values = [1, 2, 4, 8, 16, 32]

    d = X.shape[1]
    results = {}

    for k in k_values:
        row = {"k": k}

        for tag, basis in [("pos", U_pos), ("neg", U_neg), ("imag", U_A)]:
            if basis.shape[1] >= k:
                Z = project_to_subspace(X, basis[:, :k])
                dr = degeneracy_ratio(Z, labels)
                row[f"ratio_{tag}"] = dr["ratio"]
            else:
                row[f"ratio_{tag}"] = None

        # Random baseline: mean ratio over n_random random orthonormal projections
        rng = np.random.default_rng(seed=42)
        rand_ratios = []
        for _ in range(n_random):
            Q, _ = np.linalg.qr(rng.standard_normal((d, max(k, 1))))
            Z = X @ Q[:, :k]
            dr = degeneracy_ratio(Z, labels)
            if dr["ratio"] is not None:
                rand_ratios.append(dr["ratio"])
        row["ratio_random_mean"] = float(np.mean(rand_ratios)) if rand_ratios else None

        results[k] = row

    return results


# -----------
# LDA direction
# -----------

def lda_direction(
    X:       np.ndarray,
    labels:  np.ndarray,
    c1:      int,
    c2:      int,
) -> np.ndarray | None:
    """
    Compute Fisher LDA direction separating clusters c1 and c2.

    Returns
    -------
    w : (d,) unit vector — the Fisher discriminant direction
    None if either cluster has fewer than 2 tokens
    """
    mask1 = labels == c1
    mask2 = labels == c2

    if mask1.sum() < 2 or mask2.sum() < 2:
        return None

    X1 = X[mask1]
    X2 = X[mask2]

    mu1, mu2 = X1.mean(axis=0), X2.mean(axis=0)
    mu_diff   = mu1 - mu2

    # Within-class scatter
    S_w = ((X1 - mu1).T @ (X1 - mu1) + (X2 - mu2).T @ (X2 - mu2))

    # Regularised pseudo-inverse
    try:
        S_w_inv = np.linalg.pinv(S_w + 1e-6 * np.eye(S_w.shape[0]))
        w = S_w_inv @ mu_diff
    except np.linalg.LinAlgError:
        w = mu_diff

    norm = np.linalg.norm(w)
    if norm < 1e-12:
        return None
    return w / norm


# -----------
# Subspace alignment
# -----------

def subspace_alignment(
    w:     np.ndarray,
    basis: np.ndarray,
) -> float:
    """
    Squared cosine between direction w and the subspace spanned by basis.

    align = ||P_basis w||^2  where P_basis = basis @ basis^T (orthonormal)

    Returns
    -------
    float in [0, 1]
    """
    if basis.shape[1] == 0:
        return 0.0
    proj = basis.T @ w            # (r,) coefficients
    return float(np.dot(proj, proj))   # = ||P w||^2 since basis orthonormal


# -----------
# Full pipeline → SubResult
# -----------

def run_eigenspace_degeneracy(ctx: dict):
    """
    Track B sub-experiment: eigenspace degeneracy ratio sweep + LDA alignment.

    FIX 2: Average LDA alignment across all pairs (capped at N_LDA_PAIRS_MAX),
    not cherry-picking single most-separable pair.

    Required ctx keys
    -----------------
    activations_per_layer : list of (n_tokens, d_model) — one per layer/iter
    labels_per_layer      : list of (n_tokens,) int HDBSCAN labels
    layer_type_labels     : list of str — "plateau" | "merge" | "other" per layer
    projectors            : output of subspace_build.build_global_projectors
    layer_names           : list of str (matching activations_per_layer)

    Optional ctx keys
    -----------------
    k_sweep              : list[int] (default [1,2,4,8,16,32])
    n_random_baselines   : int (default 5)
    """
    acts_per_layer   = ctx["activations_per_layer"]
    labels_per_layer = ctx["labels_per_layer"]
    layer_types      = ctx["layer_type_labels"]
    projectors       = ctx["projectors"]
    layer_names      = ctx["layer_names"]

    # For ALBERT (single projector entry) broadcast across iterations
    proj_entries = projectors["per_layer"]
    if len(proj_entries) == 1 and len(acts_per_layer) > 1:
        proj_entries = proj_entries * len(acts_per_layer)

    k_values  = ctx.get("k_sweep", [1, 2, 4, 8, 16, 32])
    n_random  = ctx.get("n_random_baselines", 5)

    per_layer_results = []

    for L, (X, labels, ltype, lname, pe) in enumerate(zip(
        acts_per_layer, labels_per_layer, layer_types, layer_names, proj_entries
    )):
        U_pos = pe["U_pos"]
        U_neg = pe["U_neg"]
        U_A   = pe["U_A"]

        # B.2 — degeneracy sweep
        sweep = degeneracy_sweep(X, labels, U_pos, U_neg, U_A, k_values, n_random)

        # B.3 — LDA alignment: average across all pairs (FIX 2)
        unique_clusters = [c for c in np.unique(labels) if c >= 0]
        lda_align_neg = None
        lda_align_imag = None
        lda_n_pairs = 0

        if len(unique_clusters) >= 2:
            all_pairs = [
                (unique_clusters[i], unique_clusters[j])
                for i in range(len(unique_clusters))
                for j in range(i + 1, len(unique_clusters))
            ]
            # Cap pairs at N_LDA_PAIRS_MAX
            if len(all_pairs) > N_LDA_PAIRS_MAX:
                rng_pairs = np.random.default_rng(seed=L)
                idx = rng_pairs.choice(len(all_pairs), N_LDA_PAIRS_MAX, replace=False)
                all_pairs = [all_pairs[int(i)] for i in idx]

            align_neg_vals = []
            align_imag_vals = []

            for c1, c2 in all_pairs:
                w = lda_direction(X, labels, c1, c2)
                if w is None:
                    continue
                align_neg_vals.append(subspace_alignment(w, U_neg))
                align_imag_vals.append(subspace_alignment(w, U_A))

            if align_neg_vals:
                lda_align_neg = float(np.mean(align_neg_vals))
                lda_align_imag = float(np.mean(align_imag_vals))
                lda_n_pairs = len(align_neg_vals)

        p6_r2 = None
        if lda_align_neg is not None and lda_align_imag is not None:
            p6_r2 = bool(lda_align_neg > lda_align_imag)

        # Summary degeneracy ratio at k=max available
        best_k = max([k for k in k_values if sweep[k].get("ratio_pos") is not None],
                     default=None)
        ratio_pos_best = sweep[best_k]["ratio_pos"] if best_k else None
        ratio_rand_best = sweep[best_k]["ratio_random_mean"] if best_k else None

        # P6-R1: ratio >= 5 at plateau layers
        p6_r1 = None
        if ratio_pos_best is not None and ltype == "plateau":
            p6_r1 = ratio_pos_best >= 5.0

        per_layer_results.append({
            "layer_name":      lname,
            "layer_type":      ltype,
            "degeneracy_sweep": {str(k): sweep[k] for k in k_values},
            "ratio_pos_best_k":  ratio_pos_best,
            "ratio_rand_best_k": ratio_rand_best,
            "best_k":            best_k,
            "lda_n_pairs":       lda_n_pairs,
            "lda_align_neg":     lda_align_neg,
            "lda_align_imag":    lda_align_imag,
            "p6_r1":             p6_r1,
            "p6_r2":             p6_r2,
        })

    # Aggregate across layers
    plateau_layers = [r for r in per_layer_results if r["layer_type"] == "plateau"]

    def _safe_mean(vals):
        v = [x for x in vals if x is not None]
        return float(np.mean(v)) if v else None

    mean_ratio_plateau = _safe_mean([r["ratio_pos_best_k"] for r in plateau_layers])
    mean_lda_neg       = _safe_mean([r["lda_align_neg"]  for r in per_layer_results])
    mean_lda_imag      = _safe_mean([r["lda_align_imag"] for r in per_layer_results])

    n_p6r1_pass = sum(1 for r in plateau_layers if r["p6_r1"] is True)
    n_p6r2_pass = sum(1 for r in per_layer_results if r["p6_r2"] is True)

    # Return a SubResult-compatible dict (update as needed for your framework)
    return {
        "name": "eigenspace_degeneracy",
        "applicable": True,
        "payload": {
            "n_layers": len(per_layer_results),
            "n_plateau_layers": len(plateau_layers),
            "mean_ratio_plateau": mean_ratio_plateau,
            "mean_lda_align_neg": mean_lda_neg,
            "mean_lda_align_imag": mean_lda_imag,
            "n_p6r1_pass": n_p6r1_pass,
            "n_p6r2_pass": n_p6r2_pass,
            "per_layer": per_layer_results,
        },
        "verdict_contribution": {
            "deg_p6_r1_satisfied": n_p6r1_pass > len(plateau_layers) // 2 if plateau_layers else False,
            "deg_p6_r2_satisfied": n_p6r2_pass > len(per_layer_results) // 2 if per_layer_results else False,
        }
    }