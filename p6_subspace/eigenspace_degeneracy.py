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

Bug fixes applied in this version
-----------------------------------
1. Weighted degeneracy ratio (degeneracy_ratio)
   var_within previously used an unweighted mean over clusters, giving equal
   weight to a 3-token and a 300-token cluster. Small clusters have
   artificially low within-variance (3 points are close to their own mean),
   which inflates the ratio. Both var_within and var_between now use
   np.average(..., weights=cluster_sizes).

2. Averaged LDA alignment across all pairs (run_eigenspace_degeneracy)
   P6-R2 previously selected the single most separable cluster pair (highest
   LDA score) for the alignment test. Cherry-picking the maximum inflates the
   pass rate because some pair will always align with U_neg by chance. Now
   computes LDA alignment for every pair (capped at N_LDA_PAIRS_MAX=50 for
   large K) and reports the mean. The verdict uses mean(align_neg) >
   mean(align_imag), which is an unbiased test of the prediction.

Functions
---------
project_to_subspace       : project token matrix onto a basis
degeneracy_ratio          : within/between cluster variance in a subspace
degeneracy_sweep          : sweep k from 1..max_k for U_pos and random baseline
lda_direction             : Fisher LDA direction for two clusters
subspace_alignment        : cosine alignment between a direction and a subspace, dimension-normalised
run_eigenspace_degeneracy : full pipeline → SubResult
"""

import numpy as np
from scipy.stats import spearmanr

from p6_subspace.p6_io import SubResult, _fmt, _bullet, _verdict_line, SEP_THICK, SEP_THIN

# Cap on number of cluster pairs to evaluate per layer for LDA alignment.
# Avoids O(K²) cost when K is large. Pairs sampled reproducibly.
N_LDA_PAIRS_MAX = 50


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

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
    return X @ basis


# ---------------------------------------------------------------------------
# Within/between cluster variance
# ---------------------------------------------------------------------------

def degeneracy_ratio(
    Z:      np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Compute within-cluster and between-cluster variance in projection space Z.

    Noise tokens (label == -1) are excluded.

    Returns
    -------
    dict with ratio, var_within, var_between, n_clusters, n_tokens.
    ratio = var_between / var_within  (higher = more degenerate within clusters)
    None if fewer than 2 valid clusters.
    """
    valid = labels >= 0
    Z_v   = Z[valid].astype(np.float32)
    L_v   = labels[valid]

    unique_clusters = [int(c) for c in np.unique(L_v) if c >= 0]
    n_clusters      = len(unique_clusters)

    if n_clusters < 2:
        return {
            "ratio":      None,
            "var_within": None,
            "var_between": None,
            "n_clusters": n_clusters,
            "n_tokens":   int(valid.sum()),
        }

    cluster_sizes = np.array([int((L_v == c).sum()) for c in unique_clusters], dtype=float)
    centroids     = np.stack([Z_v[L_v == c].mean(axis=0) for c in unique_clusters])
    global_mean   = np.average(centroids, axis=0, weights=cluster_sizes)

    # Between-cluster variance: weighted mean squared centroid distance from global mean.
    # Fix 1a: weight by cluster size so large clusters dominate the between estimate.
    var_between = float(np.average(
        np.sum((centroids - global_mean) ** 2, axis=1),
        weights=cluster_sizes,
    ))

    # Within-cluster variance: for each cluster, mean squared token distance from centroid.
    # Fix 1b: weight the per-cluster variances by cluster size (was: np.mean = equal weight).
    within_vars = np.array([
        float(np.mean(np.sum((Z_v[L_v == c] - centroids[i]) ** 2, axis=1)))
        for i, c in enumerate(unique_clusters)
    ])
    var_within = float(np.average(within_vars, weights=cluster_sizes))

    ratio = var_between / max(var_within, 1e-12)

    return {
        "ratio":       ratio,
        "var_within":  var_within,
        "var_between": var_between,
        "n_clusters":  n_clusters,
        "n_tokens":    int(valid.sum()),
    }


# ---------------------------------------------------------------------------
# Degeneracy sweep over k
# ---------------------------------------------------------------------------

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

    Returns
    -------
    dict keyed by k, each containing ratio for 'pos', 'neg', 'imag', 'random_mean'.
    """
    if k_values is None:
        k_values = [1, 2, 4, 8, 16, 32]

    d       = X.shape[1]
    results = {}

    for k in k_values:
        row = {"k": k}

        for tag, basis in [("pos", U_pos), ("neg", U_neg), ("imag", U_A)]:
            if basis.shape[1] >= k:
                Z  = project_to_subspace(X, basis[:, :k])
                dr = degeneracy_ratio(Z, labels)
                row[f"ratio_{tag}"] = dr["ratio"]
            else:
                row[f"ratio_{tag}"] = None

        # Random baseline: mean ratio over n_random random orthonormal projections
        rng         = np.random.default_rng(seed=42)
        rand_ratios = []
        for _ in range(n_random):
            Q, _ = np.linalg.qr(rng.standard_normal((d, max(k, 1))))
            Z    = X @ Q[:, :k]
            dr   = degeneracy_ratio(Z, labels)
            if dr["ratio"] is not None:
                rand_ratios.append(dr["ratio"])
        row["ratio_random_mean"] = float(np.mean(rand_ratios)) if rand_ratios else None

        results[k] = row

    return results


# ---------------------------------------------------------------------------
# LDA direction
# ---------------------------------------------------------------------------

def lda_direction(
    X:      np.ndarray,
    labels: np.ndarray,
    c1:     int,
    c2:     int,
) -> np.ndarray | None:
    """
    Compute Fisher LDA direction separating clusters c1 and c2.

    Returns
    -------
    w : (d,) unit vector — the Fisher discriminant direction
    None if either cluster has fewer than 2 tokens.
    """
    mask1 = labels == c1
    mask2 = labels == c2

    if mask1.sum() < 2 or mask2.sum() < 2:
        return None

    X1, X2   = X[mask1], X[mask2]
    mu1, mu2 = X1.mean(axis=0), X2.mean(axis=0)
    mu_diff  = mu1 - mu2

    S_w = (X1 - mu1).T @ (X1 - mu1) + (X2 - mu2).T @ (X2 - mu2)

    try:
        S_w_inv = np.linalg.pinv(S_w + 1e-6 * np.eye(S_w.shape[0]))
        w       = S_w_inv @ mu_diff
    except np.linalg.LinAlgError:
        w = mu_diff

    norm = np.linalg.norm(w)
    if norm < 1e-12:
        return None
    return w / norm


# ---------------------------------------------------------------------------
# Subspace alignment
# ---------------------------------------------------------------------------

def subspace_alignment(w: np.ndarray, basis: np.ndarray) -> float:
    """
    Raw squared projection of unit vector w onto span(basis).
    Returns float in [0, 1].  NOT comparable across subspaces of different
    dimensions — use subspace_alignment_normed for cross-subspace comparisons.
    """
    if basis.shape[1] == 0:
        return 0.0
    proj = basis.T @ w
    return float(np.dot(proj, proj))

def subspace_alignment_normed(w: np.ndarray, basis: np.ndarray) -> float:
    """
    Mean squared projection per basis direction.

    Divides the raw alignment by the subspace dimension, making the result
    comparable across subspaces of different sizes.

    A random unit vector in R^d has expected value 1/d in both numerator
    (raw alignment) and denominator (dim), so the normalised value expected
    under the null is 1/d regardless of subspace size.  After normalisation,
    a subspace that genuinely contains the direction will score near 1.0;
    one that is irrelevant will score near 1/d for any dim.

    Returns float in [0, 1].
    """
    if basis.shape[1] == 0:
        return 0.0
    proj = basis.T @ w
    return float(np.dot(proj, proj)) / basis.shape[1]

# ---------------------------------------------------------------------------
# Full pipeline → SubResult
# ---------------------------------------------------------------------------

def run_eigenspace_degeneracy(ctx: dict) -> SubResult:
    """
    Track B sub-experiment: eigenspace degeneracy ratio sweep + LDA alignment.

    Required ctx keys
    -----------------
    activations_per_layer : list of (n_tokens, d_model) — one per layer/iter
    labels_per_layer      : list of (n_tokens,) int HDBSCAN labels
    layer_type_labels     : list of str — "plateau" | "merge" | "other" per layer
    projectors            : output of subspace_build.build_global_projectors
    layer_names           : list of str

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

    proj_entries = projectors["per_layer"]
    if len(proj_entries) == 1 and len(acts_per_layer) > 1:
        proj_entries = proj_entries * len(acts_per_layer)

    k_values = ctx.get("k_sweep", [1, 2, 4, 8, 16, 32])
    n_random = ctx.get("n_random_baselines", 5)

    per_layer_results = []

    for L, (X, labels, ltype, lname, pe) in enumerate(zip(
        acts_per_layer, labels_per_layer, layer_types, layer_names, proj_entries
    )):
        U_pos = pe["U_pos"]
        U_neg = pe["U_neg"]
        U_A   = pe["U_A"]

        # B.2 — degeneracy sweep
        sweep = degeneracy_sweep(X, labels, U_pos, U_neg, U_A, k_values, n_random)

        # B.3 — LDA alignment averaged across all cluster pairs
        # Fix 2: was cherry-picking the single most separable pair, inflating P6-R2.
        # Now averages alignment across all pairs (capped at N_LDA_PAIRS_MAX).
        unique_clusters = [int(c) for c in np.unique(labels) if c >= 0]

        lda_align_neg  = None
        lda_align_imag = None
        lda_align_pos  = None
        lda_n_pairs    = 0

        if len(unique_clusters) >= 2:
            all_pairs = [
                (unique_clusters[i], unique_clusters[j])
                for i in range(len(unique_clusters))
                for j in range(i + 1, len(unique_clusters))
            ]
            # Cap pair count to avoid O(K²) cost; sample reproducibly by layer index
            if len(all_pairs) > N_LDA_PAIRS_MAX:
                rng_pairs = np.random.default_rng(seed=L)
                idx       = rng_pairs.choice(len(all_pairs), N_LDA_PAIRS_MAX, replace=False)
                all_pairs = [all_pairs[int(i)] for i in idx]

            align_neg_vals  = []
            align_imag_vals = []
            align_pos_vals  = []

            for c1, c2 in all_pairs:
                w = lda_direction(X, labels, c1, c2)
                if w is None:
                    continue
                align_neg_vals.append(subspace_alignment_normed(w, U_neg))
                align_imag_vals.append(subspace_alignment_normed(w, U_A))
                align_pos_vals.append(subspace_alignment_normed(w, U_pos))

            if align_neg_vals:
                lda_align_neg  = float(np.mean(align_neg_vals))
                lda_align_imag = float(np.mean(align_imag_vals))
                lda_align_pos  = float(np.mean(align_pos_vals))
                lda_n_pairs    = len(align_neg_vals)

        # P6-R2: mean LDA aligns more with U_neg than U_A
        p6_r2 = None
        if lda_align_neg is not None and lda_align_imag is not None:
            p6_r2 = bool(lda_align_neg > lda_align_imag)

        # Summary degeneracy ratio at largest available k
        best_k = max(
            [k for k in k_values if sweep[k].get("ratio_pos") is not None],
            default=None,
        )
        ratio_pos_best  = sweep[best_k]["ratio_pos"]       if best_k else None
        ratio_rand_best = sweep[best_k]["ratio_random_mean"] if best_k else None

        # P6-R1: ratio >= 5 at plateau layers
        p6_r1 = None
        if ratio_pos_best is not None and ltype == "plateau":
            p6_r1 = bool(ratio_pos_best >= 5.0)

        per_layer_results.append({
            "layer_name":        lname,
            "layer_type":        ltype,
            "degeneracy_sweep":  {str(k): sweep[k] for k in k_values},
            "ratio_pos_best_k":  ratio_pos_best,
            "ratio_rand_best_k": ratio_rand_best,
            "best_k":            best_k,
            "lda_n_pairs":       lda_n_pairs,
            "lda_align_neg":     lda_align_neg,
            "lda_align_pos":     lda_align_pos,
            "lda_align_imag":    lda_align_imag,
            "null_align":        1.0 / projectors["d_model"],   # expected value for a random direction in R^d, projectors["d_model"] = dimentionality
            "p6_r1":             p6_r1,
            "p6_r2":             p6_r2,
        })

    if not per_layer_results:
        return SubResult(
            name="eigenspace_degeneracy",
            applicable=False,
            payload={},
            summary_lines=["eigenspace_degeneracy: no layers processed"],
            verdict_contribution={},
        )

    # Aggregate across layers
    plateau_layers = [r for r in per_layer_results if r["layer_type"] == "plateau"]
    merge_layers   = [r for r in per_layer_results if r["layer_type"] == "merge"]

    def _safe_mean(vals):
        v = [x for x in vals if x is not None]
        return float(np.mean(v)) if v else None

    mean_ratio_plateau = _safe_mean([r["ratio_pos_best_k"] for r in plateau_layers])
    mean_ratio_merge   = _safe_mean([r["ratio_pos_best_k"] for r in merge_layers])
    mean_ratio_rand    = _safe_mean([r["ratio_rand_best_k"] for r in per_layer_results])
    mean_lda_neg       = _safe_mean([r["lda_align_neg"]     for r in per_layer_results])
    mean_lda_imag      = _safe_mean([r["lda_align_imag"]    for r in per_layer_results])

    n_p6r1_pass = sum(1 for r in plateau_layers    if r["p6_r1"] is True)
    n_p6r2_pass = sum(1 for r in per_layer_results if r["p6_r2"] is True)

    total_lda_pairs = sum(r["lda_n_pairs"] for r in per_layer_results)

    payload = {
        "n_layers":             len(per_layer_results),
        "n_plateau_layers":     len(plateau_layers),
        "n_merge_layers":       len(merge_layers),
        "mean_ratio_plateau":   mean_ratio_plateau,
        "mean_ratio_merge":     mean_ratio_merge,
        "mean_ratio_random":    mean_ratio_rand,
        "mean_lda_align_neg":   mean_lda_neg,
        "mean_lda_align_imag":  mean_lda_imag,
        "total_lda_pairs_used": total_lda_pairs,
        "n_p6r1_pass":          n_p6r1_pass,
        "n_p6r2_pass":          n_p6r2_pass,
        "per_layer":            per_layer_results,
    }

    lines = [
        SEP_THICK,
        "EIGENSPACE DEGENERACY + LDA ALIGNMENT  [Track B]",
        SEP_THICK,
        f"Layers analysed:       {len(per_layer_results)}",
        f"  plateau layers:      {len(plateau_layers)}",
        f"  merge layers:        {len(merge_layers)}",
        "",
        "B.2 — Degeneracy ratio R = σ_B² / σ_W² in U_pos subspace",
        "  (both σ² weighted by cluster size):",
        _bullet("mean R at plateau layers", mean_ratio_plateau),
        _bullet("mean R at merge layers",   mean_ratio_merge),
        _bullet("mean R (random baseline)", mean_ratio_rand),
        "",
        "Prediction P6-R1: R >= 5 at plateau layers, near 1 for random projection.",
        _bullet("plateau layers with R >= 5", n_p6r1_pass),
        _bullet("total plateau layers",       len(plateau_layers)),
        _verdict_line(
            "P6-R1",
            bool(n_p6r1_pass > len(plateau_layers) // 2) if plateau_layers else None,
            f"mean R_plateau={_fmt(mean_ratio_plateau)} R_rand={_fmt(mean_ratio_rand)}",
        ),
        "",
        "B.3 — LDA alignment: mean across all cluster pairs per layer",
        f"  (capped at {N_LDA_PAIRS_MAX} pairs/layer; {total_lda_pairs} pair-evals total):",
        _bullet("mean LDA align with U_neg (repulsive S)", mean_lda_neg),
        _bullet("mean LDA align with U_A  (imaginary)",    mean_lda_imag),
        "",
        "Prediction P6-R2: LDA aligns more with U_neg than U_A.",
        _bullet("layers where mean align_neg > mean align_imag", n_p6r2_pass),
        _verdict_line(
            "P6-R2",
            bool(n_p6r2_pass > len(per_layer_results) // 2) if per_layer_results else None,
            f"mean neg={_fmt(mean_lda_neg)} vs mean imag={_fmt(mean_lda_imag)}",
        ),
        "",
        "Per-layer detail (ratio_pos @ best_k | lda_align_neg | lda_align_imag | n_pairs):",
    ]
    for r in per_layer_results:
        lines.append(
            f"  {r['layer_name']:<18s} [{r['layer_type']:<7s}]  "
            f"R={_fmt(r['ratio_pos_best_k'])} (k={r['best_k']})  "
            f"lda_neg={_fmt(r['lda_align_neg'])}  "
            f"lda_imag={_fmt(r['lda_align_imag'])}  "
            f"pairs={r['lda_n_pairs']}"
        )

    first_plateau = next((r for r in per_layer_results if r["layer_type"] == "plateau"), None)
    if first_plateau:
        lines += [
            "",
            f"Degeneracy sweep (example: {first_plateau['layer_name']}):",
            f"  {'k':>4}  {'R_pos':>8}  {'R_neg':>8}  {'R_imag':>8}  {'R_rand':>8}",
        ]
        for k in k_values:
            row = first_plateau["degeneracy_sweep"].get(str(k), {})
            lines.append(
                f"  {k:>4}  "
                f"{_fmt(row.get('ratio_pos')):>8}  "
                f"{_fmt(row.get('ratio_neg')):>8}  "
                f"{_fmt(row.get('ratio_imag')):>8}  "
                f"{_fmt(row.get('ratio_random_mean')):>8}"
            )

    vc = {
        "deg_mean_ratio_plateau":  mean_ratio_plateau,
        "deg_mean_ratio_merge":    mean_ratio_merge,
        "deg_mean_ratio_random":   mean_ratio_rand,
        "deg_mean_lda_align_neg":  mean_lda_neg,
        "deg_mean_lda_align_imag": mean_lda_imag,
        "deg_n_p6r1_pass":         n_p6r1_pass,
        "deg_n_p6r2_pass":         n_p6r2_pass,
        "deg_p6_r1_satisfied":     bool(n_p6r1_pass > len(plateau_layers) // 2)
                                   if plateau_layers else False,
        "deg_p6_r2_satisfied":     bool(n_p6r2_pass > len(per_layer_results) // 2)
                                   if per_layer_results else False,
    }

    return SubResult(
        name="eigenspace_degeneracy",
        applicable=True,
        payload=payload,
        summary_lines=lines,
        verdict_contribution=vc,
    )
