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
degeneracy_sweep          : sweep k from 1..max_k for U_pos and random baseline
lda_direction             : Fisher LDA direction for two clusters
subspace_alignment        : cosine alignment between a direction and a subspace
run_eigenspace_degeneracy : full pipeline → SubResult
"""

import numpy as np
from scipy.stats import spearmanr

from p6_subspace.p6_io import SubResult, _fmt, _bullet, _verdict_line, SEP_THICK, SEP_THIN


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
    return X @ basis   # (n, r)


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
    dict with:
      var_within   : float — mean within-cluster variance (averaged across clusters)
      var_between  : float — between-cluster variance (centroid spread)
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

    global_mean = Z_v.mean(axis=0)   # (r,)
    centroids   = []
    within_vars = []

    for c in cluster_ids:
        mask = L_v == c
        Zc   = Z_v[mask]
        mu_c = Zc.mean(axis=0)
        centroids.append(mu_c)
        within_vars.append(float(np.mean(np.sum((Zc - mu_c) ** 2, axis=1))))

    var_within  = float(np.mean(within_vars))
    centroids   = np.stack(centroids)                                  # (K, r)
    var_between = float(np.mean(np.sum((centroids - global_mean) ** 2, axis=1)))

    ratio = var_between / max(var_within, 1e-12)

    return {
        "var_within":  var_within,
        "var_between": var_between,
        "ratio":       ratio,
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


# ---------------------------------------------------------------------------
# LDA direction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Subspace alignment
# ---------------------------------------------------------------------------

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

        # B.3 — LDA alignment: use the cluster pair with highest merge probability
        # (if merge info not available, use any two clusters)
        unique_clusters = [c for c in np.unique(labels) if c >= 0]
        best_lda = None
        best_pair = None

        if len(unique_clusters) >= 2:
            # Try all pairs, pick the one with largest LDA norm (most separable)
            best_norm = -1.0
            for ci in range(len(unique_clusters)):
                for cj in range(ci + 1, len(unique_clusters)):
                    c1, c2 = unique_clusters[ci], unique_clusters[cj]
                    w = lda_direction(X, labels, c1, c2)
                    if w is None:
                        continue
                    # Use LDA "score" = between-cluster distance in the direction
                    m1 = X[labels == c1].mean(axis=0)
                    m2 = X[labels == c2].mean(axis=0)
                    score = abs(float(w @ (m1 - m2)))
                    if score > best_norm:
                        best_norm = score
                        best_lda  = w
                        best_pair = (c1, c2)

        lda_align_neg  = subspace_alignment(best_lda, U_neg) if best_lda is not None else None
        lda_align_imag = subspace_alignment(best_lda, U_A)   if best_lda is not None else None
        lda_align_pos  = subspace_alignment(best_lda, U_pos) if best_lda is not None else None

        # P6-R2: LDA aligns more with U_neg than U_A
        p6_r2 = None
        if lda_align_neg is not None and lda_align_imag is not None:
            p6_r2 = lda_align_neg > lda_align_imag

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
            "lda_pair":          list(best_pair) if best_pair else None,
            "lda_align_neg":     lda_align_neg,
            "lda_align_pos":     lda_align_pos,
            "lda_align_imag":    lda_align_imag,
            "p6_r1":             p6_r1,
            "p6_r2":             p6_r2,
        })

    # Aggregate across layers
    plateau_layers = [r for r in per_layer_results if r["layer_type"] == "plateau"]
    merge_layers   = [r for r in per_layer_results if r["layer_type"] == "merge"]

    def _safe_mean(vals):
        v = [x for x in vals if x is not None]
        return float(np.mean(v)) if v else None

    mean_ratio_plateau = _safe_mean([r["ratio_pos_best_k"] for r in plateau_layers])
    mean_ratio_merge   = _safe_mean([r["ratio_pos_best_k"] for r in merge_layers])
    mean_ratio_rand    = _safe_mean([r["ratio_rand_best_k"] for r in per_layer_results])
    mean_lda_neg       = _safe_mean([r["lda_align_neg"]  for r in per_layer_results])
    mean_lda_imag      = _safe_mean([r["lda_align_imag"] for r in per_layer_results])

    n_p6r1_pass = sum(1 for r in plateau_layers if r["p6_r1"] is True)
    n_p6r2_pass = sum(1 for r in per_layer_results if r["p6_r2"] is True)

    payload = {
        "n_layers":            len(per_layer_results),
        "n_plateau_layers":    len(plateau_layers),
        "n_merge_layers":      len(merge_layers),
        "mean_ratio_plateau":  mean_ratio_plateau,
        "mean_ratio_merge":    mean_ratio_merge,
        "mean_ratio_random":   mean_ratio_rand,
        "mean_lda_align_neg":  mean_lda_neg,
        "mean_lda_align_imag": mean_lda_imag,
        "n_p6r1_pass":         n_p6r1_pass,
        "n_p6r2_pass":         n_p6r2_pass,
        "per_layer":           per_layer_results,
    }

    # --- Summary lines ---
    lines = [
        SEP_THICK,
        "EIGENSPACE DEGENERACY + LDA ALIGNMENT  [Track B]",
        SEP_THICK,
        f"Layers analysed:       {len(per_layer_results)}",
        f"  plateau layers:      {len(plateau_layers)}",
        f"  merge layers:        {len(merge_layers)}",
        "",
        "B.2 — Degeneracy ratio R = sigma_B^2 / sigma_W^2 in U_pos subspace:",
        _bullet("mean R at plateau layers", mean_ratio_plateau),
        _bullet("mean R at merge layers",   mean_ratio_merge),
        _bullet("mean R (random baseline)", mean_ratio_rand),
        "",
        "Prediction P6-R1: R >= 5 at plateau layers, near 1 for random projection.",
        _bullet("plateau layers with R >= 5", n_p6r1_pass),
        _bullet("total plateau layers", len(plateau_layers)),
        _verdict_line(
            "P6-R1",
            n_p6r1_pass > len(plateau_layers) // 2 if plateau_layers else None,
            f"mean R_plateau={_fmt(mean_ratio_plateau)} R_rand={_fmt(mean_ratio_rand)}",
        ),
        "",
        "B.3 — LDA alignment: cluster-separating direction vs S repulsive / imaginary:",
        _bullet("mean LDA align with U_neg (repulsive S)", mean_lda_neg),
        _bullet("mean LDA align with U_A  (imaginary)",    mean_lda_imag),
        "",
        "Prediction P6-R2: LDA aligns more with U_neg than U_A.",
        _bullet("layers where align_neg > align_imag", n_p6r2_pass),
        _verdict_line(
            "P6-R2",
            n_p6r2_pass > len(per_layer_results) // 2 if per_layer_results else None,
            f"mean neg={_fmt(mean_lda_neg)} vs mean imag={_fmt(mean_lda_imag)}",
        ),
        "",
        "Per-layer detail (ratio_pos @ best_k | lda_align_neg | lda_align_imag):",
    ]
    for r in per_layer_results:
        lines.append(
            f"  {r['layer_name']:<18s} [{r['layer_type']:<7s}]  "
            f"R={_fmt(r['ratio_pos_best_k'])} (k={r['best_k']})  "
            f"lda_neg={_fmt(r['lda_align_neg'])}  "
            f"lda_imag={_fmt(r['lda_align_imag'])}"
        )
    lines.append("")
    lines.append("Degeneracy sweep (ratio_pos | ratio_imag | ratio_random) by k:")
    # Print sweep table for first plateau layer found
    first_plateau = next((r for r in per_layer_results if r["layer_type"] == "plateau"), None)
    if first_plateau:
        lines.append(f"  (example: {first_plateau['layer_name']})")
        lines.append(f"  {'k':>4}  {'R_pos':>8}  {'R_imag':>8}  {'R_rand':>8}")
        for k in k_values:
            row = first_plateau["degeneracy_sweep"].get(str(k), {})
            lines.append(
                f"  {k:>4}  "
                f"{_fmt(row.get('ratio_pos')):>8}  "
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
        "deg_p6_r1_satisfied":     n_p6r1_pass > len(plateau_layers) // 2 if plateau_layers else False,
        "deg_p6_r2_satisfied":     n_p6r2_pass > len(per_layer_results) // 2 if per_layer_results else False,
    }

    return SubResult(
        name="eigenspace_degeneracy",
        applicable=True,
        payload=payload,
        summary_lines=lines,
        verdict_contribution=vc,
    )
