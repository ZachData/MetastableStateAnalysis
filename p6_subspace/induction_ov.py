"""
induction_ov.py — Track A: Induction score and OV write direction alignment.

Two questions answered here:

1. Which heads are induction heads?
   Score each head by how strongly attention(i, j) correlates with the
   "offset content match" pattern: attend to j when token[j-1] ≈ token[i-1].
   Heads with score above threshold are flagged as induction candidates.

2. Do induction heads write into the imaginary channel?
   For each head's W_O (output projection), compute the fraction of its
   dominant write directions that land in the imaginary (A) subspace.
   Compare induction heads to semantic (high-CC) heads.

Falsifiable predictions tested
-------------------------------
P6-I1 : Induction head OV write directions project more strongly onto the
         imaginary subspace than do semantic heads.
         Test: mean f_rot(induction heads) > mean f_rot(semantic heads)
         with MWU p < 0.05.

Functions
---------
induction_score              : per-head induction strength from attention
classify_induction_heads     : threshold-based binary classification
ov_write_alignment           : fraction of W_O singular vectors in A subspace
compare_induction_vs_semantic: P6-I1 test
run_induction_ov             : full pipeline → SubResult
"""

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from p6_subspace.p6_io import SubResult, _fmt, _bullet, _verdict_line, SEP_THICK, SEP_THIN


# ---------------------------------------------------------------------------
# Induction score
# ---------------------------------------------------------------------------

def induction_score(
    attn_weights:  np.ndarray,
    token_ids:     np.ndarray,
    sim_threshold: float = 0.7,
    token_activations: np.ndarray | None = None,
) -> float:
    """
    Measure how strongly one head follows the induction pattern.

    Score = mean attention weight A[i, j] for (i, j) induction pairs,
    minus mean attention weight for all other (i != j) pairs.

    A high positive score indicates the head preferentially attends to
    the induction target position relative to the background.

    Parameters
    ----------
    attn_weights      : (n, n) softmax attention matrix
    token_ids         : (n,) int — token IDs
    sim_threshold     : cosine threshold for content match (if activations given)
    token_activations : (n, d) L2-normed; used for soft content match if provided

    Returns
    -------
    float in roughly [-1, 1]; >0.05 is a weak positive signal
    """
    n = len(token_ids)

    # Build induction mask
    induction_mask = np.zeros((n, n), dtype=bool)
    for i in range(1, n):
        for j in range(2, n):
            if j == i:
                continue
            exact = (token_ids[j - 1] == token_ids[i - 1])
            soft  = False
            if (not exact) and (token_activations is not None):
                cos = float(token_activations[j - 1] @ token_activations[i - 1])
                soft = cos > sim_threshold
            if exact or soft:
                induction_mask[i, j] = True

    off_diag = ~np.eye(n, dtype=bool)
    non_induction_mask = off_diag & ~induction_mask

    n_ind = induction_mask.sum()
    n_non = non_induction_mask.sum()

    if n_ind == 0:
        return 0.0

    mean_ind = float(attn_weights[induction_mask].mean())
    mean_non = float(attn_weights[non_induction_mask].mean()) if n_non > 0 else 0.0

    return mean_ind - mean_non


def classify_induction_heads(
    scores:    list[float],
    threshold: float = 0.05,
) -> list[int]:
    """Return indices of heads with induction score above threshold."""
    return [i for i, s in enumerate(scores) if s > threshold]


# ---------------------------------------------------------------------------
# OV write direction alignment
# ---------------------------------------------------------------------------

def ov_write_alignment(
    WO:     np.ndarray,
    P_A:    np.ndarray,
    P_S:    np.ndarray,
    top_r:  int = 16,
) -> dict:
    """
    Compute what fraction of a head's dominant write directions land in
    each channel (imaginary A vs real S).

    The write directions of head h are the left singular vectors of W_O^(h):
    these are the directions in residual-stream space that the head writes into.

    Parameters
    ----------
    WO    : (d_model, d_head) — output projection for one head
    P_A   : (d_model, d_model) — imaginary-channel projector
    P_S   : (d_model, d_model) — real-channel projector
    top_r : number of dominant write directions to use

    Returns
    -------
    dict with:
      align_rot  : float — mean |P_A e_k|^2 over top-r left singular vectors
      align_real : float — mean |P_S e_k|^2 over top-r left singular vectors
      sing_vals  : list[float] — singular values (top_r)
    """
    U, s, _ = np.linalg.svd(WO, full_matrices=False)   # U: (d_model, d_head)
    r = min(top_r, U.shape[1])
    U_top = U[:, :r]   # (d_model, r)

    # |P_A e_k|^2 = e_k^T P_A^T P_A e_k = e_k^T P_A e_k  (projectors are idempotent)
    rot_scores  = np.array([float(U_top[:, k] @ P_A @ U_top[:, k]) for k in range(r)])
    real_scores = np.array([float(U_top[:, k] @ P_S @ U_top[:, k]) for k in range(r)])

    return {
        "align_rot":  float(rot_scores.mean()),
        "align_real": float(real_scores.mean()),
        "sing_vals":  s[:r].tolist(),
    }


# ---------------------------------------------------------------------------
# P6-I1 test
# ---------------------------------------------------------------------------

def compare_induction_vs_semantic(
    head_records:      list[dict],
    induction_indices: list[int],
    semantic_indices:  list[int],
) -> dict:
    """
    P6-I1: Induction heads have higher rotational write alignment than semantic heads.

    Parameters
    ----------
    head_records       : list of dicts, each with 'head_idx' and 'align_rot'
    induction_indices  : head indices classified as induction
    semantic_indices   : head indices classified as semantic (high CC, low PC)

    Returns
    -------
    dict with test results and P6-I1 verdict
    """
    ind_set = set(induction_indices)
    sem_set = set(semantic_indices)

    ind_vals = [r["align_rot"] for r in head_records if r["head_idx"] in ind_set]
    sem_vals = [r["align_rot"] for r in head_records if r["head_idx"] in sem_set]

    if len(ind_vals) < 2 or len(sem_vals) < 2:
        return {
            "mean_align_rot_induction": (float(np.mean(ind_vals)) if ind_vals else None),
            "mean_align_rot_semantic":  (float(np.mean(sem_vals)) if sem_vals else None),
            "mwu_pvalue":              None,
            "n_induction":             len(ind_vals),
            "n_semantic":              len(sem_vals),
            "p6_i1_satisfied":         False,
        }

    mu_ind = float(np.mean(ind_vals))
    mu_sem = float(np.mean(sem_vals))
    stat, pval = mannwhitneyu(ind_vals, sem_vals, alternative="greater")

    return {
        "mean_align_rot_induction": mu_ind,
        "mean_align_rot_semantic":  mu_sem,
        "delta_align_rot":          mu_ind - mu_sem,
        "mwu_statistic":            float(stat),
        "mwu_pvalue":               float(pval),
        "n_induction":              len(ind_vals),
        "n_semantic":               len(sem_vals),
        "p6_i1_satisfied":          (mu_ind > mu_sem and float(pval) < 0.05),
    }


# ---------------------------------------------------------------------------
# Full pipeline → SubResult
# ---------------------------------------------------------------------------

def run_induction_ov(ctx: dict) -> SubResult:
    """
    Track A sub-experiment: induction detection + OV write direction alignment.

    Required ctx keys
    -----------------
    attn_matrices       : list of (n, n) softmax attention per head
    wo_matrices         : list of (d_model, d_head) W_O per head
    token_ids           : (n,) int
    token_activations   : (n, d_model) L2-normed
    projectors          : output of subspace_build.build_global_projectors,
                          used as projectors["per_layer"][layer_idx]
    layer_idx           : int (default 0 for ALBERT)

    Optional ctx keys
    -----------------
    head_classify_result: output of head_classify.py — used to identify
                          semantic heads for P6-I1 comparison
    induction_threshold : float (default 0.05)
    layer_name          : str
    """
    attn_matrices     = ctx["attn_matrices"]
    wo_matrices       = ctx["wo_matrices"]
    token_ids         = np.asarray(ctx["token_ids"])
    X                 = ctx["token_activations"]
    projectors        = ctx["projectors"]
    layer_idx         = ctx.get("layer_idx", 0)
    layer_name        = ctx.get("layer_name", "shared")
    ind_threshold     = ctx.get("induction_threshold", 0.05)

    proj_entry = projectors["per_layer"][layer_idx]
    P_A = proj_entry["P_A"]
    P_S = proj_entry["P_S"]

    n_heads = len(attn_matrices)

    # 1. Induction scores
    scores = [
        induction_score(attn_matrices[h], token_ids, token_activations=X)
        for h in range(n_heads)
    ]
    induction_idx = classify_induction_heads(scores, ind_threshold)

    # 2. OV write alignment per head
    alignments = [
        ov_write_alignment(wo_matrices[h], P_A, P_S)
        for h in range(n_heads)
    ]

    # 3. Semantic head indices from head_classify (if available)
    hc_result = ctx.get("head_classify_result")
    if hc_result and "head_records" in hc_result:
        semantic_idx = [
            r["head_idx"] for r in hc_result["head_records"]
            if r.get("quadrant") == "semantic"
        ]
    else:
        # Fallback: heads with low induction score are proxies for semantic
        semantic_idx = [
            h for h, s in enumerate(scores)
            if s < 0.01 and h not in induction_idx
        ]

    # Merge into per-head records
    head_records = []
    for h in range(n_heads):
        head_records.append({
            "head_idx":        h,
            "induction_score": float(scores[h]),
            "is_induction":    h in induction_idx,
            "align_rot":       alignments[h]["align_rot"],
            "align_real":      alignments[h]["align_real"],
        })

    # 4. P6-I1 test
    p6i1 = compare_induction_vs_semantic(head_records, induction_idx, semantic_idx)

    # 5. Aggregate
    n_induction = len(induction_idx)
    mean_align_rot_all   = float(np.mean([a["align_rot"]  for a in alignments]))
    mean_align_real_all  = float(np.mean([a["align_real"] for a in alignments]))
    mean_align_rot_ind   = p6i1.get("mean_align_rot_induction")
    mean_align_rot_sem   = p6i1.get("mean_align_rot_semantic")

    payload = {
        "layer_name":          layer_name,
        "n_heads":             n_heads,
        "n_induction_heads":   n_induction,
        "induction_indices":   induction_idx,
        "semantic_indices":    semantic_idx,
        "mean_align_rot_all":  mean_align_rot_all,
        "mean_align_real_all": mean_align_real_all,
        "p6_i1":               p6i1,
        "head_records":        head_records,
    }

    # --- Summary lines ---
    lines = [
        SEP_THICK,
        "INDUCTION DETECTION + OV WRITE ALIGNMENT  [Track A]",
        SEP_THICK,
        f"Layer:                {layer_name}",
        f"Heads analysed:       {n_heads}",
        f"Induction threshold:  {ind_threshold}",
        "",
        "Induction scores (mean attn on induction pairs minus background):",
    ]
    for h, s in enumerate(scores):
        flag = " ← INDUCTION" if h in induction_idx else ""
        lines.append(f"  head {h:02d}:  score={_fmt(s)}{flag}")

    lines += [
        "",
        f"Induction heads detected: {n_induction} of {n_heads}",
        "",
        "OV write direction alignment with S/A channels (top-16 singular vectors):",
        _bullet("mean align_rot (all heads)", mean_align_rot_all),
        _bullet("mean align_real (all heads)", mean_align_real_all),
        _bullet("mean align_rot (induction heads)", mean_align_rot_ind),
        _bullet("mean align_rot (semantic heads)", mean_align_rot_sem),
        "",
        "P6-I1: induction heads write into imaginary channel more than semantic heads?",
        _bullet("delta align_rot (induction - semantic)", p6i1.get("delta_align_rot")),
        _bullet("MWU p-value", p6i1.get("mwu_pvalue")),
        _verdict_line(
            "P6-I1",
            p6i1["p6_i1_satisfied"],
            f"mu_ind={_fmt(mean_align_rot_ind)} vs mu_sem={_fmt(mean_align_rot_sem)}"
            f" p={_fmt(p6i1.get('mwu_pvalue'))}",
        ),
        "",
        "Note on ALBERT: shared weights mean the same heads implement both channels.",
        "If P6-I1 passes, channel separation arises from which residual-stream",
        "subspace the incoming activation occupies, not from separate weight matrices.",
    ]

    vc = {
        "ind_n_induction_heads":       n_induction,
        "ind_mean_align_rot_all":      mean_align_rot_all,
        "ind_mean_align_rot_induction": mean_align_rot_ind,
        "ind_mean_align_rot_semantic":  mean_align_rot_sem,
        "ind_p6_i1_satisfied":         p6i1["p6_i1_satisfied"],
    }

    return SubResult(
        name="induction_ov",
        applicable=True,
        payload=payload,
        summary_lines=lines,
        verdict_contribution=vc,
    )
