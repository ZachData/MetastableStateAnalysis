"""
qk_decompose.py — Track A: QK antisymmetry and induction logit partitioning.

Decomposes W_Q W_K^T into symmetric (S_QK) and antisymmetric (A_QK) parts,
then partitions the attention logit for induction pairs vs same-content pairs.

Background
----------
For an induction head attending from token i to token j (where token[j-1] ≈
token[i-1]), the positional offset "j is one ahead of the matching position"
must be encoded somewhere. If it is encoded in A_QK (antisymmetric part of
the query-key product matrix), then:

  logit(query, key) = x_query^T S_QK x_key  +  x_query^T A_QK x_key
                             ↑                          ↑
                      content similarity           positional offset

The antisymmetric contribution should be elevated for induction pairs
(content match + one-step offset) relative to same-content non-induction
pairs (content match, no offset).

Falsifiable predictions tested
-------------------------------
P6-I2 : A_QK fraction of the logit is higher for induction pairs than for
         same-content pairs.  Specifically:
           mean(A_QK_frac | induction pairs) > mean(A_QK_frac | same-content pairs)
         with a meaningful effect size (> 0.05 absolute difference).

Canonical pair convention
--------------------------
All (query, key) tuples are returned with query > key.  The query is the LATER
position attending BACK to an earlier match — consistent with the Anthropic
induction-head definition and the only direction with non-trivial post-softmax
attention in causal models.  Callers that index a_frac_mat[query, key] will
match attn_weights[query, key] (post-softmax weight from query onto key).
Note: |a_contrib[query, key]| == |a_contrib[key, query]| because A is
antisymmetric, so the orientation is a no-op for magnitude tests but matters
for cross-script joins with induction_ov.

Functions
---------
decompose_qk_matrix      : S/A split of one head's W_Q W_K^T
logit_partition          : A_QK vs S_QK contribution per (query, key) pair
find_induction_pairs     : detect (query, key) pairs fitting the induction pattern
find_same_content_pairs  : detect (query, key) pairs with content match, no offset
compare_aqk_fractions    : P6-I2 test
run_qk_decompose         : full pipeline → SubResult
"""

import numpy as np
from scipy.stats import mannwhitneyu

from p6_subspace.p6_io import SubResult, _fmt, _bullet, _verdict_line, SEP_THICK, SEP_THIN


# ---------------------------------------------------------------------------
# Core decomposition
# ---------------------------------------------------------------------------

def decompose_qk_matrix(WQ: np.ndarray, WK: np.ndarray) -> dict:
    """
    Decompose W_Q W_K^T into symmetric S and antisymmetric A parts.

    Convention: WQ and WK are per-head matrices in canonical orientation
    (d_model, d_head).  Then M = WQ @ WK.T has shape (d_model, d_model) and
    score(x_query, x_key) = x_query^T M x_key (with x as column vectors).

    logit(query,key) = q^T k = (x_query W_Q)^T (x_key W_K)
                             = x_query^T (W_Q W_K^T) x_key
                             = x_query^T M x_key  where M = W_Q W_K^T

    S_QK = (M + M^T) / 2    (symmetric part)
    A_QK = (M - M^T) / 2    (antisymmetric part)

    Fractions are based on squared Frobenius norms so that s_frac + a_frac == 1.
    This follows from S ⊥ A in the Frobenius inner product:
      ||S||_F^2 + ||A||_F^2 == ||M||_F^2

    Parameters
    ----------
    WQ : (d_model, d_head) — query projection, canonical orientation
    WK : (d_model, d_head) — key projection, canonical orientation

    If your raw weights are stored as (d_head, d_model), transpose before
    calling this function.

    Returns
    -------
    dict with:
      M      : (d_model, d_model) — W_Q W_K^T
      S_QK   : (d_model, d_model) — symmetric part
      A_QK   : (d_model, d_model) — antisymmetric part
      s_frac : float — ||S_QK||_F^2 / ||M||_F^2
      a_frac : float — ||A_QK||_F^2 / ||M||_F^2
    """
    # Bug 2 / Bug 9: enforce canonical (d_model, d_head) orientation.
    assert WQ.ndim == 2 and WK.ndim == 2, (
        f"decompose_qk_matrix: WQ/WK must be 2D, got {WQ.shape}, {WK.shape}"
    )
    assert WQ.shape == WK.shape, (
        f"decompose_qk_matrix: WQ and WK must have the same shape; "
        f"got {WQ.shape} and {WK.shape}"
    )
    assert WQ.shape[0] >= WQ.shape[1], (
        f"decompose_qk_matrix: expected canonical (d_model, d_head) with "
        f"d_model >= d_head, got {WQ.shape}.  If your raw weights are stored "
        f"as (d_head, d_model), transpose before calling this function."
    )

    M = WQ @ WK.T   # (d_model, d_model)
    S = (M + M.T) / 2.0
    A = (M - M.T) / 2.0

    norm_M2 = float(np.linalg.norm(M, "fro") ** 2)
    s_frac  = float(np.linalg.norm(S, "fro") ** 2) / max(norm_M2, 1e-24)
    a_frac  = float(np.linalg.norm(A, "fro") ** 2) / max(norm_M2, 1e-24)

    return {"M": M, "S_QK": S, "A_QK": A, "s_frac": s_frac, "a_frac": a_frac}


# ---------------------------------------------------------------------------
# Logit partitioning
# ---------------------------------------------------------------------------

def logit_partition(
    qk_decomp:         dict,
    token_activations: np.ndarray,
) -> dict:
    """
    For each (query, key) pair, compute the A_QK and S_QK fractions of
    logit(query, key).

    logit(query,key) = x_query^T M x_key
                     = x_query^T S_QK x_key + x_query^T A_QK x_key

    Returns
    -------
    dict with:
      s_contrib : (n, n) — symmetric contribution per pair
      a_contrib : (n, n) — antisymmetric contribution per pair
      a_frac_mat: (n, n) — |a_contrib| / (|s_contrib| + |a_contrib|)
    """
    X = token_activations   # (n, d_model)
    S = qk_decomp["S_QK"]
    A = qk_decomp["A_QK"]

    s_contrib = X @ S @ X.T   # (n, n)
    a_contrib = X @ A @ X.T   # (n, n)

    denom = np.abs(s_contrib) + np.abs(a_contrib)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    a_frac_mat = np.abs(a_contrib) / denom

    return {
        "s_contrib":  s_contrib,
        "a_contrib":  a_contrib,
        "a_frac_mat": a_frac_mat,
    }


# ---------------------------------------------------------------------------
# Pair detection
# ---------------------------------------------------------------------------

def find_induction_pairs(
    token_ids:         np.ndarray,
    token_activations: np.ndarray,
    sim_threshold:     float = 0.7,
    min_offset:        int   = 2,
) -> list[tuple[int, int]]:
    """
    Find (query, key) pairs satisfying the induction pattern.

    Canonical direction: query > key (LATER position attends BACK to earlier
    match).  Tuples are returned as (query, key) with query > key.

    Conditions:
      - token_ids[query-1] == token_ids[key-1]  OR
        cos_sim(x_{query-1}, x_{key-1}) > sim_threshold
      - query - key >= min_offset  (not adjacent, rule out trivial copy)
      - key >= 1  (predecessor of key must exist)

    Bug 5: previously returned (key, query) with key < query (forward
    direction); now returns (query, key) with query > key to match the
    Anthropic canonical definition and align with induction_ov.

    Parameters
    ----------
    token_ids         : (n,) int — token IDs for exact match
    token_activations : (n, d) float — L2-normed activations for soft match
    sim_threshold     : cosine similarity threshold for soft content match
    min_offset        : minimum query - key

    Returns
    -------
    list of (query, key) tuples with query > key
    """
    n = len(token_ids)
    cos_sim = token_activations @ token_activations.T   # (n, n)
    pairs = []

    for key in range(1, n):
        for query in range(key + min_offset, n):
            exact_match = (token_ids[query - 1] == token_ids[key - 1])
            soft_match  = (cos_sim[query - 1, key - 1] > sim_threshold)
            if exact_match or soft_match:
                pairs.append((query, key))

    return pairs


def find_same_content_pairs(
    token_ids:         np.ndarray,
    token_activations: np.ndarray,
    sim_threshold:     float = 0.7,
    min_offset:        int   = 2,
) -> list[tuple[int, int]]:
    """
    Find (query, key) pairs with direct content match (token[query] ≈
    token[key]) but NO induction-style offset (not matching at query-1 /
    key-1), excluding any pair already classified as induction.

    Canonical direction: query > key.  Tuples returned as (query, key).

    These are the null-distribution pairs for the P6-I2 test.  Direct
    same-content pairs have x_query ≈ x_key in content but differ in
    position; the antisymmetric form x^T A x evaluates to ≈ 0 in the limit
    x_query = x_key, so the antisymmetric fraction here is the floor against
    which induction pairs should stand out.

    Parameters
    ----------
    token_ids         : (n,) int
    token_activations : (n, d) float — L2-normed
    sim_threshold     : cosine similarity threshold
    min_offset        : minimum query - key

    Returns
    -------
    list of (query, key) tuples with query > key
    """
    n = len(token_ids)
    cos_sim = token_activations @ token_activations.T

    induction = set(find_induction_pairs(token_ids, token_activations,
                                         sim_threshold, min_offset))

    pairs = []
    for key in range(n):
        for query in range(key + min_offset, n):
            if (query, key) in induction:
                continue
            exact = (token_ids[query] == token_ids[key])
            soft  = (cos_sim[query, key] > sim_threshold)
            if exact or soft:
                pairs.append((query, key))

    return pairs


# ---------------------------------------------------------------------------
# P6-I2 test
# ---------------------------------------------------------------------------

def compare_aqk_fractions(
    a_frac_mat:      np.ndarray,
    induction_pairs: list[tuple[int, int]],
    same_cont_pairs: list[tuple[int, int]],
) -> dict:
    """
    Test P6-I2: A_QK fraction is higher for induction pairs than same-content
    pairs.

    Parameters
    ----------
    a_frac_mat      : (n, n) — |A_QK contribution| / total logit magnitude
    induction_pairs : list of (query, key)
    same_cont_pairs : list of (query, key)

    Returns
    -------
    dict with:
      mean_aqk_induction   : float
      mean_aqk_same_content: float
      delta                : float — induction minus same_content
      mwu_statistic        : float — Mann-Whitney U
      mwu_pvalue           : float
      n_induction          : int
      n_same_content       : int
      p6_i2_satisfied      : bool — delta > 0.05 and p < 0.05
    """
    ind_vals = [float(a_frac_mat[q, k]) for q, k in induction_pairs]
    sam_vals = [float(a_frac_mat[q, k]) for q, k in same_cont_pairs]

    if len(ind_vals) < 3 or len(sam_vals) < 3:
        return {
            "mean_aqk_induction":    None,
            "mean_aqk_same_content": None,
            "delta":                 None,
            "mwu_statistic":         None,
            "mwu_pvalue":            None,
            "n_induction":           len(ind_vals),
            "n_same_content":        len(sam_vals),
            "p6_i2_satisfied":       False,
        }

    mu_ind = float(np.mean(ind_vals))
    mu_sam = float(np.mean(sam_vals))
    delta  = mu_ind - mu_sam

    stat, pval = mannwhitneyu(ind_vals, sam_vals, alternative="greater")

    return {
        "mean_aqk_induction":    mu_ind,
        "mean_aqk_same_content": mu_sam,
        "delta":                 delta,
        "mwu_statistic":         float(stat),
        "mwu_pvalue":            float(pval),
        "n_induction":           len(ind_vals),
        "n_same_content":        len(sam_vals),
        # bool() cast: prevents np.True_ vs True identity failure in `is True` assertions
        "p6_i2_satisfied":       bool(delta > 0.05 and pval < 0.05),
    }


# ---------------------------------------------------------------------------
# Full pipeline → SubResult
# ---------------------------------------------------------------------------

def run_qk_decompose(ctx: dict) -> SubResult:
    """
    Track A sub-experiment: QK antisymmetry analysis.

    Required ctx keys
    -----------------
    qk_matrices      : list of (WQ, WK) per head — (d_model, d_head) each
    token_ids        : (n_tokens,) int
    token_activations: (n_tokens, d_model) L2-normed
    layer_name       : str — for labelling

    Optional ctx keys
    -----------------
    sim_threshold    : float (default 0.7) — cosine threshold for content match
    """
    qk_matrices   = ctx["qk_matrices"]
    token_ids     = ctx["token_ids"]
    X             = ctx["token_activations"]
    layer_name    = ctx.get("layer_name", "shared")
    sim_threshold = ctx.get("sim_threshold", 0.7)

    n_heads = len(qk_matrices)

    induction_pairs = find_induction_pairs(token_ids, X, sim_threshold)
    same_cont_pairs = find_same_content_pairs(token_ids, X, sim_threshold)

    per_head = []
    all_delta, all_pval = [], []

    for h, (WQ, WK) in enumerate(qk_matrices):
        decomp    = decompose_qk_matrix(WQ, WK)
        partition = logit_partition(decomp, X)
        cmp       = compare_aqk_fractions(
            partition["a_frac_mat"], induction_pairs, same_cont_pairs
        )
        per_head.append({
            "head":   h,
            "s_frac": decomp["s_frac"],
            "a_frac": decomp["a_frac"],
            **cmp,
        })
        if cmp["delta"] is not None:
            all_delta.append(cmp["delta"])
            all_pval.append(cmp["mwu_pvalue"])

    # Aggregate across heads
    n_pass        = sum(1 for r in per_head if r["p6_i2_satisfied"])
    mean_delta    = float(np.mean(all_delta)) if all_delta else None
    mean_a_frac_M = float(np.mean([r["a_frac"] for r in per_head]))

    payload = {
        "layer_name":             layer_name,
        "n_heads":                n_heads,
        "n_induction_pairs":      len(induction_pairs),
        "n_same_content_pairs":   len(same_cont_pairs),
        "mean_a_frac_M":          mean_a_frac_M,
        "mean_delta_aqk":         mean_delta,
        "n_heads_p6i2_pass":      n_pass,
        "per_head":               per_head,
    }

    # --- Summary lines ---
    lines = [
        SEP_THICK,
        "QK ANTISYMMETRY ANALYSIS  [Track A]",
        SEP_THICK,
        f"Layer:              {layer_name}",
        f"Heads analysed:     {n_heads}",
        f"Induction pairs:    {len(induction_pairs)}",
        f"Same-content pairs: {len(same_cont_pairs)}",
        "",
        "Mean A_QK energy fraction across all heads (||A_QK||_F / ||M||_F):",
        _bullet("mean_a_frac_M", mean_a_frac_M),
        "",
        "P6-I2: A_QK fraction elevated for induction vs same-content pairs?",
        _bullet("mean delta (induction - same-content)", mean_delta),
        _bullet("heads satisfying P6-I2 (delta>0.05, p<0.05)", n_pass),
        _verdict_line(
            "P6-I2",
            (n_pass > n_heads // 2) if n_heads > 0 else None,
            f"{n_pass}/{n_heads} heads pass, mean delta={_fmt(mean_delta)}",
        ),
        "",
        "Per-head A_QK fraction of M (symmetric vs antisymmetric energy):",
    ]
    for r in per_head:
        lines.append(
            f"  head {r['head']:02d}:  s_frac={_fmt(r['s_frac'])}  "
            f"a_frac={_fmt(r['a_frac'])}  "
            f"delta_aqk={_fmt(r['delta'])}  "
            f"p={_fmt(r['mwu_pvalue'])}  "
            f"P6-I2={'pass' if r['p6_i2_satisfied'] else 'fail'}"
        )

    vc = {
        "qk_mean_a_frac_M":     mean_a_frac_M,
        "qk_mean_delta_aqk":    mean_delta,
        "qk_n_heads_p6i2_pass": n_pass,
        "qk_p6_i2_satisfied":   (n_pass > n_heads // 2),
    }

    return SubResult(
        name="qk_decompose",
        applicable=True,
        payload=payload,
        summary_lines=lines,
        verdict_contribution=vc,
    )