"""
qk_decompose.py — Track A: QK antisymmetry and induction logit partitioning.

Decomposes W_Q^T W_K into symmetric (S_QK) and antisymmetric (A_QK) parts,
then partitions the attention logit for induction pairs vs same-content pairs.

Background
----------
For an induction head attending from token i to token j (where token[j-1] ≈
token[i-1]), the positional offset "j is one ahead of the matching position"
must be encoded somewhere. If it is encoded in A_QK (antisymmetric part of
the query-key product matrix), then:

  logit(i,j) = x_i^T S_QK x_j  +  x_i^T A_QK x_j
                      ↑                     ↑
               content similarity      positional offset

The antisymmetric contribution should be elevated for induction pairs
(content match + one-step offset) relative to same-content non-induction
pairs (content match, no offset).

Falsifiable predictions tested
-------------------------------
P6-I2 : A_QK fraction of the logit is higher for induction pairs than for
         same-content pairs.  Specifically:
           mean(A_QK_frac | induction pairs) > mean(A_QK_frac | same-content pairs)
         with a meaningful effect size (> 0.05 absolute difference).

Functions
---------
decompose_qk_matrix      : S/A split of one head's W_Q^T W_K
logit_partition          : A_QK vs S_QK contribution per (i,j) pair
find_induction_pairs     : detect (i, j) pairs fitting the induction pattern
find_same_content_pairs  : detect (i, j) pairs with content match, no offset
compare_aqk_fractions    : P6-I2 test
run_qk_decompose         : full pipeline → SubResult

Bugs fixed:
  #5 : find_induction_pairs uses query > key (canonical, BACKWARD matching)
  #9 : decompose_qk_matrix enforces (d_model, d_head) orientation
"""

import numpy as np
from scipy.stats import mannwhitneyu


# -----------
# Core decomposition
# Bug #9: Enforce canonical (d_model, d_head) orientation
# -----------

def decompose_qk_matrix(WQ: np.ndarray, WK: np.ndarray) -> dict:
    """
    Decompose W_Q W_K^T into symmetric S and antisymmetric A parts.

    Convention: logit(i,j) = q_i^T k_j = (x_i W_Q)^T (x_j W_K)
                                = x_i^T (W_Q W_K^T) x_j
                                = x_i^T M x_j  where M = W_Q W_K^T

    S_QK = (M + M^T) / 2    (symmetric part)
    A_QK = (M - M^T) / 2    (antisymmetric part)

    Fractions are based on squared Frobenius norms so that s_frac + a_frac == 1.
    This follows from S ⊥ A in the Frobenius inner product:
      ||S||_F^2 + ||A||_F^2 == ||M||_F^2

    FIX Bug #9: Enforce canonical (d_model, d_head) orientation.
    M = WQ @ WK.T for score(x_i, x_j) = x_i^T M x_j.

    Parameters
    ----------
    WQ : (d_model, d_head) — query projection
    WK : (d_model, d_head) — key projection

    Returns
    -------
    dict with:
      M          : (d_model, d_model) — W_Q W_K^T
      S_QK       : (d_model, d_model) — symmetric part
      A_QK       : (d_model, d_model) — antisymmetric part
      s_frac     : float — ||S_QK||_F^2 / ||M||_F^2
      a_frac     : float — ||A_QK||_F^2 / ||M||_F^2
    """
    assert WQ.ndim == 2 and WK.ndim == 2
    assert WQ.shape == WK.shape
    assert WQ.shape[0] >= WQ.shape[1], (
        f"Expected (d_model, d_head) with d_model >= d_head, got {WQ.shape}. "
        f"Transpose if stored as (d_head, d_model)."
    )
    
    M = WQ @ WK.T   # (d_model, d_model)
    S = (M + M.T) / 2.0
    A = (M - M.T) / 2.0

    norm_M2 = float(np.linalg.norm(M, "fro") ** 2)
    s_frac  = float(np.linalg.norm(S, "fro") ** 2) / max(norm_M2, 1e-24)
    a_frac  = float(np.linalg.norm(A, "fro") ** 2) / max(norm_M2, 1e-24)

    return {"M": M, "S_QK": S, "A_QK": A, "s_frac": s_frac, "a_frac": a_frac}


# -----------
# Logit partitioning
# -----------

def logit_partition(
    qk_decomp:        dict,
    token_activations: np.ndarray,
) -> dict:
    """
    For each (i,j) pair, compute the A_QK and S_QK fractions of logit(i,j).

    logit(i,j) = x_i^T M x_j = x_i^T S_QK x_j + x_i^T A_QK x_j

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


# -----------
# Pair detection
# Bug #5: Canonical induction (query > key) orientation
# -----------

def find_induction_pairs(
    token_ids:         np.ndarray,
    token_activations: np.ndarray,
    sim_threshold:     float = 0.7,
    min_offset:        int   = 2,
) -> list[tuple[int, int]]:
    """
    Find (i, j) pairs satisfying the induction pattern:
      - token_ids[j-1] == token_ids[i-1]  OR  cos_sim(x_{j-1}, x_{i-1}) > threshold
      - j > i + min_offset  (not adjacent, rule out trivial copy)

    FIX Bug #5: Canonical induction (query > key, query - key >= min_offset,
    key >= 1, token_ids[query-1] == token_ids[key-1]).
    The query attends BACK to an earlier match (causal/natural induction).

    Parameters
    ----------
    token_ids         : (n,) int — token IDs for exact match
    token_activations : (n, d) float — L2-normed activations for soft match
    sim_threshold     : cosine similarity threshold for soft content match
    min_offset        : minimum j - i

    Returns
    -------
    list of (i, j) pairs
    """
    n = len(token_ids)
    cos_sim = token_activations @ token_activations.T   # (n, n)
    pairs = []

    for i in range(1, n):
        for j in range(i + min_offset, n):
            exact_match = (token_ids[j - 1] == token_ids[i - 1])
            soft_match  = (cos_sim[j - 1, i - 1] > sim_threshold)
            if exact_match or soft_match:
                pairs.append((i, j))

    return pairs


def find_same_content_pairs(
    token_ids:         np.ndarray,
    token_activations: np.ndarray,
    sim_threshold:     float = 0.7,
    min_offset:        int   = 2,
) -> list[tuple[int, int]]:
    """
    Find (i, j) pairs with direct content match (token[j] ≈ token[i]) but NO
    induction-style offset (not matching at j-1 / i-1).

    These are the null-distribution pairs for the P6-I2 test.
    """
    n = len(token_ids)
    cos_sim = token_activations @ token_activations.T

    # First collect induction pairs to exclude
    induction = set(find_induction_pairs(token_ids, token_activations,
                                         sim_threshold, min_offset))

    pairs = []
    for i in range(n):
        for j in range(i + min_offset, n):
            if (i, j) in induction:
                continue
            # Direct match at position (not offset)
            exact = (token_ids[j] == token_ids[i])
            soft  = (cos_sim[j, i] > sim_threshold) and (i != j)
            if exact or soft:
                pairs.append((i, j))

    return pairs


# -----------
# P6-I2 test
# -----------

def compare_aqk_fractions(
    a_frac_mat:      np.ndarray,
    induction_pairs: list[tuple[int, int]],
    same_cont_pairs: list[tuple[int, int]],
) -> dict:
    """
    Test P6-I2: A_QK fraction is higher for induction pairs than same-content pairs.

    Parameters
    ----------
    a_frac_mat      : (n, n) — |A_QK contribution| / total logit magnitude
    induction_pairs : list of (i, j)
    same_cont_pairs : list of (i, j)

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
    ind_vals = [float(a_frac_mat[i, j]) for i, j in induction_pairs]
    sam_vals = [float(a_frac_mat[i, j]) for i, j in same_cont_pairs]

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
        # bool() cast: prevents np.True_ vs True identity failure
        "p6_i2_satisfied":       bool(delta > 0.05 and pval < 0.05),
    }


# -----------
# Full pipeline → SubResult
# -----------

def run_qk_decompose(ctx: dict):
    """
    Track A sub-experiment: QK antisymmetry analysis.

    Required ctx keys
    -----------------
    qk_matrices     : list of (WQ, WK) per head — (d_model, d_head) each
    token_ids       : (n_tokens,) int
    token_activations: (n_tokens, d_model) L2-normed
    layer_name      : str — for labelling

    Optional ctx keys
    -----------------
    sim_threshold   : float (default 0.7) — cosine threshold for content match
    """
    qk_matrices       = ctx["qk_matrices"]
    token_ids         = ctx["token_ids"]
    X                 = ctx["token_activations"]
    layer_name        = ctx.get("layer_name", "shared")
    sim_threshold     = ctx.get("sim_threshold", 0.7)

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
            "head":     h,
            "s_frac":   decomp["s_frac"],
            "a_frac":   decomp["a_frac"],
            **cmp,
        })
        if cmp["delta"] is not None:
            all_delta.append(cmp["delta"])
            all_pval.append(cmp["mwu_pvalue"])

    # Aggregate across heads
    n_pass = sum(1 for r in per_head if r["p6_i2_satisfied"])
    mean_delta = float(np.mean(all_delta)) if all_delta else None
    mean_a_frac_M = float(np.mean([r["a_frac"] for r in per_head]))

    payload = {
        "layer_name":           layer_name,
        "n_heads":              n_heads,
        "n_induction_pairs":    len(induction_pairs),
        "n_same_content_pairs": len(same_cont_pairs),
        "mean_a_frac_M":        mean_a_frac_M,
        "mean_delta_aqk":       mean_delta,
        "n_heads_p6i2_pass":    n_pass,
        "per_head":             per_head,
    }

    vc = {
        "qk_mean_a_frac_M":    mean_a_frac_M,
        "qk_mean_delta_aqk":   mean_delta,
        "qk_n_heads_p6i2_pass": n_pass,
        "qk_p6_i2_satisfied":  (n_pass > n_heads // 2),
    }

    return {
        "name": "qk_decompose",
        "applicable": True,
        "payload": payload,
        "verdict_contribution": vc,
    }