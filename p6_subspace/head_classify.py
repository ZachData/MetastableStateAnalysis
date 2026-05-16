"""
head_classify.py — Track A: Head classification on the CC/PC plane.

For each attention head, computes two coupling scores:

  Content-Coupling (CC):
    Spearman correlation between the attention logit A[i,j] and the
    query-key inner product ⟨q_i, k_j⟩ across all (i,j) token pairs.
    High CC → head attends by content similarity (self-similarity / semantic).

  Positional-Coupling (PC):
    Spearman correlation between A[i,j] and a positional function f(i-j).
    Three positional modes tested:
      "prev"  : f = 1[j == i-1]  — previous-token
      "local" : f = exp(-|i-j|²/σ²)  — soft local window (σ=2)
      "all"   : f = 1/(1+|i-j|)  — generic recency bias

    PC is reported for all three; the dominant one labels the head.

The (CC, PC) plane places heads in four quadrants:
  High CC, low  PC  → semantic / self-similarity  (real-channel prediction)
  Low  CC, high PC  → positional / previous-token (imaginary-channel prediction)
  Mod  CC, mod  PC  → induction (mixed)           (imaginary-channel prediction)
  Neg  CC, low  PC  → anti-similarity              (imaginary-channel prediction)

Anti-similarity heads are detected by:
  correlation between A[i,j] and -⟨x_i, x_j⟩ significantly negative
  (at non-trivial attention weights, using a threshold on A[i,j]).

Rotational energy fraction f_rot(h) from the OV Schur decomposition
(computed by rotational_schur.extract_schur_blocks) is joined to
each head's record for cross-head correlation tests (P6-A2).

Functions
---------
compute_cc_pc            : CC/PC scores for one head given attention + QK matrices
classify_heads           : full pipeline for one model's attention matrices
anti_similarity_score    : correlation of attention with -similarity
cross_head_correlations  : permutation-based p-values for P6-A2 test
head_map_data            : structured output for plotting / reporting

Bugs fixed:
  #6  : compute_cc_pc accepts is_causal mask for lower-triangle restriction
  #7  : cross_head_correlations uses |PC| (magnitude), not signed pc_dominant
  #8  : _assign_quadrant adds cc < -0.2 anti_sim branch (strong negative CC)
  #9  : decompose_qk_matrix docstring clarifies WQ @ WK.T convention
  #12 : permutation_spearman + cross_head_correlations with p-values
"""

import numpy as np
from scipy.stats import spearmanr


# -----------
# Bug #12: Permutation test for significance
# -----------

def _permutation_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 10_000,
    alternative: str = "greater",
    seed: int = 42,
) -> tuple[float, float]:
    """
    Empirical Spearman p-value via permutation of y.
    
    Returns (rho_observed, p_value) where p-value accounts for finite permutations.
    """
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} vs {len(y)}")
    n = len(x)
    if n < 4:
        rho_obs, _ = spearmanr(x, y) if n >= 2 else (0.0, 1.0)
        return float(rho_obs) if np.isfinite(rho_obs) else 0.0, 1.0
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0, 1.0

    rho_obs, _ = spearmanr(x, y)
    if not np.isfinite(rho_obs):
        return 0.0, 1.0

    rng = np.random.default_rng(seed)
    rho_null = np.empty(n_perm, dtype=np.float64)
    y_perm = np.array(y, copy=True)
    for i in range(n_perm):
        rng.shuffle(y_perm)
        r, _ = spearmanr(x, y_perm)
        rho_null[i] = r if np.isfinite(r) else 0.0

    if alternative == "greater":
        n_extreme = int(np.sum(rho_null >= rho_obs))
    elif alternative == "less":
        n_extreme = int(np.sum(rho_null <= rho_obs))
    elif alternative == "two-sided":
        n_extreme = int(np.sum(np.abs(rho_null) >= abs(rho_obs)))
    else:
        raise ValueError(f"unknown alternative: {alternative}")

    p_value = (n_extreme + 1) / (n_perm + 1)
    return float(rho_obs), float(p_value)


# -----------
# Positional coupling modes
# -----------

_SIGMA_LOCAL = 2.0   # Gaussian half-width for "local" mode

def _positional_function(
    n_tokens: int,
    mode: str,
) -> np.ndarray:
    """
    Build (n_tokens, n_tokens) positional affinity matrix for a given mode.

    Parameters
    ----------
    n_tokens : sequence length
    mode     : "prev" | "local" | "all"

    Returns
    -------
    f : (n_tokens, n_tokens) float64
    """
    i_idx = np.arange(n_tokens)
    diff = i_idx[:, None] - i_idx[None, :]    # diff[i,j] = i - j

    if mode == "prev":
        f = (diff == 1).astype(np.float64)
    elif mode == "local":
        f = np.exp(-(diff ** 2) / (2 * _SIGMA_LOCAL ** 2))
    elif mode == "all":
        f = 1.0 / (1.0 + np.abs(diff).astype(np.float64))
    else:
        raise ValueError(f"Unknown positional mode: {mode!r}")

    return f


# -----------
# Bug #6: Causal mask in compute_cc_pc
# -----------

def compute_cc_pc(
    attn_weights: np.ndarray,
    qk_logits: np.ndarray,
    n_tokens: int,
    is_causal: bool = False,
) -> dict:
    """
    Compute CC and PC scores for one attention head.

    Parameters
    ----------
    attn_weights : (n_tokens, n_tokens) — softmax attention matrix A[i,j]
    qk_logits    : (n_tokens, n_tokens) — raw logits ⟨q_i, k_j⟩
    n_tokens     : int
    is_causal    : bool — if True, restrict to strict lower triangle

    FIX Bug #6: Restrict to strict lower triangle for causal models
    (where upper triangle has post-softmax attn ≈ 0).

    Returns
    -------
    dict with:
      cc            : float — Spearman(A[i,j], qk_logit[i,j])
      pc_prev       : float — Spearman(A[i,j], f_prev[i,j])
      pc_local      : float — Spearman(A[i,j], f_local[i,j])
      pc_all        : float — Spearman(A[i,j], f_all[i,j])
      pc_dominant   : float — max(|pc_prev|, |pc_local|, |pc_all|) with sign
      pc_mode       : str   — which positional mode is dominant
    """
    # Flatten upper triangle (i != j) for correlation
    if is_causal:
        mask = np.tril(np.ones((n_tokens, n_tokens), dtype=bool), k=-1)
    else:
        mask = ~np.eye(n_tokens, dtype=bool)
    
    a_flat = attn_weights[mask].ravel()
    q_flat = qk_logits[mask].ravel()

    cc, _ = spearmanr(a_flat, q_flat)

    pc_scores = {}
    for mode in ("prev", "local", "all"):
        f = _positional_function(n_tokens, mode)[mask].ravel()
        r, _ = spearmanr(a_flat, f)
        pc_scores[mode] = float(r) if np.isfinite(r) else 0.0

    # Dominant positional mode: highest |r|, preserving sign
    dominant_mode = max(pc_scores, key=lambda m: abs(pc_scores[m]))
    pc_dominant = pc_scores[dominant_mode]

    return {
        "cc":           float(cc) if np.isfinite(cc) else 0.0,
        "pc_prev":      pc_scores["prev"],
        "pc_local":     pc_scores["local"],
        "pc_all":       pc_scores["all"],
        "pc_dominant":  pc_dominant,
        "pc_mode":      dominant_mode,
    }


# -----------
# Anti-similarity score
# -----------

def anti_similarity_score(
    attn_weights:       np.ndarray,
    token_activations:  np.ndarray,
    attn_threshold:     float = 0.05,
) -> dict:
    """
    Measure how strongly a head attends to *dissimilar* tokens.

    Parameters
    ----------
    attn_weights      : (n_tokens, n_tokens) — softmax attention matrix
    token_activations : (n_tokens, d_model)  — L2-normed token representations
                        (cosine similarity = dot product)
    attn_threshold    : minimum attention weight to include a pair

    Returns
    -------
    dict with:
      anti_sim_rho   : float — Spearman(A[i,j], -cos_sim[i,j]) for pairs
                       above threshold.  Positive = anti-similarity.
      n_pairs_used   : int   — number of (i,j) pairs above threshold
      is_anti_sim    : bool  — anti_sim_rho > 0.20 (weak threshold for flagging)
    """
    n = attn_weights.shape[0]

    # Cosine similarity (tokens already normed)
    cos_sim = token_activations @ token_activations.T   # (n, n)

    # Pairs where attention is non-trivial (and i != j)
    mask = (attn_weights > attn_threshold) & ~np.eye(n, dtype=bool)
    if mask.sum() < 4:
        return {"anti_sim_rho": 0.0, "n_pairs_used": 0, "is_anti_sim": False}

    a_vals = attn_weights[mask].ravel()
    s_vals = cos_sim[mask].ravel()

    rho, _ = spearmanr(a_vals, -s_vals)
    rho = float(rho) if np.isfinite(rho) else 0.0

    return {
        "anti_sim_rho":  rho,
        "n_pairs_used":  int(mask.sum()),
        "is_anti_sim":   rho > 0.20,
    }


# -----------
# Bug #8: Quadrant assignment with strong negative CC
# -----------

def _assign_quadrant(cc: float, pc: float, is_anti_sim: bool) -> str:
    """
    Map (CC, PC, anti_sim) to a named quadrant.

    FIX Bug #8: Add cc < -0.2 anti_sim branch (strong negative CC).
    
    Previous code had: is_anti_sim AND cc < 0.1 → "anti_similarity"
    But missed heads with strong negative CC (e.g., cc = -0.5, pc ≈ 0).
    
    Now:
      1. is_anti_sim AND cc < 0.1          → "anti_similarity" (stricter signal)
      2. cc < -0.2 AND |pc| < 0.3          → "anti_similarity" (geometric fallback)
      3. cc > 0.3 AND |pc| < 0.3           → "semantic"
      4. |pc| > 0.3 AND |cc| < 0.3         → "positional"
      5. cc > 0.15 AND |pc| > 0.3          → "induction"
      6. otherwise                         → "mixed"

    Thresholds are intentionally wide to avoid over-labelling.
    """
    if is_anti_sim and cc < 0.1:
        return "anti_similarity"
    if cc < -0.2 and abs(pc) < 0.3:
        return "anti_similarity"
    if cc > 0.3 and abs(pc) < 0.3:
        return "semantic"
    if abs(pc) > 0.3 and abs(cc) < 0.3:
        return "positional"
    if cc > 0.15 and abs(pc) > 0.15:
        return "induction"
    return "mixed"


# -----------
# Full pipeline
# -----------

def classify_heads(
    attn_matrices:      list,
    qk_logit_matrices:  list,
    token_activations:  np.ndarray,
    rot_energy_fracs:   list | None = None,
) -> list[dict]:
    """
    Classify all heads given per-head attention and QK matrices.

    Parameters
    ----------
    attn_matrices     : list of (n_tokens, n_tokens) — one per head
    qk_logit_matrices : list of (n_tokens, n_tokens) — one per head
    token_activations : (n_tokens, d_model) — L2-normed residual stream
                        (used for anti-similarity test)
    rot_energy_fracs  : optional list of float — f_rot(h) from OV Schur
                        (will be stored as None if not provided)

    Returns
    -------
    list of dicts, one per head, each containing:
      head_idx        : int
      cc              : float — content-coupling score
      pc_prev/local/all : float
      pc_dominant     : float
      pc_mode         : str
      anti_sim_rho    : float
      is_anti_sim     : bool
      f_rot           : float or None — rotational energy fraction
      quadrant        : str — "semantic" | "positional" | "induction" |
                              "anti_similarity" | "mixed"
    """
    n_heads = len(attn_matrices)
    n_tokens = attn_matrices[0].shape[0]
    results = []

    for h in range(n_heads):
        A = attn_matrices[h]          # (n, n)
        QK = qk_logit_matrices[h]     # (n, n)

        scores = compute_cc_pc(A, QK, n_tokens)
        anti   = anti_similarity_score(A, token_activations)

        f_rot = rot_energy_fracs[h] if rot_energy_fracs is not None else None

        quadrant = _assign_quadrant(
            scores["cc"],
            scores["pc_dominant"],
            anti["is_anti_sim"],
        )

        results.append({
            "head_idx":      h,
            "cc":            scores["cc"],
            "pc_prev":       scores["pc_prev"],
            "pc_local":      scores["pc_local"],
            "pc_all":        scores["pc_all"],
            "pc_dominant":   scores["pc_dominant"],
            "pc_mode":       scores["pc_mode"],
            "anti_sim_rho":  anti["anti_sim_rho"],
            "n_pairs_used":  anti["n_pairs_used"],
            "is_anti_sim":   anti["is_anti_sim"],
            "f_rot":         f_rot,
            "quadrant":      quadrant,
        })

    return results


# -----------
# Cross-head correlation tests (P6-A2)
# Bug #7, #12: Uses |PC| (magnitude) with permutation p-values
# -----------

def cross_head_correlations(head_records: list[dict]) -> dict:
    """
    Compute cross-head Spearman correlations between f_rot and CC/PC.

    Tests falsifiable prediction P6-A2:
      ρ(f_rot, -CC) > 0.4  — rotational energy anti-correlates with CC
      ρ(f_rot,  |PC|) > 0.4 — rotational energy correlates with |PC| (magnitude)

    FIX Bug #7: Use |PC| (magnitude), not signed pc_dominant.
    FIX Bug #12: Add permutation p-values; verdict requires p < α AND ρ > threshold.

    Parameters
    ----------
    head_records : output of classify_heads (all heads, all layers)

    Returns
    -------
    dict with:
      rho_frot_neg_cc   : float — Spearman(f_rot, -CC)
      p_value_neg_cc    : float — permutation p-value
      rho_frot_abs_pc   : float — Spearman(f_rot, |PC|)
      p_value_abs_pc    : float — permutation p-value
      n_heads           : int
      alpha             : float — significance threshold
      threshold         : float — effect size threshold
      n_perm            : int — number of permutations
      n_heads_too_small_for_alpha: bool
      significance_passes : bool
      effect_size_passes : bool
      p6_a2_passes      : bool — both significance AND effect size pass
    """
    valid = [r for r in head_records if r["f_rot"] is not None]
    if len(valid) < 4:
        return {
            "rho_frot_neg_cc": 0.0,
            "p_value_neg_cc": 1.0,
            "rho_frot_abs_pc": 0.0,
            "p_value_abs_pc": 1.0,
            "n_heads": len(valid),
            "alpha": 0.05,
            "threshold": 0.4,
            "n_perm": 10_000,
            "n_heads_too_small_for_alpha": True,
            "significance_passes": False,
            "effect_size_passes": False,
            "p6_a2_passes": False,
        }

    f_rot = np.array([r["f_rot"] for r in valid])
    cc    = np.array([r["cc"]    for r in valid])
    pc_mag = np.array([abs(r["pc_dominant"]) for r in valid])

    alpha = 0.05
    threshold = 0.4
    n_perm = 10_000
    n_heads = len(valid)
    min_n_for_alpha = max(5, int(np.ceil(1.0 / alpha)))
    too_small = n_heads < min_n_for_alpha

    rho_cc, p_cc = _permutation_spearman(
        f_rot, -cc, n_perm=n_perm, alternative="greater", seed=42
    )
    rho_pc, p_pc = _permutation_spearman(
        f_rot, pc_mag, n_perm=n_perm, alternative="greater", seed=43
    )

    significance_passes = (p_cc < alpha) and (p_pc < alpha)
    effect_size_passes = (rho_cc > threshold) and (rho_pc > threshold)
    p6_a2_passes = bool(significance_passes and effect_size_passes and not too_small)

    return {
        "rho_frot_neg_cc": rho_cc,
        "p_value_neg_cc": p_cc,
        "rho_frot_abs_pc": rho_pc,
        "p_value_abs_pc": p_pc,
        "n_heads": n_heads,
        "alpha": alpha,
        "threshold": threshold,
        "n_perm": n_perm,
        "n_heads_too_small_for_alpha": too_small,
        "significance_passes": bool(significance_passes),
        "effect_size_passes": bool(effect_size_passes),
        "p6_a2_passes": p6_a2_passes,
    }


# -----------
# Structured output for reporting
# -----------

def head_map_data(
    head_records:    list[dict],
    layer_name:      str = "shared",
) -> dict:
    """
    Package head classification results for plotting / reporting.

    Returns a dict suitable for JSON serialisation (no numpy arrays).
    """
    quadrant_counts = {}
    for r in head_records:
        q = r["quadrant"]
        quadrant_counts[q] = quadrant_counts.get(q, 0) + 1

    anti_sim_heads = [r["head_idx"] for r in head_records if r["is_anti_sim"]]
    positional_heads = [
        r["head_idx"] for r in head_records
        if r["quadrant"] in ("positional", "induction")
    ]

    corr = cross_head_correlations(head_records)

    return {
        "layer_name":       layer_name,
        "n_heads":          len(head_records),
        "quadrant_counts":  quadrant_counts,
        "anti_sim_heads":   anti_sim_heads,
        "positional_heads": positional_heads,
        "cross_head_corr":  corr,
        "head_records":     [
            {k: v for k, v in r.items()}
            for r in head_records
        ],
    }