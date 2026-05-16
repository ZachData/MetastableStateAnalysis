"""
head_classify.py — Track A: Head classification on the CC/PC plane.

Cumulative fixes applied:
  Bug 6  : compute_cc_pc  — causal mask (strict lower triangle for GPT-2)
  Bug 8  : _assign_quadrant — catch strong-negative-CC heads
  Bug 12 : _permutation_spearman + cross_head_correlations — permutation
           p-values replace hard threshold; incorporates Bug 7 |PC| fix.

For each attention head, computes two coupling scores:

  Content-Coupling (CC):
    Spearman correlation between the attention logit A[i,j] and the
    query-key inner product <q_i, k_j> across all (i,j) token pairs.
    High CC → head attends by content similarity (self-similarity / semantic).

  Positional-Coupling (PC):
    Spearman correlation between A[i,j] and a positional function f(i-j).
    Three positional modes tested:
      "prev"  : f = 1[j == i-1]  — previous-token
      "local" : f = exp(-|i-j|^2/sigma^2)  — soft local window (sigma=2)
      "all"   : f = 1/(1+|i-j|)  — generic recency bias

    PC is reported for all three; the dominant one labels the head.

The (CC, PC) plane places heads in four quadrants:
  High CC, low  PC  → semantic / self-similarity  (real-channel prediction)
  Low  CC, high PC  → positional / previous-token (imaginary-channel prediction)
  Mod  CC, mod  PC  → induction (mixed)           (imaginary-channel prediction)
  Neg  CC, low  PC  → anti-similarity              (imaginary-channel prediction)

Anti-similarity heads are detected by:
  correlation between A[i,j] and -<x_i, x_j> significantly negative
  (at non-trivial attention weights, using a threshold on A[i,j]).

Rotational energy fraction f_rot(h) from the OV Schur decomposition
(computed by rotational_schur.extract_schur_blocks) is joined to
each head's record for cross-head correlation tests (P6-A2).

Functions
---------
compute_cc_pc            : CC/PC scores for one head given attention + QK matrices
classify_heads           : full pipeline for one model's attention matrices
anti_similarity_score    : correlation of attention with -similarity
head_map_data            : structured output for plotting / reporting
"""

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Positional coupling modes
# ---------------------------------------------------------------------------

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
    diff  = i_idx[:, None] - i_idx[None, :]    # diff[i,j] = i - j

    if mode == "prev":
        f = (diff == 1).astype(np.float64)
    elif mode == "local":
        f = np.exp(-(diff ** 2) / (2 * _SIGMA_LOCAL ** 2))
    elif mode == "all":
        f = 1.0 / (1.0 + np.abs(diff).astype(np.float64))
    else:
        raise ValueError(f"Unknown positional mode: {mode!r}")

    return f


# ---------------------------------------------------------------------------
# Per-head CC/PC  (Bug 6 fix: causal mask)
# ---------------------------------------------------------------------------

def compute_cc_pc(
    attn_weights:      np.ndarray,
    qk_logits:         np.ndarray,
    token_activations: np.ndarray | None = None,
    pos_modes:         tuple[str, ...] = ("prev", "local", "all"),
    is_causal:         bool = False,
) -> dict:
    """
    Content Correlation (CC) and Positional Correlation (PC) for one head.

    Parameters
    ----------
    attn_weights      : (n, n) post-softmax attention weights.
    qk_logits         : (n, n) raw QK scores (pre-mask, pre-softmax).
    token_activations : optional; reserved for anti-similarity scoring.
    pos_modes         : positional functions to correlate against; each gets
                        its own entry in the returned pc_scores dict.
    is_causal         : if True, restrict correlations to the strict lower
                        triangle (query > key).  Default False preserves prior
                        behaviour for bidirectional models like ALBERT.

    Returns
    -------
    dict with:
      cc        : float — Spearman(A[i,j], qk_logit[i,j])
      pc_scores : dict[str, float] — Spearman(A[i,j], f_mode[i,j]) per mode

    Bug 6 note
    ----------
    Previously the mask was ~np.eye(n) for all model types.  That mixed the
    upper-triangle population (≈ 0 in causal models post-softmax) into the
    correlation, depressing CC and PC magnitudes.  For causal models the mask
    is now the strict lower triangle (k=-1).
    """
    n = attn_weights.shape[0]

    # Bug 6: use lower triangle for causal (GPT-2 style) models.
    if is_causal:
        mask = np.tril(np.ones((n, n), dtype=bool), k=-1)   # query > key
    else:
        mask = ~np.eye(n, dtype=bool)

    a_flat  = attn_weights[mask]
    qk_flat = qk_logits[mask]

    # Guard against degenerate (constant) arrays.
    if a_flat.std() < 1e-12 or qk_flat.std() < 1e-12:
        cc = 0.0
    else:
        rho, _ = spearmanr(a_flat, qk_flat)
        cc = float(rho) if np.isfinite(rho) else 0.0

    pc_scores: dict[str, float] = {}
    for mode in pos_modes:
        f      = _positional_function(n, mode)
        f_flat = f[mask]
        if a_flat.std() < 1e-12 or f_flat.std() < 1e-12:
            pc_scores[mode] = 0.0
        else:
            rho, _ = spearmanr(a_flat, f_flat)
            pc_scores[mode] = float(rho) if np.isfinite(rho) else 0.0

    return {"cc": cc, "pc_scores": pc_scores}


# ---------------------------------------------------------------------------
# Anti-similarity score
# ---------------------------------------------------------------------------

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

    cos_sim = token_activations @ token_activations.T   # (n, n)

    mask = (attn_weights > attn_threshold) & ~np.eye(n, dtype=bool)
    if mask.sum() < 4:
        return {"anti_sim_rho": 0.0, "n_pairs_used": 0, "is_anti_sim": False}

    a_vals = attn_weights[mask].ravel()
    s_vals = cos_sim[mask].ravel()

    rho, _ = spearmanr(a_vals, -s_vals)
    rho    = float(rho) if np.isfinite(rho) else 0.0

    return {
        "anti_sim_rho":  rho,
        "n_pairs_used":  int(mask.sum()),
        "is_anti_sim":   rho > 0.20,
    }


# ---------------------------------------------------------------------------
# Quadrant assignment  (Bug 8 fix: catch strong-negative-CC heads)
# ---------------------------------------------------------------------------

def _assign_quadrant(
    cc:               float,
    pc:               float,
    is_anti_sim:      bool,
    cc_pos_threshold: float = 0.3,
    cc_neg_threshold: float = -0.2,
    pc_threshold:     float = 0.3,
    induction_thresh: float = 0.15,
) -> str:
    """
    Quadrant label from CC, |PC|, and the anti-similarity flag.

    Returns one of: "anti_similarity", "semantic", "positional", "induction",
    "mixed".

    Decision tree (first match wins):
        1.  is_anti_sim AND cc < 0.1                      -> "anti_similarity"
            (activation-based dissimilarity test fired and CC isn't strongly
             positive — the stricter, more interpretable signal)
        2.  cc < cc_neg_threshold AND |pc| < pc_threshold  -> "anti_similarity"
            (Bug 8: geometric fallback for heads whose anti-sim signature shows
             up only as strongly anti-correlated CC; previously fell to "mixed")
        3.  cc > cc_pos_threshold AND |pc| < pc_threshold  -> "semantic"
        4.  |pc| > pc_threshold AND |cc| < cc_pos_threshold -> "positional"
        5.  cc > induction_thresh AND |pc| > induction_thresh -> "induction"
        6.  otherwise                                        -> "mixed"
    """
    if is_anti_sim and cc < 0.1:
        return "anti_similarity"
    if cc < cc_neg_threshold and abs(pc) < pc_threshold:          # Bug 8
        return "anti_similarity"
    if cc > cc_pos_threshold and abs(pc) < pc_threshold:
        return "semantic"
    if abs(pc) > pc_threshold and abs(cc) < cc_pos_threshold:
        return "positional"
    if cc > induction_thresh and abs(pc) > induction_thresh:
        return "induction"
    return "mixed"


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def classify_heads(
    attn_matrices:      list,
    qk_logit_matrices:  list,
    token_activations:  np.ndarray,
    rot_energy_fracs:   list | None = None,
    is_causal:          bool = False,
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
    is_causal         : passed through to compute_cc_pc; set True for GPT-2
                        and other autoregressive (decoder-only) models.

    Returns
    -------
    list of dicts, one per head, each containing:
      head_idx        : int
      cc              : float — content-coupling score
      pc_scores       : dict[str, float] — per-mode positional correlations
      pc_dominant     : float — highest |PC| across modes, with original sign
      pc_mode         : str   — which positional mode is dominant
      anti_sim_rho    : float
      is_anti_sim     : bool
      f_rot           : float or None — rotational energy fraction
      quadrant        : str — "semantic" | "positional" | "induction" |
                              "anti_similarity" | "mixed"
    """
    n_heads = len(attn_matrices)
    results = []

    for h in range(n_heads):
        A  = attn_matrices[h]
        QK = qk_logit_matrices[h]

        scores = compute_cc_pc(
            A, QK,
            token_activations=token_activations,
            is_causal=is_causal,
        )
        anti  = anti_similarity_score(A, token_activations)
        f_rot = rot_energy_fracs[h] if rot_energy_fracs is not None else None

        # Derive dominant PC from the pc_scores dict (highest |r|, keep sign).
        pc_scores     = scores["pc_scores"]
        dominant_mode = max(pc_scores, key=lambda m: abs(pc_scores[m]))
        pc_dominant   = pc_scores[dominant_mode]

        quadrant = _assign_quadrant(
            scores["cc"],
            pc_dominant,
            anti["is_anti_sim"],
        )

        results.append({
            "head_idx":     h,
            "cc":           scores["cc"],
            "pc_scores":    pc_scores,          # full dict kept for Bug 12
            "pc_dominant":  pc_dominant,
            "pc_mode":      dominant_mode,
            "anti_sim_rho": anti["anti_sim_rho"],
            "n_pairs_used": anti["n_pairs_used"],
            "is_anti_sim":  anti["is_anti_sim"],
            "f_rot":        f_rot,
            "quadrant":     quadrant,
        })

    return results


# ---------------------------------------------------------------------------
# Cross-head correlation tests (P6-A2)  (Bug 12 fix: permutation p-values;
#                                         Bug 7  fix: |PC| instead of signed)
# ---------------------------------------------------------------------------

def _permutation_spearman(
    x:           np.ndarray,
    y:           np.ndarray,
    n_perm:      int = 10_000,
    alternative: str = "greater",
    seed:        int = 42,
) -> tuple[float, float]:
    """
    Empirical Spearman p-value via permutation of `y`.

    Parameters
    ----------
    x, y        : 1-D arrays of equal length.
    n_perm      : number of random permutations of y.  10k is enough for
                  alpha = 0.05 with healthy margin; bump to 100k if the
                  observed rho is borderline.
    alternative : "greater"   — H1: rho > 0   (one-tailed, the P6-A2 case)
                  "less"      — H1: rho < 0
                  "two-sided" — H1: rho != 0
    seed        : RNG seed for reproducibility.

    Returns
    -------
    (rho_observed, p_value)

    The p-value uses the Phipson-Smyth (2010) adjustment
        p = (#{rho_perm beats rho_obs} + 1) / (n_perm + 1)
    which keeps the test conservative and avoids p = 0 for finite n_perm.
    With n < 4, returns (rho_obs, 1.0) since permutation is degenerate.
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

    rng      = np.random.default_rng(seed)
    rho_null = np.empty(n_perm, dtype=np.float64)
    y_perm   = np.array(y, copy=True)
    for i in range(n_perm):
        rng.shuffle(y_perm)
        r, _        = spearmanr(x, y_perm)
        rho_null[i] = r if np.isfinite(r) else 0.0

    if alternative == "greater":
        n_extreme = int(np.sum(rho_null >= rho_obs))
    elif alternative == "less":
        n_extreme = int(np.sum(rho_null <= rho_obs))
    elif alternative == "two-sided":
        n_extreme = int(np.sum(np.abs(rho_null) >= abs(rho_obs)))
    else:
        raise ValueError(f"unknown alternative: {alternative!r}")

    p_value = (n_extreme + 1) / (n_perm + 1)
    return float(rho_obs), float(p_value)


def cross_head_correlations(
    head_metrics: list[dict],
    threshold:    float = 0.4,
    alpha:        float = 0.05,
    n_perm:       int   = 10_000,
    seed:         int   = 42,
) -> dict:
    """
    Test P6-A2: rotational heads (high f_rot) should have low CC and high |PC|.

    Verdict requires BOTH:
      (a) statistical significance: permutation p < `alpha` for both
          rho(f_rot, -CC) and rho(f_rot, |PC|), one-tailed (alternative="greater")
      (b) effect size: rho > `threshold` for both

    The two-condition gate prevents both:
      — large-n false alarms (a tiny rho that scrapes p < alpha just from many
        heads),
      — small-n false negatives (a large rho that's real but p > alpha from low
        df — flagged via n_heads_too_small_for_alpha so the report doesn't read
        it as a failure when it's actually under-powered).

    Bug 7 note : uses max(|pc_scores|) instead of signed pc_dominant.
    Bug 12 note: replaces hard threshold-only verdict with permutation p-values.
    """
    if not head_metrics:
        return {
            "rho_f_rot_neg_cc":            0.0,
            "p_value_neg_cc":              1.0,
            "rho_f_rot_abs_pc":            0.0,
            "p_value_abs_pc":              1.0,
            "n_heads":                     0,
            "alpha":                       alpha,
            "threshold":                   threshold,
            "n_perm":                      n_perm,
            "n_heads_too_small_for_alpha": True,
            "significance_passes":         False,
            "effect_size_passes":          False,
            "p6_a2_passes":                False,
        }

    f_rot  = np.asarray([h["f_rot"] for h in head_metrics], dtype=np.float64)
    cc     = np.asarray([h["cc"]    for h in head_metrics], dtype=np.float64)
    pc_mag = np.asarray(
        [
            max((abs(v) for v in h["pc_scores"].values()), default=0.0)
            for h in head_metrics
        ],
        dtype=np.float64,
    )

    n_heads = int(len(f_rot))
    # Minimum n_heads to reach p < alpha under permutation.
    # Below ~5 heads the test is under-powered regardless.
    min_n_for_alpha = max(5, int(np.ceil(1.0 / alpha)))
    too_small       = n_heads < min_n_for_alpha

    rho_cc, p_cc = _permutation_spearman(
        f_rot, -cc,    n_perm=n_perm, alternative="greater", seed=seed,
    )
    rho_pc, p_pc = _permutation_spearman(
        f_rot, pc_mag, n_perm=n_perm, alternative="greater", seed=seed + 1,
    )

    significance_passes = (p_cc < alpha) and (p_pc < alpha)
    effect_size_passes  = (rho_cc > threshold) and (rho_pc > threshold)
    p6_a2_passes        = bool(significance_passes and effect_size_passes and not too_small)

    return {
        "rho_f_rot_neg_cc":            rho_cc,
        "p_value_neg_cc":              p_cc,
        "rho_f_rot_abs_pc":            rho_pc,
        "p_value_abs_pc":              p_pc,
        "n_heads":                     n_heads,
        "alpha":                       alpha,
        "threshold":                   threshold,
        "n_perm":                      n_perm,
        "n_heads_too_small_for_alpha": too_small,
        "significance_passes":         bool(significance_passes),
        "effect_size_passes":          bool(effect_size_passes),
        "p6_a2_passes":                p6_a2_passes,
    }


# ---------------------------------------------------------------------------
# Structured output for reporting
# ---------------------------------------------------------------------------

def head_map_data(
    head_records: list[dict],
    layer_name:   str = "shared",
) -> dict:
    """
    Package head classification results for plotting / reporting.

    Returns a dict suitable for JSON serialisation (no numpy arrays).
    """
    quadrant_counts: dict[str, int] = {}
    for r in head_records:
        q = r["quadrant"]
        quadrant_counts[q] = quadrant_counts.get(q, 0) + 1

    anti_sim_heads  = [r["head_idx"] for r in head_records if r["is_anti_sim"]]
    positional_heads = [
        r["head_idx"] for r in head_records
        if r["quadrant"] in ("positional", "induction")
    ]

    corr = cross_head_correlations(head_records)

    return {
        "layer_name":      layer_name,
        "n_heads":         len(head_records),
        "quadrant_counts": quadrant_counts,
        "anti_sim_heads":  anti_sim_heads,
        "positional_heads": positional_heads,
        "cross_head_corr": corr,
        "head_records":    [{k: v for k, v in r.items()} for r in head_records],
    }
