"""
tangling.py — C2: Q metric on full / S-only / A-only projections.

Implements the Russo et al. 2018 trajectory tangling metric Q ported to
transformer layer trajectories.

    Q_i(L) = max_{L' ≠ L} ||x^L_i - x^{L'}_i||² / (||Δx^L_i - Δx^{L'}_i||² + ε)

"Time" is layer depth. "Trajectory" for token i is its per-layer activation
x^(0)_i, ..., x^(T-2)_i (positions), paired with velocities Δx^(L)_i = x^(L+1)_i - x^(L)_i.

High Q: the flow field is not state-determined — the same position is visited
with very different velocities at different layers. Low Q: state-determined,
autonomous-like dynamics.

The metric is computed independently on:
  - full activation trajectories
  - S-channel projected trajectories (P_S @ x)
  - A-channel projected trajectories (P_A @ x)

Predictions tested:
  P2c-T1 : A-channel Q < S-channel Q
  P2c-T2 : Induction prompts have lower full Q than matched control prompts

Regime-mismatch note (from README): tangling was built for continuous-time
autonomous RNNs. Transformers are layer-discrete and non-autonomous. Q is
still computable and informative as a *relative* comparison across channels
and prompt types, not as an absolute autonomy test.

Functions
---------
extract_full_activations     : per-layer, per-token activations from a model
compute_velocities           : Δx = x^{L+1} - x^{L} for all layers
project_channel              : apply projector P to all (layer, token) activations
compute_Q_matrix             : (T, n_tokens) Q scores via vectorized broadcast
token_Q_summary              : max/mean/percentile aggregation over layers
tangling_three_channels      : full/S/A Q in one call
compare_prompt_groups        : T2 — induction vs control
analyze_tangling             : full C2 pipeline
print_tangling               : terminal report
tangling_to_json             : JSON-serializable summary
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Activation extraction (full per-layer, per-token)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_full_activations(
    model,
    tokenizer,
    prompt: str,
    device: str = "cpu",
) -> np.ndarray:
    """
    Run model on prompt and return all per-layer, per-token residual-stream
    activations.

    Parameters
    ----------
    model     : HuggingFace model with output_hidden_states=True support
    tokenizer : corresponding tokenizer
    prompt    : text prompt
    device    : torch device

    Returns
    -------
    activations : (n_layers, n_tokens, d) float64 array.
                  Layer 0 = embedding output; layers 1..L = transformer blocks.
    """
    inputs  = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True)
    hs = outputs.hidden_states   # tuple of (1, n_tokens, d), one per layer
    return np.stack(
        [hs[L][0].cpu().numpy() for L in range(len(hs))],
        axis=0,
    ).astype(np.float64)          # (n_layers, n_tokens, d)


# ---------------------------------------------------------------------------
# Velocity computation
# ---------------------------------------------------------------------------

def compute_velocities(activations: np.ndarray) -> np.ndarray:
    """
    Compute layer-to-layer displacement vectors.

    Parameters
    ----------
    activations : (n_layers, n_tokens, d)

    Returns
    -------
    velocities : (n_layers - 1, n_tokens, d)
                 vel[L] = act[L+1] - act[L]
    """
    return activations[1:] - activations[:-1]


# ---------------------------------------------------------------------------
# Channel projection
# ---------------------------------------------------------------------------

def project_channel(
    activations: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """
    Project every (layer, token) activation vector onto subspace P.

    Parameters
    ----------
    activations : (n_layers, n_tokens, d)
    P           : (d, d) orthogonal projector

    Returns
    -------
    projected : (n_layers, n_tokens, d)
    """
    # activations @ P.T == activations @ P since P is symmetric
    return activations @ P   # (n_layers, n_tokens, d)


# ---------------------------------------------------------------------------
# Q matrix computation
# ---------------------------------------------------------------------------

def compute_Q_matrix(
    positions:  np.ndarray,
    velocities: np.ndarray,
    eps:        float = 1e-4,
) -> np.ndarray:
    """
    Compute per-(layer, token) tangling scores Q.

    For token i at layer L:
        Q_i(L) = max_{L'≠L} ||pos[L,i] - pos[L',i]||²
                            / (||vel[L,i] - vel[L',i]||² + ε)

    Both positions and velocities must share the same time axis (T = n_layers - 1
    after velocities are computed, then positions are also sliced to T).

    Parameters
    ----------
    positions  : (T, n_tokens, d) — activations at layers 0..T-1
    velocities : (T, n_tokens, d) — Δx at layers 0..T-1
    eps        : regularisation to avoid division by zero

    Returns
    -------
    Q : (T, n_tokens) — Q_i(L) for all L, i
    """
    T, N, d = positions.shape
    assert velocities.shape == (T, N, d), (
        f"Shape mismatch: positions {positions.shape}, velocities {velocities.shape}"
    )

    # pos_diff[L, L', i] = ||pos[L,i] - pos[L',i]||²
    # We compute this by expanding the squared norm:
    #   ||a - b||² = ||a||² + ||b||² - 2 a·b

    pos_sq   = np.sum(positions ** 2, axis=-1)   # (T, N)
    vel_sq   = np.sum(velocities ** 2, axis=-1)  # (T, N)

    # CORRECT — token index 'i' shared between inputs; output is already (T, T, N)
    pos_dot = np.einsum("lid,mid->lmi", positions,  positions,  optimize=True)
    vel_dot = np.einsum("lid,mid->lmi", velocities, velocities, optimize=True)

    # Broadcast squared norms to (T, T, N)
    pos_sq_L  = pos_sq[:, np.newaxis, :]   # (T, 1, N)
    pos_sq_Lp = pos_sq[np.newaxis, :, :]   # (1, T, N)
    vel_sq_L  = vel_sq[:, np.newaxis, :]
    vel_sq_Lp = vel_sq[np.newaxis, :, :]

    pos_diff_sq = pos_sq_L + pos_sq_Lp - 2.0 * pos_dot   # (T, T, N)
    vel_diff_sq = vel_sq_L + vel_sq_Lp - 2.0 * vel_dot   # (T, T, N)

    # Clamp numerical negatives from floating-point error
    pos_diff_sq = np.maximum(pos_diff_sq, 0.0)
    vel_diff_sq = np.maximum(vel_diff_sq, 0.0)

    Q_raw = pos_diff_sq / (vel_diff_sq + eps)   # (T, T, N)

    # Mask diagonal (L == L'): set to 0 so max ignores self-comparison
    diag_mask = np.eye(T, dtype=bool)[:, :, np.newaxis]   # (T, T, 1)
    Q_raw[np.broadcast_to(diag_mask, Q_raw.shape)] = 0.0

    # Q_i(L) = max over L' ≠ L
    Q = Q_raw.max(axis=1)   # (T, N)
    return Q


# ---------------------------------------------------------------------------
# Token-level aggregation
# ---------------------------------------------------------------------------

def token_Q_summary(Q: np.ndarray) -> dict:
    """
    Aggregate Q matrix (T, n_tokens) into population-level statistics.

    Returns
    -------
    dict with:
      per_token_max    : (n_tokens,) — max Q over layers per token
      per_token_mean   : (n_tokens,) — mean Q over layers per token
      per_layer_mean   : (T,) — mean Q over tokens per layer
      population_mean  : float — mean of per_token_max
      population_median: float
      population_p95   : float — 95th percentile
    """
    per_tok_max  = Q.max(axis=0)     # (n_tokens,)
    per_tok_mean = Q.mean(axis=0)    # (n_tokens,)
    per_lay_mean = Q.mean(axis=1)    # (T,)

    return {
        "per_token_max":     per_tok_max,
        "per_token_mean":    per_tok_mean,
        "per_layer_mean":    per_lay_mean,
        "population_mean":   float(np.mean(per_tok_max)),
        "population_median": float(np.median(per_tok_max)),
        "population_p95":    float(np.percentile(per_tok_max, 95)),
    }


# ---------------------------------------------------------------------------
# Three-channel Q
# ---------------------------------------------------------------------------

def tangling_three_channels(
    activations: np.ndarray,
    P_A: np.ndarray,
    P_S: np.ndarray,
    eps: float = 1e-4,
    ) -> dict:
    """
    Compute Q for the full, S-channel, and A-channel trajectories.

    Parameters
    ----------
    activations : (n_layers, n_tokens, d)
    P_A         : (d, d) imaginary-channel projector
    P_S         : (d, d) real-channel projector
    eps         : Q regularisation

    Returns
    -------
    dict with:
      full : token_Q_summary dict for full activations
      S    : token_Q_summary dict for P_S-projected activations
      A    : token_Q_summary dict for P_A-projected activations
      p2ct1_holds : bool — A-channel population_mean < S-channel population_mean
      A_vs_S_ratio: float — A_mean / S_mean (< 1 → T1 holds)
    """
    # Velocities are aligned to positions 0..T-2 on both axes
    vels  = compute_velocities(activations)        # (T-1, n_tokens, d)
    pos   = activations[:-1]                        # (T-1, n_tokens, d)  slice to match

    pos_S = project_channel(pos,  P_S)
    vel_S = project_channel(vels, P_S)
    pos_A = project_channel(pos,  P_A)
    vel_A = project_channel(vels, P_A)

    Q_full = compute_Q_matrix(pos,   vels,  eps=eps)
    Q_S    = compute_Q_matrix(pos_S, vel_S, eps=eps)
    Q_A    = compute_Q_matrix(pos_A, vel_A, eps=eps)

    summ_full = token_Q_summary(Q_full)
    summ_S    = token_Q_summary(Q_S)
    summ_A    = token_Q_summary(Q_A)

    A_mean = summ_A["population_mean"]
    S_mean = summ_S["population_mean"]

    return {
        "full":          summ_full,
        "S":             summ_S,
        "A":             summ_A,
        # flat aliases for extended tests
        "Q_full":        summ_full["per_token_max"],
        "Q_S":           summ_S["per_token_max"],
        "Q_A":           summ_A["per_token_max"],
        "mean_Q_full":   summ_full["population_mean"],
        "mean_Q_S":      summ_S["population_mean"],
        "mean_Q_A":      summ_A["population_mean"],
        "p2ct1_holds":   A_mean < S_mean,
        "A_vs_S_ratio":  A_mean / max(S_mean, 1e-30),
    }


# ---------------------------------------------------------------------------
# T2: induction vs control comparison
# ---------------------------------------------------------------------------

def compare_prompt_groups(
    model,
    tokenizer,
    induction_prompts: list[str],
    control_prompts:   list[str],
    P_A:               np.ndarray,
    P_S:               np.ndarray,
    eps:               float = 1e-4,
    device:            str = "cpu",
) -> dict:
    """
    P2c-T2: compare full-trajectory Q between induction and control prompts.

    Induction prompts follow the `A B ... A → B` pattern; control prompts
    are matched-length with no induction structure.

    Parameters
    ----------
    model, tokenizer  : HuggingFace model + tokenizer
    induction_prompts : list of induction-pattern prompt strings
    control_prompts   : list of matched-length control prompt strings
    P_A, P_S          : (d, d) channel projectors (passed through to three_channels)
    eps               : Q regularisation
    device            : torch device

    Returns
    -------
    dict with:
      induction_Q_means   : (n_induction,) per-prompt population_mean Q
      control_Q_means     : (n_control,)   per-prompt population_mean Q
      induction_mean      : float
      control_mean        : float
      p2ct2_holds         : bool — induction_mean < control_mean
      t2_effect           : float — control_mean - induction_mean (>0 → T2 holds)
      mwu_pvalue          : float — Mann–Whitney U p-value (one-sided)
    """
    def _prompt_Q(prompts):
        means = []
        for p in prompts:
            acts = extract_full_activations(model, tokenizer, p, device=device)
            res  = tangling_three_channels(acts, P_A, P_S, eps=eps)
            means.append(res["full"]["population_mean"])
        return np.array(means)

    ind_Q = _prompt_Q(induction_prompts)
    ctl_Q = _prompt_Q(control_prompts)

    ind_mean = float(np.mean(ind_Q))
    ctl_mean = float(np.mean(ctl_Q))

    # Mann–Whitney U (one-sided: induction < control)
    try:
        from scipy.stats import mannwhitneyu
        _, pval = mannwhitneyu(ind_Q, ctl_Q, alternative="less")
        pval = float(pval)
    except ImportError:
        pval = float("nan")

    return {
        "induction_Q_means": ind_Q,
        "control_Q_means":   ctl_Q,
        "induction_mean":    ind_mean,
        "control_mean":      ctl_mean,
        "p2ct2_holds":       ind_mean < ctl_mean,
        "t2_effect":         ctl_mean - ind_mean,
        "mwu_pvalue":        pval,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def analyze_tangling(
    model,
    tokenizer,
    prompts: list[str],
    P_A: np.ndarray,
    P_S: np.ndarray,
    induction_prompts: list[str] | None = None,
    control_prompts:   list[str] | None = None,
    eps:   float = 1e-4,
    device: str = "cpu",
) -> dict:
    """
    Full C2 analysis.

    Parameters
    ----------
    model, tokenizer   : HuggingFace model + tokenizer
    prompts            : Phase 1 prompts — used for T1 (full/S/A channel comparison)
    P_A, P_S           : (d, d) channel projectors
    induction_prompts  : optional list of induction-pattern prompts for T2
    control_prompts    : optional matched-length control prompts for T2
    eps                : Q regularisation
    device             : torch device

    Returns
    -------
    dict with:
      per_prompt_t1    : list of tangling_three_channels results, one per prompt
      pooled_t1        : pooled T1 statistics (mean across prompts)
      t2               : compare_prompt_groups result (or None if not provided)
      p2ct1_holds      : bool
      p2ct2_holds      : bool (or None)
    """
    per_prompt = []
    for p in prompts:
        acts = extract_full_activations(model, tokenizer, p, device=device)
        per_prompt.append(tangling_three_channels(acts, P_A, P_S, eps=eps))

    # Pooled T1
    A_means    = [r["A"]["population_mean"]    for r in per_prompt]
    S_means    = [r["S"]["population_mean"]    for r in per_prompt]
    full_means = [r["full"]["population_mean"] for r in per_prompt]
    pool_A = float(np.mean(A_means))
    pool_S = float(np.mean(S_means))

    pooled_t1 = {
        "mean_Q_full": float(np.mean(full_means)),
        "mean_Q_S":    pool_S,
        "mean_Q_A":    pool_A,
        "A_vs_S_ratio": pool_A / max(pool_S, 1e-30),
        "p2ct1_holds":  pool_A < pool_S,
    }

    t2 = None
    if induction_prompts and control_prompts:
        t2 = compare_prompt_groups(
            model, tokenizer,
            induction_prompts, control_prompts,
            P_A, P_S, eps=eps, device=device,
        )

    return {
        "per_prompt_t1": per_prompt,
        "pooled_t1":     pooled_t1,
        "t2":            t2,
        "p2ct1_holds":   pooled_t1["p2ct1_holds"],
        "p2ct2_holds":   t2["p2ct2_holds"] if t2 else None,
    }


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

def print_tangling(result: dict) -> None:
    sep = "-" * 60
    print(sep)
    print("C2 — Trajectory Tangling")
    print(sep)
    pt = result["pooled_t1"]
    print(f"  Pooled mean Q (full) : {pt['mean_Q_full']:.4f}")
    print(f"  Pooled mean Q (S)    : {pt['mean_Q_S']:.4f}")
    print(f"  Pooled mean Q (A)    : {pt['mean_Q_A']:.4f}")
    print(f"  A/S ratio            : {pt['A_vs_S_ratio']:.4f}")
    t1 = "HOLDS" if result["p2ct1_holds"] else "FAILS"
    print(f"  P2c-T1 {t1}: A-channel Q < S-channel Q")
    print()

    if result["t2"] is not None:
        t2 = result["t2"]
        v  = "HOLDS" if result["p2ct2_holds"] else "FAILS"
        print(f"  Induction Q mean : {t2['induction_mean']:.4f}  "
              f"(n={len(t2['induction_Q_means'])})")
        print(f"  Control   Q mean : {t2['control_mean']:.4f}  "
              f"(n={len(t2['control_Q_means'])})")
        print(f"  P2c-T2 {v}: induction Q < control Q  "
              f"(effect {t2['t2_effect']:+.4f}, MWU p={t2['mwu_pvalue']:.4f})")
    print(sep)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def tangling_to_json(result: dict) -> dict:
    per_prompt_summary = []
    for r in result["per_prompt_t1"]:
        per_prompt_summary.append({
            "Q_full":      float(r["full"]["population_mean"]),
            "Q_S":         float(r["S"]["population_mean"]),
            "Q_A":         float(r["A"]["population_mean"]),
            "A_vs_S_ratio": float(r["A_vs_S_ratio"]),
            "p2ct1_holds": bool(r["p2ct1_holds"]),
        })

    out = {
        "per_prompt": per_prompt_summary,
        "pooled_t1":  result["pooled_t1"],
        "p2ct1_holds": bool(result["p2ct1_holds"]),
        "p2ct2_holds": result["p2ct2_holds"],
    }
    if result["t2"] is not None:
        t2 = result["t2"]
        out["t2"] = {
            "induction_Q_means": t2["induction_Q_means"].tolist(),
            "control_Q_means":   t2["control_Q_means"].tolist(),
            "induction_mean":    float(t2["induction_mean"]),
            "control_mean":      float(t2["control_mean"]),
            "t2_effect":         float(t2["t2_effect"]),
            "mwu_pvalue":        float(t2["mwu_pvalue"]),
        }
    return out
