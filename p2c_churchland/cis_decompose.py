"""
cis_decompose.py — C3: dPCA-style invariant/specific split, projection onto S/A.

Implements the Kaufman et al. 2016 CIS decomposition ported to transformer
layer trajectories.

For K matched-length prompts, at each layer L:
  invariant^(L) = (1/K) Σ_k X_k^(L)           — cross-prompt mean  (n_tokens, d)
  specific_k^(L) = X_k^(L) - invariant^(L)     — per-prompt residual (n_tokens, d)

Then project each component onto P_A and P_S to ask which channel carries
prompt-shared vs prompt-specific variance.

Predictions tested:
  P2c-K1 : invariant variance projects predominantly onto A;
            specific variance projects predominantly onto S.
  P2c-K2 : invariant component velocity (||Δinvariant^(L)||_F) peaks at
            Phase 1 merge layers, not plateau layers.

Caveat (from README): use a purpose-built prompt grid (matched_length.json),
not the heterogeneous Phase 1 prompts. Prompts should vary along one axis only.

Functions
---------
compute_cis_decomposition    : cross-prompt mean + per-prompt residual
channel_variance_per_layer   : per-layer Frobenius norms in A and S channels
aggregate_channel_fractions  : K1 verdict — fraction of invariant/specific in A vs S
invariant_layer_velocity     : per-layer ||Δinvariant||_F (K2 input)
merge_layer_test             : K2 — velocity spike at merge layers
analyze_cis                  : full C3 pipeline
print_cis                    : terminal report
cis_to_json                  : JSON-serializable summary
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Step 1 – CIS decomposition
# ---------------------------------------------------------------------------

def compute_cis_decomposition(
    activations_per_prompt: list[np.ndarray],
) -> dict:
    """
    Decompose K prompt trajectories into invariant (cross-prompt mean) and
    specific (per-prompt residual) components.

    Parameters
    ----------
    activations_per_prompt : list of K arrays, each (n_layers, n_tokens, d).
                             All arrays must have the same shape.

    Returns
    -------
    dict with:
      invariant : (n_layers, n_tokens, d) — cross-prompt mean at each layer
      specific  : (K, n_layers, n_tokens, d) — per-prompt residuals
      K         : int — number of prompts
      n_layers  : int
      n_tokens  : int
      d         : int
    """
    K = len(activations_per_prompt)
    assert K >= 2, "Need at least 2 prompts for CIS decomposition."

    stack = np.stack(activations_per_prompt, axis=0)   # (K, n_layers, n_tokens, d)
    invariant = stack.mean(axis=0)                      # (n_layers, n_tokens, d)
    specific  = stack - invariant[np.newaxis]           # (K, n_layers, n_tokens, d)

    n_layers, n_tokens, d = invariant.shape
    return {
        "invariant": invariant,
        "specific":  specific,
        "K":         K,
        "n_layers":  n_layers,
        "n_tokens":  n_tokens,
        "d":         d,
    }


# ---------------------------------------------------------------------------
# Step 2 – Channel variance per layer
# ---------------------------------------------------------------------------

def channel_variance_per_layer(
    decomp: dict,
    P_A: np.ndarray,
    P_S: np.ndarray,
) -> dict:
    """
    For each layer, compute the Frobenius-norm² of invariant and mean-specific
    components projected onto P_A and P_S.

    Variance metric: for a matrix M (n_tokens, d),
        var_P(M) = ||M @ P||²_F  =  Σ_i (P m_i)·(P m_i)

    Parameters
    ----------
    decomp : output of compute_cis_decomposition
    P_A    : (d, d) imaginary-channel projector
    P_S    : (d, d) real-channel projector

    Returns
    -------
    dict with per-layer arrays (n_layers,):
      inv_var_A    : invariant variance in A channel
      inv_var_S    : invariant variance in S channel
      inv_var_total: invariant total Frobenius² (sanity check: ≈ inv_A + inv_S)
      spec_var_A   : mean specific variance in A channel (averaged over K prompts)
      spec_var_S   : mean specific variance in S channel
      spec_var_total: mean specific total Frobenius²
      inv_frac_A   : inv_var_A / (inv_var_A + inv_var_S) per layer
      spec_frac_S  : spec_var_S / (spec_var_S + spec_var_A) per layer
    """
    inv  = decomp["invariant"]   # (n_layers, n_tokens, d)
    spec = decomp["specific"]    # (K, n_layers, n_tokens, d)
    K    = decomp["K"]

    def _frob2_proj(M, P):
        """||M @ P||²_F for M (n_tokens, d)"""
        MP = M @ P   # (n_tokens, d)
        return float(np.sum(MP ** 2))

    n_layers = decomp["n_layers"]
    inv_A, inv_S, inv_tot = [], [], []
    spec_A, spec_S, spec_tot = [], [], []

    for L in range(n_layers):
        iA = _frob2_proj(inv[L], P_A)
        iS = _frob2_proj(inv[L], P_S)
        inv_A.append(iA);  inv_S.append(iS)
        inv_tot.append(_frob2_proj(inv[L], np.eye(P_A.shape[0])))

        sA = np.mean([_frob2_proj(spec[k, L], P_A) for k in range(K)])
        sS = np.mean([_frob2_proj(spec[k, L], P_S) for k in range(K)])
        spec_A.append(sA);  spec_S.append(sS)
        spec_tot.append(np.mean([
            _frob2_proj(spec[k, L], np.eye(P_A.shape[0])) for k in range(K)
        ]))

    inv_A  = np.array(inv_A);   inv_S  = np.array(inv_S)
    spec_A = np.array(spec_A);  spec_S = np.array(spec_S)

    eps = 1e-30
    inv_frac_A  = inv_A  / np.maximum(inv_A  + inv_S,  eps)
    spec_frac_S = spec_S / np.maximum(spec_S + spec_A, eps)

    return {
        "inv_var_A":     inv_A,
        "inv_var_S":     inv_S,
        "inv_var_total": np.array(inv_tot),
        "spec_var_A":    spec_A,
        "spec_var_S":    spec_S,
        "spec_var_total":np.array(spec_tot),
        "inv_frac_A":    inv_frac_A,
        "spec_frac_S":   spec_frac_S,
    }


# ---------------------------------------------------------------------------
# Step 3 – Aggregate K1 verdict
# ---------------------------------------------------------------------------

def aggregate_channel_fractions(cv: dict) -> dict:
    """
    Summarise K1: is invariant variance predominantly in A?
                  Is specific variance predominantly in S?

    Uses layer-summed (not per-layer-averaged) variances for the final
    verdict, so layers with more total variance contribute proportionally.

    Parameters
    ----------
    cv : output of channel_variance_per_layer

    Returns
    -------
    dict with:
      global_inv_frac_A   : float — Σ_L inv_A / Σ_L (inv_A + inv_S)
      global_spec_frac_S  : float — Σ_L spec_S / Σ_L (spec_S + spec_A)
      p2ck1_inv_in_A      : bool — global_inv_frac_A > 0.5
      p2ck1_spec_in_S     : bool — global_spec_frac_S > 0.5
      p2ck1_holds         : bool — both hold
    """
    eps = 1e-30
    inv_frac  = float(cv["inv_var_A"].sum()  /
                      max((cv["inv_var_A"]  + cv["inv_var_S"]).sum(),  eps))
    spec_frac = float(cv["spec_var_S"].sum() /
                      max((cv["spec_var_S"] + cv["spec_var_A"]).sum(), eps))

    return {
        "global_inv_frac_A":  inv_frac,
        "global_spec_frac_S": spec_frac,
        "p2ck1_inv_in_A":     inv_frac  > 0.5,
        "p2ck1_spec_in_S":    spec_frac > 0.5,
        "p2ck1_holds":        inv_frac > 0.5 and spec_frac > 0.5,
    }


# ---------------------------------------------------------------------------
# Step 4 – Invariant velocity (K2 input)
# ---------------------------------------------------------------------------

def invariant_layer_velocity(invariant: np.ndarray) -> np.ndarray:
    """
    Per-layer velocity of the invariant component:
        vel^(L) = ||invariant^(L+1) - invariant^(L)||_F  (mean over tokens)

    Parameters
    ----------
    invariant : (n_layers, n_tokens, d)

    Returns
    -------
    vel : (n_layers - 1,) — one value per layer transition
    """
    delta = invariant[1:] - invariant[:-1]          # (n_layers-1, n_tokens, d)
    # Per-token norm, then mean over tokens
    token_norms = np.linalg.norm(delta, axis=-1)    # (n_layers-1, n_tokens)
    return token_norms.mean(axis=-1)                 # (n_layers-1,)


# ---------------------------------------------------------------------------
# Step 5 – K2 verdict
# ---------------------------------------------------------------------------

def merge_layer_test(
    velocities:     np.ndarray,
    merge_layers:   list[int],
    plateau_layers: list[int],
) -> dict:
    """
    K2: does the invariant-component velocity spike at merge layers?

    velocity[L] is the transition L → L+1. A "merge event at layer M"
    corresponds to the transition M-1 → M, so we use velocity[M-1] as the
    merge-layer velocity (clamped to valid range).

    Parameters
    ----------
    velocities     : (n_layers - 1,) from invariant_layer_velocity
    merge_layers   : layer indices flagged as merge events by Phase 1
    plateau_layers : layer indices flagged as plateau interiors

    Returns
    -------
    dict with:
      merge_mean_vel   : float
      plateau_mean_vel : float
      p2ck2_holds      : bool — merge_mean_vel > plateau_mean_vel
      k2_effect        : float — merge - plateau (>0 → K2 holds)
    """
    T = len(velocities)

    def _idx(layer):
        i = max(layer - 1, 0)
        return min(i, T - 1)

    merge_vals   = [velocities[_idx(L)] for L in merge_layers   if 0 <= _idx(L) < T]
    plateau_vals = [velocities[_idx(L)] for L in plateau_layers if 0 <= _idx(L) < T]

    merge_mean   = float(np.mean(merge_vals))   if merge_vals   else float("nan")
    plateau_mean = float(np.mean(plateau_vals)) if plateau_vals else float("nan")
    effect       = merge_mean - plateau_mean

    return {
        "merge_mean_vel":   merge_mean,
        "plateau_mean_vel": plateau_mean,
        "p2ck2_holds":      merge_mean > plateau_mean,
        "k2_effect":        effect,
        "merge_vals":       merge_vals,
        "plateau_vals":     plateau_vals,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def analyze_cis(
    activations_per_prompt: list[np.ndarray],
    P_A: np.ndarray,
    P_S: np.ndarray,
    merge_layers:   list[int] | None = None,
    plateau_layers: list[int] | None = None,
) -> dict:
    """
    Full C3 CIS analysis.

    Parameters
    ----------
    activations_per_prompt : list of K arrays (n_layers, n_tokens, d)
    P_A, P_S               : (d, d) channel projectors
    merge_layers           : Phase 1 merge layer indices (for K2)
    plateau_layers         : Phase 1 plateau layer indices (for K2)

    Returns
    -------
    dict with:
      decomp      : compute_cis_decomposition output
      cv          : channel_variance_per_layer output
      k1          : aggregate_channel_fractions output
      velocities  : (n_layers-1,) invariant velocity per transition
      k2          : merge_layer_test output (or None if layers not provided)
      p2ck1_holds : bool
      p2ck2_holds : bool (or None)
    """
    decomp = compute_cis_decomposition(activations_per_prompt)
    cv     = channel_variance_per_layer(decomp, P_A, P_S)
    k1     = aggregate_channel_fractions(cv)
    vels   = invariant_layer_velocity(decomp["invariant"])

    k2 = None
    if merge_layers is not None and plateau_layers is not None:
        k2 = merge_layer_test(vels, merge_layers, plateau_layers)

    return {
        "decomp":      decomp,
        "cv":          cv,
        "k1":          k1,
        "velocities":  vels,
        "k2":          k2,
        "p2ck1_holds": k1["p2ck1_holds"],
        "p2ck2_holds": k2["p2ck2_holds"] if k2 else None,
    }


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

def print_cis(result: dict) -> None:
    sep = "-" * 60
    print(sep)
    print("C3 — Condition-Invariant Signal")
    print(sep)
    k1 = result["k1"]
    print(f"  Global invariant fraction in A : {k1['global_inv_frac_A']:.4f}")
    print(f"  Global specific  fraction in S : {k1['global_spec_frac_S']:.4f}")
    v1 = "HOLDS" if result["p2ck1_holds"] else "FAILS"
    print(f"  P2c-K1 {v1}: inv→A, spec→S")
    print()
    if result["k2"] is not None:
        k2 = result["k2"]
        v2 = "HOLDS" if result["p2ck2_holds"] else "FAILS"
        print(f"  Merge   mean velocity : {k2['merge_mean_vel']:.4f}")
        print(f"  Plateau mean velocity : {k2['plateau_mean_vel']:.4f}")
        print(f"  P2c-K2 {v2}: velocity spike at merge layers "
              f"(effect {k2['k2_effect']:+.4f})")
    print(sep)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def cis_to_json(result: dict) -> dict:
    cv = result["cv"]
    k1 = result["k1"]

    out: dict = {
        "k1": {
            "global_inv_frac_A":  float(k1["global_inv_frac_A"]),
            "global_spec_frac_S": float(k1["global_spec_frac_S"]),
            "p2ck1_holds":        bool(k1["p2ck1_holds"]),
            "inv_frac_A_per_layer":  cv["inv_frac_A"].tolist(),
            "spec_frac_S_per_layer": cv["spec_frac_S"].tolist(),
        },
        "velocities": result["velocities"].tolist(),
        "p2ck1_holds": bool(result["p2ck1_holds"]),
        "p2ck2_holds": result["p2ck2_holds"],
    }
    if result["k2"] is not None:
        k2 = result["k2"]
        out["k2"] = {
            "merge_mean_vel":   float(k2["merge_mean_vel"]),
            "plateau_mean_vel": float(k2["plateau_mean_vel"]),
            "k2_effect":        float(k2["k2_effect"]),
            "p2ck2_holds":      bool(k2["p2ck2_holds"]),
        }
    return out
