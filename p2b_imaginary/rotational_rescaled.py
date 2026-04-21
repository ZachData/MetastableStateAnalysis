"""
rotational_rescaled.py — Causal isolation of rotational vs signed dynamics.

Phase 2's rescaled frame factors out ALL of V_eff via z = e^{-tV} x.
This removes both signed and rotational components simultaneously.

This script decomposes V_eff = S + A where:
  S = (V + V^T) / 2  — symmetric part (signed, no rotation)
  A = (V - V^T) / 2  — antisymmetric part (pure rotation, no signed component)

Then applies three separate rescalings:
  1. Full:         z_full  = e^{-tV} x       (removes everything — Phase 2 baseline)
  2. Signed-only:  z_sign  = e^{-tS} x       (removes signed, keeps rotation)
  3. Rotation-only: z_rot  = e^{-tA} x       (removes rotation, keeps signed)

Recomputes energy violations in each frame. The comparison:
  - If signed-only removal matches full removal → rotation is dynamically neutral
  - If rotation-only removal also reduces violations → rotation contributes independently
  - If neither alone matches full → S and A interact nonlinearly

For per-layer models (GPT-2), each layer's own S and A are used.

Functions
---------
decompose_symmetric_antisymmetric : split V into S and A
rescaled_trajectory_component     : apply one-component rescaling
compare_rescaled_frames           : run all three + original, compare violations
analyze_rotational_rescaling      : full pipeline for one model × prompt
"""

import numpy as np
from scipy.linalg import expm, svdvals


# ---------------------------------------------------------------------------
# S/A decomposition
# ---------------------------------------------------------------------------

def decompose_symmetric_antisymmetric(OV: np.ndarray) -> dict:
    """
    Split V_eff into symmetric (signed) and antisymmetric (rotational) parts.

    S = (V + V^T) / 2  — eigenvalues are real, capture attraction/repulsion
    A = (V - V^T) / 2  — eigenvalues are purely imaginary, capture rotation

    Also computes the relative magnitude: ||A||_F / ||S||_F.
    When this ratio is large, the rotational component dominates.

    Parameters
    ----------
    OV : (d, d) ndarray

    Returns
    -------
    dict with:
      S              : (d, d) symmetric part
      A              : (d, d) antisymmetric part
      S_frob         : float — ||S||_F
      A_frob         : float — ||A||_F
      rotation_ratio : float — ||A||_F / ||S||_F
      V_frob         : float — ||V||_F (for reference)
    """
    S = (OV + OV.T) / 2.0
    A = (OV - OV.T) / 2.0

    s_norm = float(np.linalg.norm(S, 'fro'))
    a_norm = float(np.linalg.norm(A, 'fro'))
    v_norm = float(np.linalg.norm(OV, 'fro'))

    return {
        "S": S,
        "A": A,
        "S_frob": s_norm,
        "A_frob": a_norm,
        "rotation_ratio": float(a_norm / max(s_norm, 1e-12)),
        "V_frob": v_norm,
    }


# ---------------------------------------------------------------------------
# Component-wise rescaled trajectory
# ---------------------------------------------------------------------------

def rescaled_trajectory_component(
    activations: np.ndarray,
    matrix: np.ndarray,
    beta_values: list,
    is_per_layer: bool = False,
    matrices_list: list = None,
) -> dict:
    """
    Apply rescaling z_i(L) = x_i(L) @ ((e^{-M})^L)^T and recompute metrics.

    For shared-weight models: matrix is a single (d,d), applied cumulatively.
    For per-layer models: matrices_list is a list of (d,d), one per layer.
      The cumulative rescaling at layer L is Π_{l=0}^{L-1} e^{-M_l}.

    Parameters
    ----------
    activations   : (n_layers, n_tokens, d)
    matrix        : (d, d) for shared models
    beta_values   : list of float
    is_per_layer  : bool
    matrices_list : list of (d, d) for per-layer models

    Returns
    -------
    dict with:
      ip_mean       : (n_layers,) float
      ip_mass_near_1: (n_layers,) float
      energies      : {beta: (n_layers,) float}
      n_violations  : {beta: int}
      effective_rank: (n_layers,) float
      max_valid_layer: int — truncation point if rescaling diverges
    """
    n_layers, n_tokens, d = activations.shape

    # Precompute per-layer rescaling matrices
    if is_per_layer and matrices_list is not None:
        R_per_layer = [expm(-M).astype(np.float64) for M in matrices_list]
    else:
        R_single = expm(-matrix).astype(np.float64)

    # Apply rescaling cumulatively
    R_cum = np.eye(d, dtype=np.float64)
    rescaled = np.zeros((n_layers, n_tokens, d), dtype=np.float64)
    max_valid = 0

    for L in range(n_layers):
        raw = activations[L].astype(np.float64) @ R_cum.T
        if not np.all(np.isfinite(raw)):
            break
        norms = np.linalg.norm(raw, axis=-1, keepdims=True)
        rescaled[L] = raw / np.maximum(norms, 1e-10)
        max_valid = L + 1

        # Advance cumulative matrix
        if is_per_layer and matrices_list is not None:
            idx = min(L, len(R_per_layer) - 1)
            R_cum = R_cum @ R_per_layer[idx]
        else:
            R_cum = R_cum @ R_single

        if np.abs(R_cum).max() > 1e15:
            break

    # Compute metrics on rescaled trajectory
    ip_means = np.full(n_layers, np.nan)
    ip_mass = np.full(n_layers, np.nan)
    energies = {b: np.full(n_layers, np.nan) for b in beta_values}
    eff_rank = np.full(n_layers, np.nan)

    for L in range(max_valid):
        X = rescaled[L].astype(np.float32)
        G = X @ X.T

        idx_upper = np.triu_indices(n_tokens, k=1)
        ips = G[idx_upper]
        ip_means[L] = float(ips.mean())
        ip_mass[L] = float((ips > 0.9).mean())

        for beta in beta_values:
            exp_G = np.exp(beta * G)
            energies[beta][L] = float(
                exp_G.sum() / (2.0 * beta * n_tokens * n_tokens)
            )

        sv = svdvals(X)
        sv = sv[sv > 1e-10]
        if len(sv) > 0:
            sv_n = sv / sv.sum()
            entropy = -np.sum(sv_n * np.log(sv_n + 1e-12))
            eff_rank[L] = float(np.exp(entropy))

    # Count violations
    n_violations = {}
    for beta in beta_values:
        E = energies[beta]
        count = 0
        for L in range(1, max_valid):
            if (np.isfinite(E[L]) and np.isfinite(E[L - 1])
                    and E[L] - E[L - 1] < -1e-6
                    and eff_rank[L] >= 3.0):
                count += 1
        n_violations[beta] = count

    return {
        "ip_mean":        ip_means,
        "ip_mass_near_1": ip_mass,
        "energies":       energies,
        "n_violations":   n_violations,
        "effective_rank": eff_rank,
        "max_valid_layer": max_valid,
    }


# ---------------------------------------------------------------------------
# Original trajectory metrics (no rescaling)
# ---------------------------------------------------------------------------

def original_trajectory_metrics(
    activations: np.ndarray,
    beta_values: list,
) -> dict:
    """
    Compute energy and violation count on the unrescaled trajectory.
    Baseline for comparison with the three rescaled frames.
    """
    n_layers, n_tokens, d = activations.shape

    ip_means = np.full(n_layers, np.nan)
    ip_mass = np.full(n_layers, np.nan)
    energies = {b: np.full(n_layers, np.nan) for b in beta_values}
    eff_rank = np.full(n_layers, np.nan)

    for L in range(n_layers):
        X = activations[L]
        G = X @ X.T
        idx_upper = np.triu_indices(n_tokens, k=1)
        ips = G[idx_upper]
        ip_means[L] = float(ips.mean())
        ip_mass[L] = float((ips > 0.9).mean())

        for beta in beta_values:
            exp_G = np.exp(beta * G)
            energies[beta][L] = float(
                exp_G.sum() / (2.0 * beta * n_tokens * n_tokens)
            )

        sv = svdvals(X)
        sv = sv[sv > 1e-10]
        if len(sv) > 0:
            sv_n = sv / sv.sum()
            entropy = -np.sum(sv_n * np.log(sv_n + 1e-12))
            eff_rank[L] = float(np.exp(entropy))

    n_violations = {}
    for beta in beta_values:
        E = energies[beta]
        count = 0
        for L in range(1, n_layers):
            if (np.isfinite(E[L]) and np.isfinite(E[L - 1])
                    and E[L] - E[L - 1] < -1e-6
                    and eff_rank[L] >= 3.0):
                count += 1
        n_violations[beta] = count

    return {
        "ip_mean":        ip_means,
        "ip_mass_near_1": ip_mass,
        "energies":       energies,
        "n_violations":   n_violations,
        "effective_rank": eff_rank,
        "max_valid_layer": n_layers,
    }


# ---------------------------------------------------------------------------
# Compare all frames
# ---------------------------------------------------------------------------

def compare_rescaled_frames(
    activations: np.ndarray,
    ov_data: dict,
    beta_values: list,
) -> dict:
    """
    Run original + three rescaled frames and compare violation counts.

    Parameters
    ----------
    activations : (n_layers, n_tokens, d) — L2-normed
    ov_data     : output of weights.extract_ov_circuit
    beta_values : list of float

    Returns
    -------
    dict with:
      original      : trajectory metrics (baseline)
      full_rescaled : metrics after removing all of V
      signed_only   : metrics after removing S (signed part) only
      rotation_only : metrics after removing A (rotation part) only
      sa_decomp     : S/A norms and rotation ratio (per-layer for GPT-2)
      comparison    : per-beta violation counts and elimination rates
    """
    is_per_layer = ov_data["is_per_layer"]

    # Decompose V into S and A
    if is_per_layer:
        ov_list = ov_data["ov_total"]
        sa_list = [decompose_symmetric_antisymmetric(OV) for OV in ov_list]
        S_list = [sa["S"] for sa in sa_list]
        A_list = [sa["A"] for sa in sa_list]

        sa_summary = {
            "per_layer_rotation_ratio": [sa["rotation_ratio"] for sa in sa_list],
            "mean_rotation_ratio": float(np.mean([sa["rotation_ratio"] for sa in sa_list])),
            "layer_names": ov_data["layer_names"],
        }

        # The "V" for the full rescaling is the per-layer OV list
        original = original_trajectory_metrics(activations, beta_values)
        full_rescaled = rescaled_trajectory_component(
            activations, None, beta_values,
            is_per_layer=True, matrices_list=ov_list,
        )
        signed_only = rescaled_trajectory_component(
            activations, None, beta_values,
            is_per_layer=True, matrices_list=S_list,
        )
        rotation_only = rescaled_trajectory_component(
            activations, None, beta_values,
            is_per_layer=True, matrices_list=A_list,
        )
    else:
        OV = ov_data["ov_total"]
        sa = decompose_symmetric_antisymmetric(OV)
        S = sa["S"]
        A = sa["A"]

        sa_summary = {
            "rotation_ratio": sa["rotation_ratio"],
            "S_frob": sa["S_frob"],
            "A_frob": sa["A_frob"],
            "V_frob": sa["V_frob"],
        }

        original = original_trajectory_metrics(activations, beta_values)
        full_rescaled = rescaled_trajectory_component(
            activations, OV, beta_values,
        )
        signed_only = rescaled_trajectory_component(
            activations, S, beta_values,
        )
        rotation_only = rescaled_trajectory_component(
            activations, A, beta_values,
        )

    # Build comparison table
    comparison = {}
    for beta in beta_values:
        n_orig = original["n_violations"][beta]
        n_full = full_rescaled["n_violations"][beta]
        n_sign = signed_only["n_violations"][beta]
        n_rot = rotation_only["n_violations"][beta]

        comparison[beta] = {
            "n_original":          n_orig,
            "n_full_rescaled":     n_full,
            "n_signed_only":       n_sign,
            "n_rotation_only":     n_rot,
            "elim_full":           _elim_rate(n_orig, n_full),
            "elim_signed":         _elim_rate(n_orig, n_sign),
            "elim_rotation":       _elim_rate(n_orig, n_rot),
        }

    return {
        "original":       original,
        "full_rescaled":  full_rescaled,
        "signed_only":    signed_only,
        "rotation_only":  rotation_only,
        "sa_decomp":      sa_summary,
        "comparison":     comparison,
    }


def _elim_rate(n_orig: int, n_rescaled: int) -> float:
    """Fraction of violations eliminated by rescaling."""
    if n_orig == 0:
        return 0.0
    return float((n_orig - n_rescaled) / n_orig)


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def interpret_comparison(comparison: dict) -> dict:
    """
    Classify the rotational dynamics' causal role from violation comparison.

    Categories:
      rotation_neutral     — removing rotation alone doesn't help;
                             signed structure carries all causal weight
      rotation_contributes — removing rotation alone reduces violations;
                             rotational dynamics contribute independently
      rotation_dominant    — removing rotation alone matches or exceeds
                             signed-only removal
      interaction          — neither alone matches full removal;
                             S and A interact nonlinearly

    Returns per-beta classification and an overall assessment.
    """
    per_beta = {}
    classifications = []

    for beta, comp in comparison.items():
        e_full = comp["elim_full"]
        e_sign = comp["elim_signed"]
        e_rot = comp["elim_rotation"]

        if e_rot < 0.1:
            cat = "rotation_neutral"
        elif e_rot >= e_sign - 0.1:
            cat = "rotation_dominant"
        elif e_rot >= 0.1:
            cat = "rotation_contributes"
        else:
            cat = "rotation_neutral"

        # Check for interaction effect
        # If full > max(sign, rot) by a large margin, there's interaction
        if e_full > max(e_sign, e_rot) + 0.2:
            cat = "interaction"

        per_beta[beta] = {
            "classification": cat,
            "elim_full": e_full,
            "elim_signed": e_sign,
            "elim_rotation": e_rot,
        }
        classifications.append(cat)

    # Overall: majority vote across betas
    from collections import Counter
    counts = Counter(classifications)
    overall = counts.most_common(1)[0][0]

    return {
        "per_beta": per_beta,
        "overall": overall,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def analyze_rotational_rescaling(
    activations: np.ndarray,
    ov_data: dict,
    beta_values: list = None,
) -> dict:
    """
    Full Block 1b analysis: compare three rescaled frames.

    Parameters
    ----------
    activations : (n_layers, n_tokens, d) — L2-normed
    ov_data     : from weights.extract_ov_circuit
    beta_values : list of float (defaults to [0.1, 1.0, 2.0, 5.0])

    Returns
    -------
    dict with comparison results and interpretation
    """
    if beta_values is None:
        beta_values = [0.1, 1.0, 2.0, 5.0]

    frames = compare_rescaled_frames(activations, ov_data, beta_values)
    interp = interpret_comparison(frames["comparison"])

    return {
        "frames":         frames,
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------

def comparison_to_json(result: dict) -> dict:
    """Extract JSON-serializable summary from analyze_rotational_rescaling."""
    frames = result["frames"]

    out = {
        "sa_decomp":      frames["sa_decomp"],
        "comparison":      {},
        "interpretation":  result["interpretation"],
    }

    # Convert numpy in comparison
    for beta, comp in frames["comparison"].items():
        out["comparison"][str(beta)] = {
            k: int(v) if isinstance(v, (int, np.integer)) else float(v)
            for k, v in comp.items()
        }

    return out
