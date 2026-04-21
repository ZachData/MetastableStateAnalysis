"""
ffn_rotation.py — FFN projection onto V_eff's rotation planes.

Phase 2's ffn_subspace.py projects FFN deltas onto the signed (symmetric)
eigensubspaces. This script extends that to the rotational structure.

Questions answered:
  - Does the FFN selectively amplify specific rotation planes?
  - Is the FFN blind to rotational structure (acts only on signed directions)?
  - At violation layers, does the FFN's rotational component differ from
    non-violation layers?

If the FFN amplifies rotation planes, then removing A alone (in
rotational_rescaled.py) may not eliminate violations because the FFN
re-introduces rotational displacement after the linear removal.

Functions
---------
project_ffn_onto_rotation_planes : per-layer projection fractions
compare_ffn_rotation_at_violations : z-score at violation vs population
analyze_ffn_rotation             : full pipeline
"""

import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Core projection
# ---------------------------------------------------------------------------

def project_ffn_onto_rotation_planes(
    ffn_deltas: np.ndarray,
    rotation_projectors,
    is_per_layer: bool = False,
) -> dict:
    """
    Project FFN updates onto V_eff's rotation planes, real subspace,
    and the residual (neither rotation nor real eigensubspace).

    Parameters
    ----------
    ffn_deltas          : (n_layers, n_tokens, d) — FFN contribution per layer
    rotation_projectors : from rotational_schur.build_rotation_plane_projectors
                          Single dict for shared, list of dicts for per-layer.
    is_per_layer        : bool

    Returns
    -------
    dict with per-layer arrays (n_layers,):
      ffn_rotation_frac  : fraction of FFN energy in combined rotation subspace
      ffn_real_frac      : fraction in real (non-rotating) subspace
      ffn_residual_frac  : fraction in neither
      ffn_top_plane_fracs: (n_layers, top_k) — fraction in each top-k plane
      ffn_total_energy   : total ||FFN_delta||² per layer
    """
    n_layers = ffn_deltas.shape[0]
    rot_list = rotation_projectors if isinstance(rotation_projectors, list) else [rotation_projectors]

    # Determine top_k from the first entry
    top_k = len(rot_list[0]["top_k_projectors"])

    result = {
        "ffn_rotation_frac":   np.zeros(n_layers),
        "ffn_real_frac":       np.zeros(n_layers),
        "ffn_residual_frac":   np.zeros(n_layers),
        "ffn_top_plane_fracs": np.zeros((n_layers, top_k)),
        "ffn_total_energy":    np.zeros(n_layers),
    }

    for L in range(n_layers):
        D = ffn_deltas[L]                              # (n_tokens, d)
        total = np.sum(D ** 2)
        result["ffn_total_energy"][L] = total

        if total < 1e-12:
            continue

        # Select projectors for this layer
        if is_per_layer:
            rp = rot_list[min(L, len(rot_list) - 1)]
        else:
            rp = rot_list[0]

        # Combined rotation subspace
        P_rot = rp["combined_rotation"]
        rot_energy = np.sum((D @ P_rot) ** 2)
        result["ffn_rotation_frac"][L] = rot_energy / total

        # Real (non-rotating) subspace
        P_real = rp["real_subspace"]
        real_energy = np.sum((D @ P_real) ** 2)
        result["ffn_real_frac"][L] = real_energy / total

        # Residual
        result["ffn_residual_frac"][L] = max(
            0.0, 1.0 - (rot_energy + real_energy) / total
        )

        # Per top-k plane
        for j, P_j in enumerate(rp["top_k_projectors"]):
            if j >= top_k:
                break
            result["ffn_top_plane_fracs"][L, j] = np.sum((D @ P_j) ** 2) / total

    return result


# ---------------------------------------------------------------------------
# Violation vs population comparison
# ---------------------------------------------------------------------------

def compare_ffn_rotation_at_violations(
    ffn_rot_projection: dict,
    phase1_events: dict,
    beta: float = 1.0,
) -> dict:
    """
    Compare FFN rotation fractions at violation layers vs non-violation layers.

    If the FFN selectively amplifies rotation at violation layers,
    ffn_rotation_frac should be elevated (positive z-score).
    If the FFN is blind to rotational structure, z-score ≈ 0.

    Parameters
    ----------
    ffn_rot_projection : from project_ffn_onto_rotation_planes
    phase1_events      : from trajectory.load_phase1_events
    beta               : which beta for violations

    Returns
    -------
    dict with z-scores and means for rotation_frac, real_frac
    """
    violations = set(phase1_events.get("energy_violations", {}).get(beta, []))
    n_layers = len(ffn_rot_projection["ffn_rotation_frac"])

    # FFN at layer L-1 produces the update that causes violation at layer L
    violation_indices = {v - 1 for v in violations if 0 <= v - 1 < n_layers}

    result = {}
    for metric in ["ffn_rotation_frac", "ffn_real_frac"]:
        values = ffn_rot_projection[metric]
        v_idx = sorted(violation_indices)
        pop_idx = [i for i in range(n_layers) if i not in violation_indices]

        if not v_idx or not pop_idx:
            result[metric] = {
                "z_score": float("nan"),
                "v_mean": float("nan"),
                "pop_mean": float("nan"),
            }
            continue

        v_vals = values[v_idx]
        pop_vals = values[pop_idx]
        pop_std = float(np.std(pop_vals))

        result[metric] = {
            "z_score":  float((np.mean(v_vals) - np.mean(pop_vals)) / (pop_std + 1e-12)),
            "v_mean":   float(np.mean(v_vals)),
            "pop_mean": float(np.mean(pop_vals)),
            "pop_std":  pop_std,
            "n_violations": len(v_idx),
            "n_population": len(pop_idx),
        }

    return result


# ---------------------------------------------------------------------------
# Per-violation classification
# ---------------------------------------------------------------------------

def classify_ffn_rotation_per_violation(
    ffn_deltas: np.ndarray,
    rotation_projectors,
    phase1_events: dict,
    is_per_layer: bool = False,
    beta: float = 1.0,
) -> list:
    """
    For each violation layer, classify the FFN's rotational role.

    Categories:
      rotation_dominant  : FFN pushes primarily into rotation planes
      real_dominant      : FFN pushes primarily into real (signed) subspace
      mixed              : FFN engages both roughly equally
      orthogonal         : FFN operates outside V_eff's eigenstructure entirely

    Returns list of per-violation dicts.
    """
    violations = phase1_events.get("energy_violations", {}).get(beta, [])
    n_layers = ffn_deltas.shape[0]
    rot_list = rotation_projectors if isinstance(rotation_projectors, list) else [rotation_projectors]

    results = []
    for v_layer in violations:
        t_idx = v_layer - 1
        if t_idx < 0 or t_idx >= n_layers:
            continue

        D = ffn_deltas[t_idx]
        total = np.sum(D ** 2)
        if total < 1e-12:
            continue

        rp = rot_list[min(t_idx, len(rot_list) - 1)] if is_per_layer else rot_list[0]

        rot_frac = np.sum((D @ rp["combined_rotation"]) ** 2) / total
        real_frac = np.sum((D @ rp["real_subspace"]) ** 2) / total
        residual_frac = max(0.0, 1.0 - rot_frac - real_frac)

        if rot_frac > 0.5 and rot_frac > real_frac:
            role = "rotation_dominant"
        elif real_frac > 0.5 and real_frac > rot_frac:
            role = "real_dominant"
        elif rot_frac + real_frac > 0.5:
            role = "mixed"
        else:
            role = "orthogonal"

        results.append({
            "layer":         v_layer,
            "rotation_frac": float(rot_frac),
            "real_frac":     float(real_frac),
            "residual_frac": float(residual_frac),
            "role":          role,
        })

    return results


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def analyze_ffn_rotation(
    ffn_deltas: np.ndarray,
    rotation_projectors,
    phase1_events: dict,
    is_per_layer: bool = False,
    beta: float = 1.0,
) -> dict:
    """
    Full FFN-rotation analysis for one model × prompt.

    Parameters
    ----------
    ffn_deltas          : (n_layers, n_tokens, d)
    rotation_projectors : from rotational_schur
    phase1_events       : from trajectory.load_phase1_events
    is_per_layer        : bool
    beta                : float

    Returns
    -------
    dict with projection, comparison, and per-violation results
    """
    projection = project_ffn_onto_rotation_planes(
        ffn_deltas, rotation_projectors, is_per_layer,
    )
    comparison = compare_ffn_rotation_at_violations(
        projection, phase1_events, beta,
    )
    per_violation = classify_ffn_rotation_per_violation(
        ffn_deltas, rotation_projectors, phase1_events, is_per_layer, beta,
    )

    # Summary counts
    role_counts = {}
    for pv in per_violation:
        role_counts[pv["role"]] = role_counts.get(pv["role"], 0) + 1

    return {
        "projection":     projection,
        "comparison":     comparison,
        "per_violation":  per_violation,
        "role_counts":    role_counts,
        "n_violations":   len(per_violation),
    }


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------

def ffn_rotation_to_json(result: dict) -> dict:
    """Extract JSON-serializable summary."""
    comp = result["comparison"]

    out = {
        "role_counts":   result["role_counts"],
        "n_violations":  result["n_violations"],
    }

    for metric, data in comp.items():
        if isinstance(data, dict):
            out[f"{metric}_z_score"] = data.get("z_score")
            out[f"{metric}_v_mean"] = data.get("v_mean")
            out[f"{metric}_pop_mean"] = data.get("pop_mean")

    return out
