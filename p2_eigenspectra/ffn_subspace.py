"""
ffn_subspace.py — Project FFN updates onto V's eigensubspaces.

The core question: when the FFN causes an energy drop, is it pushing
tokens into V's repulsive subspace (amplifying V's repulsive effect)
or doing something orthogonal to V's eigenstructure?

If FFN updates at violation layers land primarily in V's repulsive
subspace, the FFN is acting as V's intermediary — V shapes the geometry,
FFN executes the repulsive displacement.  If FFN updates are orthogonal
to V's subspaces, FFN is an independent mechanism.

Uses saved ffn_deltas.npz (from decompose.py) and projectors (from
weights.py).  No model loading required.

Functions
---------
project_ffn_onto_v_subspaces  : per-layer projection fractions
compare_violation_vs_population : z-score of FFN-repulsive fraction at violations
run_ffn_subspace_analysis     : full pipeline
"""

import numpy as np
from pathlib import Path

from p2_eigenspectra.trajectory import load_phase1_events


# ---------------------------------------------------------------------------
# Core projection
# ---------------------------------------------------------------------------

def project_ffn_onto_v_subspaces(
    ffn_deltas: np.ndarray,
    projectors_list: list,
    is_per_layer: bool,
) -> dict:
    """
    Project FFN residual updates onto V's attractive and repulsive subspaces.

    For per-layer models, uses each layer's own projectors (the FFN at
    layer L produces an update that lives in layer L's representation space).

    For shared-weight models, uses the single projector set.

    Parameters
    ----------
    ffn_deltas     : (n_layers, n_tokens, d_model) — FFN contribution per layer
    projectors_list: list of dicts (one per layer for per-layer models,
                     or single dict for shared), each from build_subspace_projectors.
                     Each dict has keys: sym_attract, sym_repulse (d_model, d_model)
    is_per_layer   : whether projectors vary by layer

    Returns
    -------
    dict with per-layer arrays (n_layers,):
      ffn_repulse_frac   : fraction of FFN update energy in repulsive subspace
      ffn_attract_frac   : fraction in attractive subspace
      ffn_residual_frac  : fraction in neither (rotation / complex)
      ffn_total_energy   : total ||FFN_delta||^2 per layer
    """
    n_layers = ffn_deltas.shape[0]

    result = {
        "ffn_repulse_frac":  np.zeros(n_layers),
        "ffn_attract_frac":  np.zeros(n_layers),
        "ffn_residual_frac": np.zeros(n_layers),
        "ffn_total_energy":  np.zeros(n_layers),
    }

    for L in range(n_layers):
        D = ffn_deltas[L]                          # (n_tokens, d_model)
        total = np.sum(D ** 2)
        result["ffn_total_energy"][L] = total

        if total < 1e-12:
            continue

        # Select projectors
        if is_per_layer:
            proj_idx = min(L, len(projectors_list) - 1)
            proj = projectors_list[proj_idx]
        else:
            proj = projectors_list if isinstance(projectors_list, dict) else projectors_list[0]

        P_rep = proj["sym_repulse"]
        P_att = proj["sym_attract"]

        rep_energy = np.sum((D @ P_rep) ** 2)
        att_energy = np.sum((D @ P_att) ** 2)

        result["ffn_repulse_frac"][L] = rep_energy / total
        result["ffn_attract_frac"][L] = att_energy / total
        result["ffn_residual_frac"][L] = max(0, 1.0 - (rep_energy + att_energy) / total)

    return result


# ---------------------------------------------------------------------------
# Violation vs population comparison
# ---------------------------------------------------------------------------

def compare_violation_vs_population(
    ffn_projection: dict,
    phase1_events: dict,
    beta: float = 1.0,
) -> dict:
    """
    Compare FFN subspace fractions at violation layers vs all other layers.

    If the FFN is amplifying V's repulsive effect, ffn_repulse_frac should
    be elevated at violation layers (positive z-score).

    Returns
    -------
    dict with z-scores for ffn_repulse_frac, ffn_attract_frac, ffn_total_energy
    """
    violations = set(phase1_events["energy_violations"].get(beta, []))
    n_layers = len(ffn_projection["ffn_repulse_frac"])

    # FFN deltas are indexed by layer (the FFN at layer L).
    # A violation at layer L means energy dropped from L-1 to L,
    # so the FFN at layer L-1 produced the update.  Use index L-1.
    violation_indices = {v - 1 for v in violations if 0 <= v - 1 < n_layers}

    result = {}
    for metric in ["ffn_repulse_frac", "ffn_attract_frac", "ffn_total_energy"]:
        values = ffn_projection[metric]

        v_indices   = sorted(violation_indices)
        pop_indices = [i for i in range(n_layers) if i not in violation_indices]

        if not v_indices or not pop_indices:
            result[metric] = {
                "z_score": float("nan"),
                "v_mean": float("nan"),
                "pop_mean": float("nan"),
            }
            continue

        v_vals   = values[v_indices]
        pop_vals = values[pop_indices]
        pop_std  = float(np.std(pop_vals))

        result[metric] = {
            "z_score":  float((np.mean(v_vals) - np.mean(pop_vals)) / (pop_std + 1e-12)),
            "v_mean":   float(np.mean(v_vals)),
            "pop_mean": float(np.mean(pop_vals)),
            "pop_std":  pop_std,
            "n_violations": len(v_indices),
            "n_population": len(pop_indices),
        }

    return result


# ---------------------------------------------------------------------------
# Per-violation detail
# ---------------------------------------------------------------------------

def per_violation_ffn_projection(
    ffn_deltas: np.ndarray,
    projectors_list: list,
    is_per_layer: bool,
    phase1_events: dict,
    beta: float = 1.0,
) -> list:
    """
    For each violation layer, compute the FFN update's projection onto
    V's subspaces and classify the FFN's role.

    Classification:
      - "amplifies_repulsive": ffn_repulse_frac > ffn_attract_frac
        AND ffn_repulse_frac > 0.5 (FFN is pushing into V's repulsive subspace)
      - "amplifies_attractive": ffn_attract_frac > ffn_repulse_frac
        AND ffn_attract_frac > 0.5 (FFN is pushing into V's attractive subspace)
      - "orthogonal": neither subspace captures > 50%
        (FFN operates outside V's eigenstructure)

    Returns
    -------
    list of dicts, one per violation
    """
    violations = phase1_events["energy_violations"].get(beta, [])
    n_layers = ffn_deltas.shape[0]

    results = []
    for v_layer in violations:
        t_idx = v_layer - 1
        if t_idx < 0 or t_idx >= n_layers:
            continue

        D = ffn_deltas[t_idx]
        total = np.sum(D ** 2)
        if total < 1e-12:
            continue

        if is_per_layer:
            proj_idx = min(t_idx, len(projectors_list) - 1)
            proj = projectors_list[proj_idx]
        else:
            proj = projectors_list if isinstance(projectors_list, dict) else projectors_list[0]

        rep_frac = np.sum((D @ proj["sym_repulse"]) ** 2) / total
        att_frac = np.sum((D @ proj["sym_attract"]) ** 2) / total

        if rep_frac > att_frac and rep_frac > 0.5:
            role = "amplifies_repulsive"
        elif att_frac > rep_frac and att_frac > 0.5:
            role = "amplifies_attractive"
        else:
            role = "orthogonal"

        results.append({
            "layer":          v_layer,
            "ffn_repulse_frac": float(rep_frac),
            "ffn_attract_frac": float(att_frac),
            "ffn_residual_frac": float(max(0, 1.0 - rep_frac - att_frac)),
            "ffn_total_norm":  float(np.sqrt(total)),
            "role":            role,
        })

    return results


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_ffn_subspace_analysis(
    ffn_dir: Path,
    ov_data: dict,
    phase1_run_dir: Path = None,
    beta: float = 1.0,
) -> dict:
    """
    Full FFN subspace analysis for one model × prompt.

    Loads saved ffn_deltas and Phase 1 events.  Uses projectors from ov_data.

    File loading strategy
    ---------------------
    After fix 2 (save_decomposed), two files exist per component:
      ffn_deltas_raw.npz   — unnormalised; use for energy magnitude analysis
      ffn_deltas_normed.npz — unit-normalised per token; use for direction/projection

    Backward compatibility: if the new files are absent, falls back to the old
    single ffn_deltas.npz (which was normed).  In that case ffn_total_energy
    z-scores are unreliable and a warning is recorded in the result.

    Parameters
    ----------
    ffn_dir        : directory containing ffn_deltas_{raw,normed}.npz
    ov_data        : from weights.analyze_weights
    phase1_run_dir : Phase 1 run directory containing metrics.json.
                     If None, uses ffn_dir.

    Returns
    -------
    dict with projection, z-scores, per-violation detail
    """
    ffn_dir = Path(ffn_dir)

    # --- Load raw deltas (for energy magnitude) ---
    raw_path   = ffn_dir / "ffn_deltas_raw.npz"
    normed_path = ffn_dir / "ffn_deltas_normed.npz"
    legacy_path = ffn_dir / "ffn_deltas.npz"

    energy_warning = None

    if raw_path.exists() and normed_path.exists():
        ffn_raw    = np.load(raw_path)["ffn_deltas"]    # for energy
        ffn_normed = np.load(normed_path)["ffn_deltas"]  # for projection
    elif legacy_path.exists():
        # Old single normed file.  Projection fracs are still valid; energy
        # z-scores are not (all norms ≈ 1 after per-token normalisation).
        ffn_raw    = np.load(legacy_path)["ffn_deltas"]
        ffn_normed = ffn_raw
        energy_warning = (
            "Loaded legacy ffn_deltas.npz (normed only). "
            "ffn_total_energy z-scores are unreliable. "
            "Re-run decompose.save_decomposed to generate raw + normed files."
        )
    else:
        return {"applicable": False,
                "reason": f"no ffn_deltas files found in {ffn_dir}"}

    events_dir = Path(phase1_run_dir) if phase1_run_dir is not None else ffn_dir
    events = load_phase1_events(events_dir)

    projectors   = ov_data["projectors"]
    is_per_layer = ov_data["is_per_layer"]
    # projectors is already either a single dict (shared-weight models) or a
    # list of dicts (per-layer models).  Pass it through directly; the
    # is_per_layer flag tells project_ffn_onto_v_subspaces which case it is.
    proj_list = projectors

    # Subspace projection fractions use normed deltas (direction only).
    projection = project_ffn_onto_v_subspaces(ffn_normed, proj_list, is_per_layer)

    # Overwrite ffn_total_energy using raw deltas so magnitudes are meaningful.
    n_layers = ffn_raw.shape[0]
    raw_energy = np.zeros(n_layers)
    for L in range(n_layers):
        raw_energy[L] = float(np.sum(ffn_raw[L] ** 2))
    projection["ffn_total_energy"] = raw_energy

    zscores = compare_violation_vs_population(projection, events, beta)
    per_violation = per_violation_ffn_projection(
        ffn_normed, proj_list, is_per_layer, events, beta
    )

    # Summary
    n = len(per_violation)
    summary = {}
    if n > 0:
        summary["n_violations"] = n
        summary["frac_amplifies_repulsive"] = sum(
            1 for v in per_violation if v["role"] == "amplifies_repulsive"
        ) / n
        summary["frac_amplifies_attractive"] = sum(
            1 for v in per_violation if v["role"] == "amplifies_attractive"
        ) / n
        summary["frac_orthogonal"] = sum(
            1 for v in per_violation if v["role"] == "orthogonal"
        ) / n
        summary["mean_ffn_repulse_frac"] = float(np.mean(
            [v["ffn_repulse_frac"] for v in per_violation]
        ))
        summary["mean_ffn_attract_frac"] = float(np.mean(
            [v["ffn_attract_frac"] for v in per_violation]
        ))

    result = {
        "applicable": True,
        "projection": {k: v.tolist() if hasattr(v, "tolist") else v
                       for k, v in projection.items()},
        "zscores": zscores,
        "per_violation": per_violation,
        "summary": summary,
    }
    if energy_warning:
        result["energy_warning"] = energy_warning
    return result


def print_ffn_subspace_summary(result: dict, model_name: str, prompt_key: str):
    """Print concise FFN subspace analysis summary.  Delegates to summary_lines."""
    if not result.get("applicable"):
        return
    print(f"\n  FFN subspace projection ({model_name} | {prompt_key}):")
    for line in ffn_subspace_summary_lines(result):
        print(f"    {line}")


def ffn_subspace_summary_lines(result: dict) -> list[str]:
    """
    Return LLM-ready plain-text lines summarising FFN subspace analysis.
 
    This is the text-generation body extracted from print_ffn_subspace_summary.
    The print function delegates to this so disk output and terminal output
    are always identical.
    """
    if not result.get("applicable"):
        return [f"ffn_subspace: not applicable — {result.get('reason', 'no ffn_deltas')}"]
 
    s = result.get("summary", {})
    if not s:
        return ["ffn_subspace: applicable but no summary produced"]
 
    L = []
    L.append(f"FFN subspace projection at {s['n_violations']} violation layers:")
    L.append(f"  Amplifies V-repulsive:   {s['frac_amplifies_repulsive']:.0%}")
    L.append(f"  Amplifies V-attractive:  {s['frac_amplifies_attractive']:.0%}")
    L.append(f"  Orthogonal to V:         {s['frac_orthogonal']:.0%}")
    L.append(f"  Mean FFN→repulsive frac: {s['mean_ffn_repulse_frac']:.3f}")
    L.append(f"  Mean FFN→attract frac:   {s['mean_ffn_attract_frac']:.3f}")
 
    zs = result.get("zscores", {})
    for metric in ("ffn_repulse_frac", "ffn_attract_frac"):
        if metric in zs:
            z = zs[metric]
            L.append(
                f"  z({metric}): {z['z_score']:+.2f}  "
                f"(viol={z['v_mean']:.3f}  pop={z['pop_mean']:.3f})"
            )
 
    per_v = result.get("per_violation", [])
    if per_v:
        L.append("  First 5 per-violation classifications:")
        for v in per_v[:5]:
            L.append(
                f"    L{v['layer']:3d}  rep={v['ffn_repulse_frac']:.3f}  "
                f"att={v['ffn_attract_frac']:.3f}  "
                f"resid={v.get('ffn_residual_frac', float('nan')):.3f}  → {v['role']}"
            )
 
    ew = result.get("energy_warning")
    if ew:
        L.append(f"  Warning: {ew}")
 
    return L