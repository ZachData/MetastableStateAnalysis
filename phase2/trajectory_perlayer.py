"""
trajectory_perlayer.py — Per-layer-correct trajectory analysis.

Drop-in replacement for trajectory.py's analysis path when the model has
per-layer V matrices (GPT-2, BERT).  Uses each layer's own OV, projectors,
and eigendecomposition rather than layer 0's.

The shared-weight path (ALBERT) is unaffected — trajectory.py already
handles it correctly since there is only one V.

Functions
---------
subspace_activation_perlayer   : per-layer projectors
self_interaction_perlayer      : per-layer V
displacement_projection_perlayer : per-layer projectors
analyze_trajectory_offline_perlayer : full pipeline
"""

import numpy as np
from pathlib import Path

from core.config import BETA_VALUES
from phase2.trajectory import (
    step_size_trajectory,
    centroid_projection,
    rescaled_trajectory,
    load_phase1_events,
)


# ---------------------------------------------------------------------------
# Per-layer subspace activation
# ---------------------------------------------------------------------------

def subspace_activation_perlayer(
    activations: np.ndarray,
    projectors_list: list,
) -> dict:
    """
    At each layer, measure variance in attractive/repulsive subspace
    using that layer's own projectors.

    Parameters
    ----------
    activations    : (n_layers, n_tokens, d_model)
    projectors_list: list of dicts (one per layer), each from
                     build_subspace_projectors

    Returns
    -------
    dict with per-layer arrays (n_layers,):
      schur_attract_frac, schur_repulse_frac
      sym_attract_frac, sym_repulse_frac
    """
    n_layers = activations.shape[0]
    n_proj   = len(projectors_list)

    result = {
        "schur_attract_frac": np.zeros(n_layers),
        "schur_repulse_frac": np.zeros(n_layers),
        "sym_attract_frac":   np.zeros(n_layers),
        "sym_repulse_frac":   np.zeros(n_layers),
    }

    for L in range(n_layers):
        X = activations[L]
        total_energy = np.sum(X ** 2)
        if total_energy < 1e-12:
            continue

        # Use the layer's own projectors, clamped to available range
        proj_idx = min(L, n_proj - 1)
        proj = projectors_list[proj_idx]

        for key_prefix, proj_key in [
            ("schur_attract", "schur_attract"),
            ("schur_repulse", "schur_repulse"),
            ("sym_attract",   "sym_attract"),
            ("sym_repulse",   "sym_repulse"),
        ]:
            P = proj[proj_key]
            projected = X @ P
            frac = np.sum(projected ** 2) / total_energy
            result[f"{key_prefix}_frac"][L] = frac

    return result


# ---------------------------------------------------------------------------
# Per-layer self-interaction
# ---------------------------------------------------------------------------

def self_interaction_perlayer(
    activations: np.ndarray,
    ov_list: list,
) -> dict:
    """
    Compute x_i^T V_eff^{(L)} x_i using each layer's own V.

    Parameters
    ----------
    activations : (n_layers, n_tokens, d_model)
    ov_list     : list of (d_model, d_model) ndarrays, one per layer

    Returns
    -------
    dict with:
      self_int      : (n_layers, n_tokens) float
      self_int_mean : (n_layers,) float
      self_int_std  : (n_layers,) float
      frac_negative : (n_layers,) float
    """
    n_layers, n_tokens, d = activations.shape
    n_ov = len(ov_list)

    si = np.zeros((n_layers, n_tokens), dtype=np.float64)

    for L in range(n_layers):
        ov_idx = min(L, n_ov - 1)
        OV = ov_list[ov_idx]
        X = activations[L]                          # (n, d)
        X_OV = X @ OV                               # (n, d)
        si[L] = np.einsum("nd,nd->n", X, X_OV)     # (n,)

    return {
        "self_int":      si.astype(np.float32),
        "self_int_mean": si.mean(axis=1),
        "self_int_std":  si.std(axis=1),
        "frac_negative": (si < 0).mean(axis=1),
    }


# ---------------------------------------------------------------------------
# Per-layer displacement projection
# ---------------------------------------------------------------------------

def displacement_projection_perlayer(
    activations: np.ndarray,
    projectors_list: list,
) -> dict:
    """
    Project Δx = x_{L+1} - x_L onto V's subspaces using the projectors
    for the TARGET layer (L+1), since the displacement lands there.

    Parameters
    ----------
    activations    : (n_layers, n_tokens, d_model)
    projectors_list: list of dicts (one per layer)

    Returns
    -------
    dict with per-transition arrays (n_layers-1,)
    """
    diffs   = activations[1:] - activations[:-1]
    n_trans = diffs.shape[0]
    n_proj  = len(projectors_list)

    result = {
        "schur_attract_disp_frac": np.zeros(n_trans),
        "schur_repulse_disp_frac": np.zeros(n_trans),
        "sym_attract_disp_frac":   np.zeros(n_trans),
        "sym_repulse_disp_frac":   np.zeros(n_trans),
        "total_disp_energy":       np.zeros(n_trans),
    }

    for t in range(n_trans):
        D = diffs[t]
        total = np.sum(D ** 2)
        result["total_disp_energy"][t] = total
        if total < 1e-12:
            continue

        # Use target layer's projectors (transition t goes to layer t+1)
        proj_idx = min(t + 1, n_proj - 1)
        proj = projectors_list[proj_idx]

        for key_prefix, proj_key in [
            ("schur_attract_disp", "schur_attract"),
            ("schur_repulse_disp", "schur_repulse"),
            ("sym_attract_disp",   "sym_attract"),
            ("sym_repulse_disp",   "sym_repulse"),
        ]:
            P = proj[proj_key]
            projected = D @ P
            result[f"{key_prefix}_frac"][t] = np.sum(projected ** 2) / total

    return result


# ---------------------------------------------------------------------------
# Full per-layer offline pipeline
# ---------------------------------------------------------------------------

def analyze_trajectory_offline_perlayer(
    run_dir: Path,
    ov_data: dict,
) -> dict:
    """
    Run Phase 2 trajectory analysis with per-layer-correct projectors.

    For shared-weight models (ALBERT), falls through to the standard
    trajectory.analyze_trajectory_offline.  For per-layer models, uses
    each layer's own V and projectors.

    Parameters
    ----------
    run_dir : Phase 1 run directory
    ov_data : dict from weights.analyze_weights

    Returns
    -------
    dict with all trajectory analysis results
    """
    if not ov_data["is_per_layer"]:
        # Shared weights — use the standard path
        from phase2.trajectory import analyze_trajectory_offline
        return analyze_trajectory_offline(run_dir, ov_data)

    run_dir = Path(run_dir)

    act_data    = np.load(run_dir / "activations.npz")
    activations = act_data["activations"]
    events      = load_phase1_events(run_dir)

    ov_list     = ov_data["ov_total"]       # list of (d, d) per layer
    proj_list   = ov_data["projectors"]     # list of dicts per layer

    print(f"    [per-layer] Step sizes...", end="", flush=True)
    steps = step_size_trajectory(activations)

    print(" subspace...", end="", flush=True)
    subspace = subspace_activation_perlayer(activations, proj_list)

    print(" self-interaction...", end="", flush=True)
    self_int = self_interaction_perlayer(activations, ov_list)

    print(" displacement...", end="", flush=True)
    disp = displacement_projection_perlayer(activations, proj_list)
    print(" done.")

    # Centroid projection uses the middle layer's projectors as representative
    centroids = None
    clusters_path = run_dir / "clusters.npz"
    if clusters_path.exists():
        mid_idx = len(proj_list) // 2
        print(f"    Centroid projection (layer {mid_idx} projectors)...", end="", flush=True)
        centroids = centroid_projection(clusters_path, proj_list[mid_idx])
        print(" done.")

    # Rescaled trajectory: for per-layer models, use the mean V as approximation.
    # A proper per-layer rescaling would need cumulative products of e^{-V_L},
    # which is expensive and numerically fragile.  The mean-V version gives a
    # useful lower bound.
    print(f"    Rescaled trajectory (mean V)...", end="", flush=True)
    mean_ov = np.mean(np.stack(ov_list), axis=0)
    rescaled = rescaled_trajectory(activations, mean_ov)
    print(" done.")

    return {
        "events":    events,
        "steps":     steps,
        "subspace":  subspace,
        "self_int":  self_int,
        "disp":      disp,
        "centroids": centroids,
        "rescaled":  rescaled,
    }
