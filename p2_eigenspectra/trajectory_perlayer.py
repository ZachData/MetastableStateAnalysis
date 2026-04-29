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
rescaled_trajectory_perlayer   : exact cumulative e^{-V_L} rescaling (fix 9)
analyze_trajectory_offline_perlayer : full pipeline
"""

import numpy as np
from pathlib import Path
from scipy.linalg import expm

from core.config import BETA_VALUES
from p2_eigenspectra.trajectory import (
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
# Fix 9: Exact cumulative per-layer rescaling
# ---------------------------------------------------------------------------

def rescaled_trajectory_perlayer(
    activations: np.ndarray,
    ov_list: list,
    beta_values: list = None,
) -> dict:
    """
    Exact cumulative per-layer rescaling: z_i(L) = x_i(L) @ R_L

    where R_L = expm(-OV_0) @ expm(-OV_1) @ ... @ expm(-OV_{L-1})
    (row-vector convention: the L-th layer's rescaling matrix is the
    cumulative product of all previous layers' e^{-OV}).

    Motivation
    ----------
    The original implementation used a single mean-V matrix for all layers,
    which can over-correct (eliminating violations that are not V-mediated)
    or under-correct (missing the layer-specific repulsive contribution).

    The exact version applies each layer's own e^{-OV_l} contribution
    incrementally, making the rescaling sensitive to depth-dependent
    variation in V's repulsive strength.

    Comparison with mean-V
    ----------------------
    The returned dict includes violation counts from both methods so callers
    can quantify the approximation error.  If exact_perlayer gives the same
    elimination rate as mean_v, the approximation is valid.  If exact gives
    fewer eliminations, mean-V was over-correcting.

    Parameters
    ----------
    activations : (n_layers, n_tokens, d_model) — L2-normed
    ov_list     : list of n_layers (d_model, d_model) ndarrays — per-layer OV
    beta_values : list of floats for energy computation

    Returns
    -------
    dict with keys matching rescaled_trajectory output, plus:
      method    : "exact_perlayer"

    Note: the comparison against the mean-V approximation is assembled by
    the caller (analyze_trajectory_offline_perlayer) and attached to the
    returned dict under "comparison_with_meanv" after this function returns.
    """
    from scipy.linalg import svdvals as _svdvals

    if beta_values is None:
        beta_values = BETA_VALUES

    n_layers, n_tokens, d = activations.shape
    n_ov = len(ov_list)

    # Pre-compute per-layer matrix exponentials in float64.
    # expm is expensive — O(d^3) per layer.  GPT-2-xl has 48 layers × 1600²,
    # which takes ~30s on CPU.  Cache the results to avoid recomputation.
    print(" computing expm per layer...", end="", flush=True)
    expm_list = []
    for l in range(min(n_layers, n_ov)):
        expm_list.append(expm(-ov_list[l].astype(np.float64)))
    # If ov_list is shorter than n_layers, repeat the last matrix
    while len(expm_list) < n_layers:
        expm_list.append(expm_list[-1])

    # Build cumulative rescaling matrices: R_L = expm(-OV_0) @ ... @ expm(-OV_{L-1})
    # R_0 = I (no rescaling at the embedding layer)
    # R_{L+1} = R_L @ expm(-OV_L)
    R_cumulative = np.eye(d, dtype=np.float64)
    rescaled = np.zeros((n_layers, n_tokens, d), dtype=np.float64)
    max_valid_layer = 0

    for L in range(n_layers):
        # At layer L, apply cumulative rescaling R_L to activations[L]
        raw = activations[L].astype(np.float64) @ R_cumulative.T   # (n, d)
        if not np.all(np.isfinite(raw)):
            break
        norms = np.linalg.norm(raw, axis=-1, keepdims=True)
        rescaled[L] = raw / np.maximum(norms, 1e-10)
        max_valid_layer = L + 1

        # Update: R_{L+1} = R_L @ expm(-OV_L)
        R_cumulative = R_cumulative @ expm_list[L]
        if not np.all(np.isfinite(R_cumulative)) or np.abs(R_cumulative).max() > 1e15:
            max_valid_layer = L + 1
            break

    rescaled = rescaled[:max_valid_layer].astype(np.float32)
    n_valid  = max_valid_layer

    # Recompute metrics on rescaled trajectory (same logic as rescaled_trajectory)
    ip_means = np.full(n_layers, float("nan"))
    ip_mass  = np.full(n_layers, float("nan"))
    energies = {beta: np.full(n_layers, float("nan")) for beta in beta_values}
    eff_rank = np.full(n_layers, float("nan"))

    for L in range(n_valid):
        X = rescaled[L]
        G = X @ X.T
        idx = np.triu_indices(n_tokens, k=1)
        ips = G[idx]
        ip_means[L] = float(ips.mean())
        ip_mass[L]  = float((ips > 0.9).mean())
        for beta in beta_values:
            exp_G = np.exp(beta * G)
            energies[beta][L] = float(exp_G.sum() / (2.0 * beta * n_tokens * n_tokens))
        sv = _svdvals(X)
        sv = sv[sv > 1e-10]
        if len(sv) > 0:
            sv_n = sv / sv.sum()
            entropy = -np.sum(sv_n * np.log(sv_n + 1e-12))
            eff_rank[L] = float(np.exp(entropy))

    n_violations = {}
    for beta in beta_values:
        E = energies[beta]
        count = 0
        for L in range(1, n_valid):
            if (np.isfinite(E[L]) and np.isfinite(E[L-1])
                    and E[L] - E[L-1] < -1e-6 and eff_rank[L] >= 3.0):
                count += 1
        n_violations[beta] = count

    return {
        "rescaled_activations": rescaled,
        "ip_mean":              ip_means,
        "ip_mass_near_1":       ip_mass,
        "energies":             energies,
        "n_violations":         n_violations,
        "effective_rank":       eff_rank,
        "n_valid_layers":       n_valid,
        "method":               "exact_perlayer",
    }


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

    Fix 9: runs both the exact cumulative per-layer rescaling and the
    mean-V approximation, storing both results for comparison.

    Parameters
    ----------
    run_dir : Phase 1 run directory
    ov_data : dict from weights.analyze_weights

    Returns
    -------
    dict with all trajectory analysis results, plus:
      rescaled_exact : result of rescaled_trajectory_perlayer (exact)
      rescaled       : result of rescaled_trajectory with mean V (approx)
    """
    if not ov_data["is_per_layer"]:
        # Shared weights — use the standard path
        from p2_eigenspectra.trajectory import analyze_trajectory_offline
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

    # Fix 9: Exact cumulative per-layer rescaling.
    # The mean-V approximation is also computed for comparison.  Both are
    # stored — analysis.py uses rescaled_exact when available.
    print(f"    Rescaled trajectory (exact per-layer)...", end="", flush=True)
    rescaled_exact = rescaled_trajectory_perlayer(activations, ov_list)
    print(" done.")

    print(f"    Rescaled trajectory (mean-V approx)...", end="", flush=True)
    mean_ov = np.mean(np.stack(ov_list), axis=0)
    rescaled_meanv = rescaled_trajectory(activations, mean_ov)
    print(" done.")

    # Log comparison so downstream code can quantify approximation error
    comparison = {}
    for beta in BETA_VALUES:
        exact_n = rescaled_exact["n_violations"].get(beta, 0)
        meanv_n = rescaled_meanv["n_violations"].get(beta, 0)
        orig_n  = len(events["energy_violations"].get(beta, []))
        comparison[f"beta_{beta}"] = {
            "original_violations":  orig_n,
            "exact_violations":     exact_n,
            "meanv_violations":     meanv_n,
            "exact_eliminated":     orig_n - exact_n,
            "meanv_eliminated":     orig_n - meanv_n,
            "approx_error_pct":     (
                abs(exact_n - meanv_n) / max(orig_n, 1) * 100.0
            ),
        }
    rescaled_exact["comparison_with_meanv"] = comparison

    # Primary rescaled is the exact version; mean-V kept for back-compat
    return {
        "events":          events,
        "steps":           steps,
        "subspace":        subspace,
        "self_int":        self_int,
        "disp":            disp,
        "centroids":       centroids,
        "rescaled":        rescaled_exact,      # primary (exact)
        "rescaled_meanv":  rescaled_meanv,      # secondary (approximation)
    }
