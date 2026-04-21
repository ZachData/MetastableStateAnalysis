"""
trajectory.py — Offline Phase 2 analysis on saved Phase 1 activations.

No model loading required.  Operates on activations.npz, clusters.npz,
metrics.json (from Phase 1) and the weight decomposition (from weights.py).

Each function takes pre-loaded data and returns results dicts that
analysis.py cross-references with Phase 1 events.

Functions
---------
step_size_trajectory       : ‖Δx‖/‖x‖ per token per layer — overshoot test
subspace_activation        : fraction of variance in attractive/repulsive subspace
self_interaction_trajectory: x^T V_eff x per token per layer — local sign
displacement_projection    : project Δx onto V subspaces per layer
centroid_projection        : project Phase 1 centroids onto V subspaces
rescaled_trajectory        : apply (e^{-V})^L and recompute metrics
load_phase1_events         : extract violation/plateau/merge info from metrics.json
"""

import json
import numpy as np
from pathlib import Path

from core.config import BETA_VALUES
from phase2.weights import rescale_matrix


# ---------------------------------------------------------------------------
# Load Phase 1 events
# ---------------------------------------------------------------------------

def load_phase1_events(run_dir: Path) -> dict:
    """
    Extract structured event info from a Phase 1 metrics.json.

    Returns
    -------
    dict with:
      n_layers         : int
      n_tokens         : int
      d_model          : int
      tokens           : list of str

      energies         : dict {beta: list of floats per layer}
      energy_violations: dict {beta: list of layer indices where E drops}
      energy_drop_pairs: dict {beta: {layer_idx: [(i, j, delta), ...]}}

      effective_rank   : list of floats per layer
      ip_mean          : list of floats per layer
      ip_mass_near_1   : list of floats per layer
      cka_prev         : list of floats per layer (nan for layer 0)

      spectral_k       : list of ints per layer
      kmeans_k         : list of ints per layer
    """
    run_dir = Path(run_dir)
    with open(run_dir / "metrics.json") as f:
        results = json.load(f)

    layers = results.get("layers", [])
    n_layers = len(layers)

    # Energy trajectories and violations
    energies          = {beta: [] for beta in BETA_VALUES}
    energy_violations = {beta: [] for beta in BETA_VALUES}
    energy_drop_pairs = {beta: {} for beta in BETA_VALUES}

    for layer in layers:
        layer_idx = layer["layer"]
        # Rehydrate float keys
        layer_energies = {float(k): v for k, v in layer.get("energies", {}).items()}
        layer_drops    = layer.get("energy_drop_pairs", {})
        if isinstance(layer_drops, list):
            layer_drops = {1.0: layer_drops} if layer_drops else {}
        else:
            layer_drops = {float(k): v for k, v in layer_drops.items()}

        for beta in BETA_VALUES:
            e = layer_energies.get(beta, float("nan"))
            energies[beta].append(e)

            # Check for violation: energy decreased vs previous layer
            if len(energies[beta]) >= 2:
                e_prev = energies[beta][-2]
                if (not _isnan(e) and not _isnan(e_prev)
                        and e - e_prev < -1e-6
                        and layer.get("effective_rank", 0) >= 3.0):
                    energy_violations[beta].append(layer_idx)

            pairs = layer_drops.get(beta, [])
            if pairs:
                energy_drop_pairs[beta][layer_idx] = pairs

    return {
        "n_layers":          n_layers,
        "n_tokens":          results.get("n_tokens", 0),
        "d_model":           results.get("d_model", 0),
        "tokens":            results.get("tokens", []),
        "model":             results.get("model", ""),
        "prompt":            results.get("prompt", ""),

        "energies":          energies,
        "energy_violations": energy_violations,
        "energy_drop_pairs": energy_drop_pairs,

        "effective_rank":    [l.get("effective_rank", 0) for l in layers],
        "ip_mean":           [l.get("ip_mean", 0) for l in layers],
        "ip_mass_near_1":    [l.get("ip_mass_near_1", 0) for l in layers],
        "cka_prev":          [l.get("cka_prev", float("nan")) for l in layers],

        "spectral_k":        [l.get("spectral", {}).get("k_eigengap", 1) for l in layers],
        "kmeans_k":          [l["clustering"]["kmeans"]["best_k"] for l in layers],
    }


def _isnan(v):
    return isinstance(v, float) and v != v


# ---------------------------------------------------------------------------
# Step-size trajectory (overshoot test)
# ---------------------------------------------------------------------------

def step_size_trajectory(activations: np.ndarray) -> dict:
    """
    Compute ‖x_{L+1} - x_L‖ / ‖x_L‖ per token per layer transition.

    A large step norm at a violation layer suggests the discrete update
    overshot the gradient-flow basin, producing an energy drop without
    any repulsive mechanism.

    Parameters
    ----------
    activations : (n_layers, n_tokens, d_model) float32 — L2-normed

    Returns
    -------
    dict with:
      step_norms      : (n_layers-1, n_tokens) float — per-token step norms
      step_mean       : (n_layers-1,) float — mean over tokens
      step_std        : (n_layers-1,) float — std over tokens
      global_mean     : float — mean over all layers and tokens
      global_std      : float — std over all layers and tokens
      overshoot_threshold : float — global_mean + 2 * global_std
    """
    n_layers = activations.shape[0]
    diffs    = activations[1:] - activations[:-1]             # (L-1, n, d)
    norms_x  = np.linalg.norm(activations[:-1], axis=-1)     # (L-1, n)
    norms_dx = np.linalg.norm(diffs, axis=-1)                # (L-1, n)

    # Avoid division by zero for collapsed tokens
    step_norms = norms_dx / np.maximum(norms_x, 1e-10)       # (L-1, n)

    step_mean   = step_norms.mean(axis=1)                     # (L-1,)
    step_std    = step_norms.std(axis=1)                      # (L-1,)
    global_mean = float(step_norms.mean())
    global_std  = float(step_norms.std())

    return {
        "step_norms":          step_norms,
        "step_mean":           step_mean,
        "step_std":            step_std,
        "global_mean":         global_mean,
        "global_std":          global_std,
        "overshoot_threshold": global_mean + 2.0 * global_std,
    }


# ---------------------------------------------------------------------------
# Subspace activation energy
# ---------------------------------------------------------------------------

def subspace_activation(
    activations: np.ndarray,
    projectors: dict,
) -> dict:
    """
    At each layer, measure how much token variance lives in the attractive
    vs repulsive eigensubspace of OV.

    For each projector P, the "subspace energy" is ‖X @ P‖_F^2 / ‖X‖_F^2
    where X is the (n_tokens, d) activation matrix at that layer.

    Two projector sets (Schur-based and symmetric-part).

    Parameters
    ----------
    activations : (n_layers, n_tokens, d_model)
    projectors  : dict from build_subspace_projectors

    Returns
    -------
    dict with per-layer arrays (n_layers,):
      schur_attract_frac, schur_repulse_frac
      sym_attract_frac, sym_repulse_frac
    """
    n_layers = activations.shape[0]

    result = {
        "schur_attract_frac": np.zeros(n_layers),
        "schur_repulse_frac": np.zeros(n_layers),
        "sym_attract_frac":   np.zeros(n_layers),
        "sym_repulse_frac":   np.zeros(n_layers),
    }

    for L in range(n_layers):
        X = activations[L]                              # (n, d)
        total_energy = np.sum(X ** 2)
        if total_energy < 1e-12:
            continue

        for key_prefix, proj_key in [
            ("schur_attract", "schur_attract"),
            ("schur_repulse", "schur_repulse"),
            ("sym_attract",   "sym_attract"),
            ("sym_repulse",   "sym_repulse"),
        ]:
            P = projectors[proj_key]
            projected = X @ P                           # (n, d)
            frac = np.sum(projected ** 2) / total_energy
            result[f"{key_prefix}_frac"][L] = frac

    return result


# ---------------------------------------------------------------------------
# Per-token self-interaction
# ---------------------------------------------------------------------------

def self_interaction_trajectory(
    activations: np.ndarray,
    OV: np.ndarray,
) -> dict:
    """
    Compute x_i^T V_eff x_i per token per layer.

    Positive = OV is locally attractive for this token.
    Negative = OV is locally repulsive.

    In row-vector convention: self_interaction[i] = x_i @ OV @ x_i^T

    Parameters
    ----------
    activations : (n_layers, n_tokens, d_model)
    OV          : (d_model, d_model) — composed OV matrix (row convention)

    Returns
    -------
    dict with:
      self_int     : (n_layers, n_tokens) float
      self_int_mean: (n_layers,) float
      self_int_std : (n_layers,) float
      frac_negative: (n_layers,) float — fraction of tokens with negative self-int
    """
    # X @ OV gives (n_layers, n_tokens, d_model)
    # Element-wise multiply with X and sum over d gives x @ OV @ x^T per token
    X_OV  = np.einsum("lnd,dk->lnk", activations, OV)   # (L, n, d)
    si    = np.einsum("lnd,lnd->ln", activations, X_OV)  # (L, n)

    return {
        "self_int":      si,
        "self_int_mean": si.mean(axis=1),
        "self_int_std":  si.std(axis=1),
        "frac_negative": (si < 0).mean(axis=1),
    }


# ---------------------------------------------------------------------------
# Displacement projection
# ---------------------------------------------------------------------------

def displacement_projection(
    activations: np.ndarray,
    projectors: dict,
) -> dict:
    """
    Project per-layer displacement Δx = x_{L+1} - x_L onto V subspaces.

    At violation layers, does the repulsive projection dominate?

    Parameters
    ----------
    activations : (n_layers, n_tokens, d_model)
    projectors  : dict from build_subspace_projectors

    Returns
    -------
    dict with per-transition arrays (n_layers-1,):
      schur_attract_disp_frac, schur_repulse_disp_frac
      sym_attract_disp_frac, sym_repulse_disp_frac
      total_disp_energy : (n_layers-1,) float — total ‖Δx‖_F^2 per transition
    """
    diffs    = activations[1:] - activations[:-1]       # (L-1, n, d)
    n_trans  = diffs.shape[0]

    result = {
        "schur_attract_disp_frac": np.zeros(n_trans),
        "schur_repulse_disp_frac": np.zeros(n_trans),
        "sym_attract_disp_frac":   np.zeros(n_trans),
        "sym_repulse_disp_frac":   np.zeros(n_trans),
        "total_disp_energy":       np.zeros(n_trans),
    }

    for t in range(n_trans):
        D = diffs[t]                                    # (n, d)
        total = np.sum(D ** 2)
        result["total_disp_energy"][t] = total
        if total < 1e-12:
            continue

        for key_prefix, proj_key in [
            ("schur_attract_disp", "schur_attract"),
            ("schur_repulse_disp", "schur_repulse"),
            ("sym_attract_disp",   "sym_attract"),
            ("sym_repulse_disp",   "sym_repulse"),
        ]:
            P = projectors[proj_key]
            projected = D @ P
            result[f"{key_prefix}_frac"][t] = np.sum(projected ** 2) / total

    return result


# ---------------------------------------------------------------------------
# Centroid projection
# ---------------------------------------------------------------------------

def centroid_projection(
    clusters_path: Path,
    projectors: dict,
) -> dict:
    """
    Project Phase 1 KMeans centroids onto V's eigensubspaces per layer.

    Centroids are already on S^{d-1} (from Phase 1's analysis.py).

    Parameters
    ----------
    clusters_path : path to clusters.npz
    projectors    : dict from build_subspace_projectors

    Returns
    -------
    dict with per-layer lists:
      centroid_attract_frac : list of floats — mean fraction per layer
      centroid_repulse_frac : list of floats
      centroid_self_int     : list of lists — per-centroid self-interaction
    """
    data = np.load(clusters_path)

    # Find layer indices from key names
    layer_indices = sorted(
        int(k.split("_L")[1])
        for k in data.files
        if k.startswith("kmeans_centroids_L")
    )

    attract_fracs = []
    repulse_fracs = []

    for i in layer_indices:
        key = f"kmeans_centroids_L{i}"
        if key not in data.files:
            attract_fracs.append(float("nan"))
            repulse_fracs.append(float("nan"))
            continue

        C = data[key]                              # (k, d)
        total = np.sum(C ** 2)
        if total < 1e-12:
            attract_fracs.append(float("nan"))
            repulse_fracs.append(float("nan"))
            continue

        Ca = C @ projectors["sym_attract"]
        Cr = C @ projectors["sym_repulse"]
        attract_fracs.append(float(np.sum(Ca ** 2) / total))
        repulse_fracs.append(float(np.sum(Cr ** 2) / total))

    return {
        "centroid_attract_frac": attract_fracs,
        "centroid_repulse_frac": repulse_fracs,
        "layer_indices":         layer_indices,
    }


# ---------------------------------------------------------------------------
# Rescaled trajectory
# ---------------------------------------------------------------------------

def rescaled_trajectory(
    activations: np.ndarray,
    OV: np.ndarray,
    beta_values: list = None,
) -> dict:
    """
    Apply the paper's Section 9 rescaling: z_i(L) = x_i(L) @ ((e^{-OV})^L)^T

    Then recompute Phase 1's core metrics on the rescaled trajectory.

    If metastability is sharper in the rescaled frame, OV is the right
    coordinate system.  If the energy monotonicity violation rate drops,
    the violations were caused by V's own dynamics rather than a failure
    of the gradient-flow model.

    Parameters
    ----------
    activations : (n_layers, n_tokens, d_model) — L2-normed
    OV          : (d_model, d_model) — composed OV in row-vector convention
    beta_values : list of floats for energy computation

    Returns
    -------
    dict with:
      rescaled_activations : (n_layers, n_tokens, d_model)
      ip_mean              : (n_layers,) float
      ip_mass_near_1       : (n_layers,) float
      energies             : {beta: (n_layers,) float}
      n_violations         : {beta: int}
      effective_rank       : (n_layers,) float — from rescaled (raw=normed here)
    """
    from scipy.linalg import svdvals as _svdvals

    if beta_values is None:
        beta_values = BETA_VALUES

    n_layers, n_tokens, d = activations.shape
    R     = rescale_matrix(OV).astype(np.float64)   # e^{-OV}, float64 for stability
    R_T   = R.T

    # Apply R incrementally.  R_pow = R^L can overflow when OV has negative
    # eigenvalues (making R's eigenvalues > 1).  Use float64 and detect
    # divergence — if R_pow overflows, truncate the rescaled trajectory.
    R_pow    = np.eye(d, dtype=np.float64)
    rescaled = np.zeros((n_layers, n_tokens, d), dtype=np.float64)
    max_valid_layer = 0

    for L in range(n_layers):
        raw = activations[L].astype(np.float64) @ R_pow.T    # (n, d)
        if not np.all(np.isfinite(raw)):
            break  # R_pow has diverged; stop here
        norms = np.linalg.norm(raw, axis=-1, keepdims=True)
        rescaled[L] = raw / np.maximum(norms, 1e-10)
        max_valid_layer = L + 1
        R_pow = R_pow @ R
        # Early exit if the matrix itself is diverging
        if np.abs(R_pow).max() > 1e15:
            max_valid_layer = L + 1
            break

    # Trim to valid layers
    rescaled = rescaled[:max_valid_layer].astype(np.float32)

    # Recompute metrics on rescaled trajectory
    n_valid    = max_valid_layer
    ip_means   = np.full(n_layers, float("nan"))
    ip_mass    = np.full(n_layers, float("nan"))
    energies   = {beta: np.full(n_layers, float("nan")) for beta in beta_values}
    eff_rank   = np.full(n_layers, float("nan"))

    for L in range(n_valid):
        X = rescaled[L]                                     # (n, d)
        G = X @ X.T                                         # (n, n)

        # Inner products (upper triangle)
        idx  = np.triu_indices(n_tokens, k=1)
        ips  = G[idx]
        ip_means[L] = float(ips.mean())
        ip_mass[L]  = float((ips > 0.9).mean())

        # Energies
        for beta in beta_values:
            exp_G = np.exp(beta * G)
            energies[beta][L] = float(exp_G.sum() / (2.0 * beta * n_tokens * n_tokens))

        # Effective rank (using rescaled as "raw" since we've already normed)
        sv = _svdvals(X)
        sv = sv[sv > 1e-10]
        if len(sv) > 0:
            sv_n = sv / sv.sum()
            entropy = -np.sum(sv_n * np.log(sv_n + 1e-12))
            eff_rank[L] = float(np.exp(entropy))

    # Count violations in rescaled frame
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
    }


# ---------------------------------------------------------------------------
# Full offline analysis pipeline
# ---------------------------------------------------------------------------

def analyze_trajectory_offline(
    run_dir: Path,
    ov_data: dict,
) -> dict:
    """
    Run the full offline Phase 2 trajectory analysis for a single Phase 1 run.

    Parameters
    ----------
    run_dir : path to a Phase 1 run directory (contains activations.npz,
              clusters.npz, metrics.json)
    ov_data : dict from weights.analyze_weights — must contain ov_total,
              projectors, decomps

    Returns
    -------
    dict with all trajectory analysis results, keyed by analysis type.
    """
    run_dir = Path(run_dir)

    # Load Phase 1 data
    act_data    = np.load(run_dir / "activations.npz")
    activations = act_data["activations"]                # (n_layers, n_tokens, d)
    events      = load_phase1_events(run_dir)

    # Resolve OV and projectors (handle shared vs per-layer)
    if ov_data["is_per_layer"]:
        # Per-layer models must go through analyze_trajectory_offline_perlayer,
        # which uses each layer's own projectors and OV.  Reaching this branch
        # means the caller bypassed that path — fail loudly rather than silently
        # using layer 0's projectors everywhere (the original bug this fixes).
        raise ValueError(
            "analyze_trajectory_offline called on a per-layer model "
            f"({ov_data.get('layer_names', ['?'])[0]}...).  "
            "Use analyze_trajectory_offline_perlayer instead."
        )
    else:
        OV         = ov_data["ov_total"]
        projectors = ov_data["projectors"]

    print(f"    Step sizes...", end="", flush=True)
    steps = step_size_trajectory(activations)
    print(" subspace activation...", end="", flush=True)
    subspace = subspace_activation(activations, projectors)
    print(" self-interaction...", end="", flush=True)
    self_int = self_interaction_trajectory(activations, OV)
    print(" displacement...", end="", flush=True)
    disp = displacement_projection(activations, projectors)
    print(" done.")

    # Centroid projection (optional — clusters.npz may not exist)
    centroids = None
    clusters_path = run_dir / "clusters.npz"
    if clusters_path.exists():
        print(f"    Centroid projection...", end="", flush=True)
        centroids = centroid_projection(clusters_path, projectors)
        print(" done.")

    # Rescaled trajectory
    print(f"    Rescaled trajectory...", end="", flush=True)
    rescaled = rescaled_trajectory(activations, OV)
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
