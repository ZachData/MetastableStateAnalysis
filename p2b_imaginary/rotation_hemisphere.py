"""
rotation_hemisphere.py — Test rotation planes against hemispheric structure.

Central hypothesis: dominant rotation planes rotate tokens WITHIN their
hemisphere (orthogonal to the Fiedler axis), preserving the bipartition.
At merge events, an across-hemisphere rotation plane activates.

This script tests three things:
  1. Plane-vs-Fiedler alignment: are top rotation planes within-hemisphere?
  2. Token trajectory projection: do tokens actually move along the Fiedler
     axis during plateaus vs merge events?
  3. Within-hemisphere displacement coherence: are tokens in the same
     hemisphere displaced in parallel (rigid rotation) or independently?

The token trajectory test (2) is PRIMARY — it measures the full nonlinear
dynamics including softmax. The linear plane alignment (1) is SECONDARY —
it characterizes V_eff's geometry but misses attention reweighting.

Functions
---------
plane_fiedler_alignment       : geometric alignment of rotation planes with Fiedler
token_fiedler_displacement    : per-token projection of Δx onto Fiedler axis
displacement_coherence        : within-hemisphere displacement parallelism
analyze_rotation_hemisphere   : full pipeline
"""

import numpy as np


# ---------------------------------------------------------------------------
# Plane-vs-Fiedler alignment (linear, secondary)
# ---------------------------------------------------------------------------

def plane_fiedler_alignment(
    plane_projectors,
    fiedler_vecs: np.ndarray,
    valid: np.ndarray,
    activations: np.ndarray = None,
) -> dict:
    """
    For each top-k rotation plane, measure alignment with the Fiedler axis.

    The Fiedler vector lives in token space (n_tokens,). Rotation planes
    live in activation space (d_model,). To compare them, we convert
    the Fiedler partition into an activation-space direction: the unit
    centroid difference between the two hemispheres.

    A plane is "within-hemisphere" if its spanning vectors are orthogonal
    to this axis — rotation doesn't change hemisphere assignment.
    A plane is "across-hemisphere" if it has large Fiedler-axis component.

    Metric: for each plane j with basis vectors (u1, u2), compute
      alignment_j = (u1 · f_axis)² + (u2 · f_axis)²
    where f_axis is the unit centroid difference in activation space.
    Range [0, 1]:
      0 = plane is orthogonal to Fiedler axis (within-hemisphere)
      1 = plane contains the Fiedler axis (across-hemisphere)

    Parameters
    ----------
    plane_projectors : from rotational_schur.build_rotation_plane_projectors
    fiedler_vecs     : (n_layers, n_tokens) — Fiedler vectors per layer
    valid            : (n_layers,) bool
    activations      : (n_layers, n_tokens, d) — needed to compute Fiedler axis
                       in activation space

    Returns
    -------
    dict with:
      per_layer : list of dicts, each containing:
                    plane_alignments — alignment of each top-k plane with Fiedler
                    mean_alignment   — mean over top-k planes
                    max_alignment    — max (the most across-hemisphere plane)
                    frac_within      — fraction of planes with alignment < 0.1
                    frac_across      — fraction with alignment > 0.3
      overall   : dict with means across valid layers
    """
    if activations is None:
        return {"per_layer": [], "overall": {"error": "activations required"}}

    # For shared models, plane_projectors is a single dict.
    # For per-layer, it's a list. Handle both.
    planes_list = plane_projectors if isinstance(plane_projectors, list) else [plane_projectors]
    n_layers = fiedler_vecs.shape[0]

    per_layer = []
    for L in range(n_layers):
        if not valid[L]:
            per_layer.append(None)
            continue

        f = fiedler_vecs[L]
        # Convert Fiedler partition to activation-space axis
        pos_mask = f >= 0
        neg_mask = ~pos_mask
        if pos_mask.sum() < 1 or neg_mask.sum() < 1:
            per_layer.append(None)
            continue

        X = activations[L]
        centroid_diff = X[pos_mask].mean(axis=0) - X[neg_mask].mean(axis=0)
        cd_norm = np.linalg.norm(centroid_diff)
        if cd_norm < 1e-10:
            per_layer.append(None)
            continue
        f_axis = centroid_diff / cd_norm  # unit vector in activation space (d,)

        # Select the rotation planes for this layer
        if len(planes_list) > 1:
            pp = planes_list[min(L, len(planes_list) - 1)]
        else:
            pp = planes_list[0]

        top_planes = pp["top_k_planes"]
        if not top_planes:
            per_layer.append(None)
            continue

        alignments = []
        for plane in top_planes:
            # plane is (d, 2)
            u1 = plane[:, 0]
            u2 = plane[:, 1]
            alignment = float((np.dot(u1, f_axis)) ** 2 + (np.dot(u2, f_axis)) ** 2)
            alignments.append(alignment)

        n_planes = len(alignments)
        per_layer.append({
            "plane_alignments": alignments,
            "mean_alignment":   float(np.mean(alignments)),
            "max_alignment":    float(np.max(alignments)),
            "frac_within":      float(sum(1 for a in alignments if a < 0.1) / n_planes),
            "frac_across":      float(sum(1 for a in alignments if a > 0.3) / n_planes),
        })

    # Overall: mean of per-layer means
    valid_means = [p["mean_alignment"] for p in per_layer if p is not None]
    valid_within = [p["frac_within"] for p in per_layer if p is not None]
    valid_across = [p["frac_across"] for p in per_layer if p is not None]

    overall = {
        "mean_alignment":  float(np.mean(valid_means)) if valid_means else float("nan"),
        "mean_frac_within": float(np.mean(valid_within)) if valid_within else float("nan"),
        "mean_frac_across": float(np.mean(valid_across)) if valid_across else float("nan"),
    }

    return {"per_layer": per_layer, "overall": overall}


# ---------------------------------------------------------------------------
# Token trajectory projection onto Fiedler axis (nonlinear, primary)
# ---------------------------------------------------------------------------

def token_fiedler_displacement(
    activations: np.ndarray,
    fiedler_vecs: np.ndarray,
    valid: np.ndarray,
) -> dict:
    """
    Project each token's layer-to-layer displacement onto the Fiedler axis.

    Tokens in a rigidly rotating hemisphere should have near-zero
    Fiedler-axis displacement during plateaus and a sharp displacement
    at merge events.

    Parameters
    ----------
    activations  : (n_layers, n_tokens, d)
    fiedler_vecs : (n_layers, n_tokens)
    valid        : (n_layers,) bool

    Returns
    -------
    dict with:
      fiedler_displacement : (n_layers - 1, n_tokens) float
                             — signed projection of Δx_i onto Fiedler axis
      abs_displacement     : (n_layers - 1,) float
                             — mean |projection| per transition
      displacement_std     : (n_layers - 1,) float
                             — std of |projection| per transition
    """
    n_layers, n_tokens, d = activations.shape

    fiedler_disp = np.full((n_layers - 1, n_tokens), np.nan)
    abs_disp = np.full(n_layers - 1, np.nan)
    disp_std = np.full(n_layers - 1, np.nan)

    for L in range(n_layers - 1):
        if not valid[L]:
            continue

        # Use the Fiedler vector at the current layer as the reference axis
        f = fiedler_vecs[L]
        f_norm = np.linalg.norm(f)
        if f_norm < 1e-10:
            continue

        # The Fiedler vector lives in token space (n_tokens,).
        # We need a direction in activation space (d,).
        # Compute the Fiedler axis in activation space:
        # the direction of maximum separation between hemispheres.
        # This is the centroid difference direction.
        X = activations[L]
        pos_mask = f >= 0
        neg_mask = ~pos_mask
        if pos_mask.sum() < 1 or neg_mask.sum() < 1:
            continue

        centroid_diff = X[pos_mask].mean(axis=0) - X[neg_mask].mean(axis=0)
        cd_norm = np.linalg.norm(centroid_diff)
        if cd_norm < 1e-10:
            continue
        fiedler_axis = centroid_diff / cd_norm  # unit vector in activation space

        # Displacement
        delta = activations[L + 1] - activations[L]   # (n_tokens, d)

        # Project each token's displacement onto the Fiedler axis
        projections = delta @ fiedler_axis              # (n_tokens,)
        fiedler_disp[L] = projections
        abs_disp[L] = float(np.abs(projections).mean())
        disp_std[L] = float(np.abs(projections).std())

    return {
        "fiedler_displacement": fiedler_disp,
        "abs_displacement":     abs_disp,
        "displacement_std":     disp_std,
    }


# ---------------------------------------------------------------------------
# Effective crossing ratio
# ---------------------------------------------------------------------------

def effective_crossing_ratio(
    plane_alignment: dict,
    token_displacement: dict,
) -> dict:
    """
    Compare linear plane-Fiedler alignment (V_eff geometry) against
    actual Fiedler-axis displacement (full nonlinear dynamics).

    If ratio > 1: softmax amplifies hemisphere-crossing beyond V_eff's
    linear structure.
    If ratio < 1: softmax suppresses hemisphere-crossing.

    Returns per-layer ratio and overall statistics.
    """
    per_layer_align = plane_alignment["per_layer"]
    abs_disp = token_displacement["abs_displacement"]

    n = min(len(per_layer_align), len(abs_disp))

    # Normalize: compare the rank-order, not raw magnitudes (different units)
    # Use Spearman correlation between linear alignment and actual displacement
    from scipy.stats import spearmanr

    linear_vals = []
    actual_vals = []
    for L in range(n):
        pl = per_layer_align[L]
        if pl is None or not np.isfinite(abs_disp[L]):
            continue
        linear_vals.append(pl["mean_alignment"])
        actual_vals.append(abs_disp[L])

    if len(linear_vals) < 5:
        return {
            "spearman_rho": float("nan"),
            "spearman_p":   float("nan"),
            "n_valid":      len(linear_vals),
        }

    rho, pval = spearmanr(linear_vals, actual_vals)

    return {
        "spearman_rho": float(rho),
        "spearman_p":   float(pval),
        "n_valid":      len(linear_vals),
    }


# ---------------------------------------------------------------------------
# Within-hemisphere displacement coherence
# ---------------------------------------------------------------------------

def displacement_coherence(
    activations: np.ndarray,
    assignments: np.ndarray,
    valid: np.ndarray,
    plane_projectors: dict = None,
) -> dict:
    """
    Test whether tokens in the same hemisphere are displaced in parallel.

    If a hemisphere undergoes rigid rotation, all its tokens' displacements
    should be parallel when projected into the dominant rotation plane.
    High pairwise cosine similarity of projected displacements = rigid rotation.
    Low similarity = independent or chaotic movement.

    Parameters
    ----------
    activations      : (n_layers, n_tokens, d)
    assignments      : (n_layers, n_tokens) int — hemisphere labels
    valid            : (n_layers,) bool
    plane_projectors : from rotational_schur (optional, for projection)
                       If None, uses full displacement without projection.

    Returns
    -------
    dict with:
      coherence_hemi_0 : (n_layers - 1,) float — mean pairwise cosine in hemisphere 0
      coherence_hemi_1 : (n_layers - 1,) float — mean pairwise cosine in hemisphere 1
      coherence_mean   : (n_layers - 1,) float — average of both hemispheres
    """
    n_layers, n_tokens, d = activations.shape

    coh_0 = np.full(n_layers - 1, np.nan)
    coh_1 = np.full(n_layers - 1, np.nan)
    coh_mean = np.full(n_layers - 1, np.nan)

    # Optional: projector onto combined rotation subspace
    P_rot = None
    if plane_projectors is not None:
        pp = plane_projectors if isinstance(plane_projectors, list) else [plane_projectors]

    for L in range(n_layers - 1):
        if not valid[L]:
            continue

        delta = activations[L + 1] - activations[L]  # (n_tokens, d)

        # Optionally project into rotation subspace
        if plane_projectors is not None:
            if isinstance(plane_projectors, list):
                idx = min(L, len(plane_projectors) - 1)
                P_rot = plane_projectors[idx]["combined_rotation"]
            else:
                P_rot = plane_projectors["combined_rotation"]
            delta = delta @ P_rot   # project

        coh_vals = []
        for hemi in [0, 1]:
            mask = assignments[L] == hemi
            n_h = mask.sum()
            if n_h < 2:
                continue

            D = delta[mask]                         # (n_h, d)
            norms = np.linalg.norm(D, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            D_unit = D / norms

            # Pairwise cosine similarity
            cos_matrix = D_unit @ D_unit.T          # (n_h, n_h)
            idx_upper = np.triu_indices(n_h, k=1)
            cos_vals = cos_matrix[idx_upper]

            mean_cos = float(np.abs(cos_vals).mean())  # absolute: direction may flip
            coh_vals.append(mean_cos)

            if hemi == 0:
                coh_0[L] = mean_cos
            else:
                coh_1[L] = mean_cos

        if coh_vals:
            coh_mean[L] = float(np.mean(coh_vals))

    return {
        "coherence_hemi_0": coh_0,
        "coherence_hemi_1": coh_1,
        "coherence_mean":   coh_mean,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def analyze_rotation_hemisphere(
    activations: np.ndarray,
    fiedler_data: dict,
    hemi_data: dict,
    plane_projectors,
) -> dict:
    """
    Full rotation-hemisphere analysis.

    Parameters
    ----------
    activations      : (n_layers, n_tokens, d)
    fiedler_data     : from fiedler_tracking.extract_fiedler_per_layer
    hemi_data        : from fiedler_tracking.hemisphere_assignments
    plane_projectors : from rotational_schur.build_rotation_plane_projectors
                       (single dict for shared, list for per-layer)

    Returns
    -------
    dict with all sub-analyses
    """
    valid = fiedler_data["valid"]
    fiedler_vecs = fiedler_data["fiedler_vecs"]
    assignments = hemi_data["assignments"]

    alignment = plane_fiedler_alignment(
        plane_projectors, fiedler_vecs, valid, activations,
    )
    displacement = token_fiedler_displacement(
        activations, fiedler_vecs, valid,
    )
    crossing_ratio = effective_crossing_ratio(alignment, displacement)
    coherence = displacement_coherence(
        activations, assignments, valid, plane_projectors,
    )

    return {
        "plane_alignment":  alignment,
        "token_displacement": displacement,
        "crossing_ratio":   crossing_ratio,
        "coherence":        coherence,
    }


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------

def rotation_hemisphere_to_json(result: dict) -> dict:
    """Extract JSON-serializable summary."""
    coh = result["coherence"]["coherence_mean"]
    valid_coh = coh[np.isfinite(coh)]

    return {
        "plane_fiedler_alignment": result["plane_alignment"]["overall"],
        "crossing_ratio": result["crossing_ratio"],
        "coherence_mean": float(valid_coh.mean()) if len(valid_coh) else None,
        "coherence_std":  float(valid_coh.std()) if len(valid_coh) else None,
    }
