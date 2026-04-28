"""
fiedler_tracking.py — Track hemispheric structure across layers.

The Fiedler vector (second eigenvector of the graph Laplacian) defines
a bipartition of tokens into two hemispheres at each layer. This script
tracks that bipartition across depth.

If hemispheres are dynamically stable (as the metastability picture requires),
the Fiedler vector should be nearly constant during plateau windows and
rotate sharply at merge events.

Functions
---------
extract_fiedler_per_layer   : compute Fiedler vectors at every layer
hemisphere_assignments      : sign-partition of Fiedler vector per layer
hemisphere_crossing_rate    : fraction of tokens switching sides per layer transition
fiedler_stability           : cosine similarity of consecutive Fiedler vectors
hemisphere_centroid_separation : angle between hemisphere centroids
analyze_fiedler_tracking    : full pipeline
"""

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian, connected_components


# ---------------------------------------------------------------------------
# Fiedler vector extraction
# ---------------------------------------------------------------------------

def extract_fiedler_per_layer(activations: np.ndarray) -> dict:
    """
    Compute the Fiedler vector at every layer.

    Parameters
    ----------
    activations : (n_layers, n_tokens, d) — L2-normed

    Returns
    -------
    dict with:
      fiedler_vecs  : (n_layers, n_tokens) — the Fiedler vector at each layer
      fiedler_vals  : (n_layers,) — the Fiedler eigenvalue (algebraic connectivity)
      valid         : (n_layers,) bool — whether the computation succeeded
    """
    n_layers, n_tokens, d = activations.shape

    fiedler_vecs = np.zeros((n_layers, n_tokens))
    fiedler_vals = np.full(n_layers, np.nan)
    valid        = np.zeros(n_layers, dtype=bool)

    for L in range(n_layers):
        X     = activations[L]
        G     = X @ X.T
        G_pos = np.clip(G, 0, None)
        np.fill_diagonal(G_pos, 1.0)

        if n_tokens < 3:
            continue

        try:
            Lap = laplacian(G_pos, normed=True)
            k = min(3, n_tokens - 1)
            vals, vecs = eigh(Lap, subset_by_index=[0, k - 1])  # Lap computed above
            vals = np.real(vals)
            vecs = np.real(vecs)

            if float(vals[1]) < 1e-8:
                from scipy.sparse.csgraph import connected_components
                n_comps, comp_labels = connected_components(
                    csgraph=(G_pos > 1e-9).astype(np.float64), directed=False
                )
                if n_comps == 2:
                    vec = np.where(comp_labels == 0, 1.0, -1.0).astype(np.float64)
                    fiedler_vecs[L] = vec / np.linalg.norm(vec)
                    fiedler_vals[L] = float(vals[1])
                    valid[L] = True
                    continue
                var0 = float(np.var(vecs[:, 0]))
                var1 = float(np.var(vecs[:, 1]))
                fv   = vecs[:, 0] if var0 > var1 else vecs[:, 1]
            else:
                fv = vecs[:, 1]

            fiedler_vecs[L] = fv
            fiedler_vals[L] = float(vals[1])
            valid[L]        = True
        except Exception:
            continue

    return {"fiedler_vecs": fiedler_vecs, "fiedler_vals": fiedler_vals, "valid": valid}


# ---------------------------------------------------------------------------
# Hemisphere assignments
# ---------------------------------------------------------------------------

def hemisphere_assignments(fiedler_data: dict) -> dict:
    """
    Assign tokens to hemispheres based on the sign of the Fiedler vector.

    Positive Fiedler → hemisphere A, negative → hemisphere B.
    Tokens at exactly zero (rare) assigned to A.

    Returns
    -------
    dict with:
      assignments : (n_layers, n_tokens) int — 0 or 1
      sizes       : (n_layers, 2) int — number of tokens per hemisphere
    """
    vecs = fiedler_data["fiedler_vecs"]
    valid = fiedler_data["valid"]
    n_layers, n_tokens = vecs.shape

    assignments = np.zeros((n_layers, n_tokens), dtype=int)
    sizes = np.zeros((n_layers, 2), dtype=int)

    for L in range(n_layers):
        if not valid[L]:
            continue
        assignments[L] = (vecs[L] >= 0).astype(int)
        sizes[L, 0] = int((assignments[L] == 0).sum())
        sizes[L, 1] = int((assignments[L] == 1).sum())

    return {"assignments": assignments, "sizes": sizes}


# ---------------------------------------------------------------------------
# Crossing rate
# ---------------------------------------------------------------------------

def hemisphere_crossing_rate(assignments: np.ndarray, valid: np.ndarray) -> dict:
    """
    Fraction of tokens that switch hemispheres between consecutive layers.

    Near zero during plateaus (hemisphere assignments frozen).
    Spikes at merge events.

    Parameters
    ----------
    assignments : (n_layers, n_tokens) int — from hemisphere_assignments
    valid       : (n_layers,) bool

    Returns
    -------
    dict with:
      crossing_rate : (n_layers - 1,) float — fraction of tokens switching
      crossing_count: (n_layers - 1,) int — number of tokens switching
    """
    n_layers, n_tokens = assignments.shape
    rates = np.full(n_layers - 1, np.nan)
    counts = np.zeros(n_layers - 1, dtype=int)

    for L in range(n_layers - 1):
        if not valid[L] or not valid[L + 1]:
            continue
        switched = (assignments[L] != assignments[L + 1]).sum()
        # The Fiedler vector can flip globally (all signs reverse) without
        # any real structure change. Detect and correct: if >50% switched,
        # the Fiedler vector flipped and the actual switches are n - switched.
        if switched > n_tokens / 2:
            switched = n_tokens - switched
        counts[L] = int(switched)
        rates[L] = float(switched / n_tokens)

    return {
        "crossing_rate":  rates,
        "crossing_count": counts,
    }


# ---------------------------------------------------------------------------
# Fiedler stability
# ---------------------------------------------------------------------------

def fiedler_stability(fiedler_data: dict) -> dict:
    """
    Cosine similarity between Fiedler vectors at consecutive layers.

    Plateau = high similarity (Fiedler vector stable).
    Merge = low or negative similarity (Fiedler vector rotates).

    Uses absolute value of cosine to handle global sign flips.

    Returns
    -------
    dict with:
      fiedler_cosine : (n_layers - 1,) float — |cos(v_L, v_{L+1})|
    """
    vecs = fiedler_data["fiedler_vecs"]
    valid = fiedler_data["valid"]
    n_layers = vecs.shape[0]

    cosines = np.full(n_layers - 1, np.nan)

    for L in range(n_layers - 1):
        if not valid[L] or not valid[L + 1]:
            continue
        v1 = vecs[L]
        v2 = vecs[L + 1]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            continue
        cosines[L] = float(abs(np.dot(v1, v2) / (n1 * n2)))

    return {"fiedler_cosine": cosines}


# ---------------------------------------------------------------------------
# Hemisphere centroid separation
# ---------------------------------------------------------------------------

def hemisphere_centroid_separation(
    activations: np.ndarray,
    assignments: np.ndarray,
    valid: np.ndarray,
) -> dict:
    """
    Angle between the mean positions of the two hemisphere token sets.

    Perfect antipodal = π. Converging toward merger → angle decreases.

    Also checks whether the Fiedler vector is aligned with the line
    connecting the two centroids (it should be, if the spectral Laplacian
    is genuinely finding the dominant bipartition axis).

    Parameters
    ----------
    activations : (n_layers, n_tokens, d)
    assignments : (n_layers, n_tokens) int — 0 or 1
    valid       : (n_layers,) bool

    Returns
    -------
    dict with:
      centroid_angle    : (n_layers,) float — angle in radians between centroids
      centroid_cos      : (n_layers,) float — cosine between centroids
      fiedler_centroid_alignment : (n_layers,) float — cosine between Fiedler vec
                                   and the centroid difference direction
    """
    n_layers, n_tokens, d = activations.shape
    angles = np.full(n_layers, np.nan)
    cosines = np.full(n_layers, np.nan)

    for L in range(n_layers):
        if not valid[L]:
            continue

        mask_a = assignments[L] == 0
        mask_b = assignments[L] == 1

        if mask_a.sum() < 1 or mask_b.sum() < 1:
            continue

        centroid_a = activations[L][mask_a].mean(axis=0)
        centroid_b = activations[L][mask_b].mean(axis=0)

        na = np.linalg.norm(centroid_a)
        nb = np.linalg.norm(centroid_b)
        if na < 1e-10 or nb < 1e-10:
            continue

        cos_val = float(np.dot(centroid_a, centroid_b) / (na * nb))
        cos_val = np.clip(cos_val, -1.0, 1.0)
        cosines[L] = cos_val
        angles[L] = float(np.arccos(cos_val))

    return {
        "centroid_angle": angles,
        "centroid_cos":   cosines,
    }


# ---------------------------------------------------------------------------
# Cross-reference with Phase 1 events
# ---------------------------------------------------------------------------

def crossref_with_events(
    crossing_rate: np.ndarray,
    fiedler_cosine: np.ndarray,
    phase1_events: dict,
    beta: float = 1.0,
) -> dict:
    """
    Test whether Fiedler instabilities align with Phase 1's merge events
    and energy violations.

    Parameters
    ----------
    crossing_rate  : (n_layers - 1,) float
    fiedler_cosine : (n_layers - 1,) float
    phase1_events  : from trajectory.load_phase1_events
    beta           : which beta to use for violations

    Returns
    -------
    dict with:
      violations_with_crossing : fraction of violation layers that also have
                                 elevated crossing rate (> 2σ above mean)
      merge_fiedler_cosine     : mean Fiedler cosine at merge event layers
      plateau_fiedler_cosine   : mean Fiedler cosine at plateau layers
    """
    # Get violation and merge layers
    violations = set(phase1_events.get("energy_violations", {}).get(beta, []))

    # Crossing rate threshold: 2σ above mean
    valid_rates = crossing_rate[np.isfinite(crossing_rate)]
    if len(valid_rates) < 3:
        return {"error": "insufficient valid crossing rate data"}

    mean_rate = float(valid_rates.mean())
    std_rate = float(valid_rates.std())
    threshold = mean_rate + 2 * std_rate

    # Fraction of violations that have elevated crossing
    n_v_with_crossing = 0
    n_v_total = 0
    for v_layer in violations:
        idx = v_layer - 1   # crossing_rate[L] is the L→L+1 transition
        if 0 <= idx < len(crossing_rate) and np.isfinite(crossing_rate[idx]):
            n_v_total += 1
            if crossing_rate[idx] > threshold:
                n_v_with_crossing += 1

    # Fiedler cosine at violations vs non-violations
    v_cosines = []
    nv_cosines = []
    for L in range(len(fiedler_cosine)):
        if not np.isfinite(fiedler_cosine[L]):
            continue
        if (L + 1) in violations:  # L+1 because cosine[L] = similarity(L, L+1)
            v_cosines.append(fiedler_cosine[L])
        else:
            nv_cosines.append(fiedler_cosine[L])

    return {
        "violations_with_elevated_crossing": float(
            n_v_with_crossing / max(n_v_total, 1)
        ),
        "n_violations_tested":   n_v_total,
        "crossing_threshold":    threshold,
        "mean_crossing_rate":    mean_rate,
        "violation_fiedler_cos": float(np.mean(v_cosines)) if v_cosines else float("nan"),
        "non_violation_fiedler_cos": float(np.mean(nv_cosines)) if nv_cosines else float("nan"),
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def analyze_fiedler_tracking(
    activations: np.ndarray,
    phase1_events: dict = None,
    beta: float = 1.0,
) -> dict:
    """
    Full Block 2 Fiedler tracking analysis.

    Parameters
    ----------
    activations   : (n_layers, n_tokens, d) — L2-normed
    phase1_events : from trajectory.load_phase1_events (optional, for cross-ref)
    beta          : which beta for violation cross-referencing

    Returns
    -------
    dict with all sub-analysis results
    """
    fiedler = extract_fiedler_per_layer(activations)
    hemi = hemisphere_assignments(fiedler)
    crossing = hemisphere_crossing_rate(hemi["assignments"], fiedler["valid"])
    stability = fiedler_stability(fiedler)
    centroids = hemisphere_centroid_separation(
        activations, hemi["assignments"], fiedler["valid"]
    )

    result = {
        "fiedler_vals":     fiedler["fiedler_vals"],
        "hemisphere_sizes": hemi["sizes"],
        "crossing_rate":    crossing["crossing_rate"],
        "crossing_count":   crossing["crossing_count"],
        "fiedler_cosine":   stability["fiedler_cosine"],
        "centroid_angle":   centroids["centroid_angle"],
        "centroid_cos":     centroids["centroid_cos"],
        "n_layers":         activations.shape[0],
        "n_tokens":         activations.shape[1],
    }

    if phase1_events is not None:
        result["crossref"] = crossref_with_events(
            crossing["crossing_rate"],
            stability["fiedler_cosine"],
            phase1_events,
            beta=beta,
        )

    return result


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------

def fiedler_to_json(result: dict) -> dict:
    """Extract JSON-serializable summary."""
    n = result["n_layers"]

    # Compute summary statistics
    cr = result["crossing_rate"]
    fc = result["fiedler_cosine"]
    ca = result["centroid_angle"]

    valid_cr = cr[np.isfinite(cr)]
    valid_fc = fc[np.isfinite(fc)]
    valid_ca = ca[np.isfinite(ca)]

    out = {
        "n_layers": n,
        "n_tokens": result["n_tokens"],
        "crossing_rate_mean":  float(valid_cr.mean()) if len(valid_cr) else None,
        "crossing_rate_max":   float(valid_cr.max()) if len(valid_cr) else None,
        "crossing_rate_std":   float(valid_cr.std()) if len(valid_cr) else None,
        "fiedler_cosine_mean": float(valid_fc.mean()) if len(valid_fc) else None,
        "fiedler_cosine_min":  float(valid_fc.min()) if len(valid_fc) else None,
        "centroid_angle_mean": float(valid_ca.mean()) if len(valid_ca) else None,
        "centroid_angle_near_pi": float((valid_ca > 2.8).mean()) if len(valid_ca) else None,
    }

    if "crossref" in result:
        out["crossref"] = result["crossref"]

    return out
