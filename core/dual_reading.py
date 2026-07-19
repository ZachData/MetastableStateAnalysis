"""
core/dual_reading.py — Dual-reading primitive (transition plan v2, core
analysis primitives, item 4 of 4).

Schema fixed in core/DESIGN_dual_reading.md BEFORE this file was written,
per the plan's own instruction ("Its output schema is written into
DESIGN.md before any implementation — this is the primitive most at risk
of becoming a god-function"). Read that file for the full rationale; this
module is deliberately thin — an orchestrator over existing primitives
(core.metrics.effective_rank, tuned_lens_cluster.frozen_head_decode), not
a new computation engine. It does not fit probes, fit LDA directions,
build projectors, or load models — every one of those is supplied by the
caller, already computed.

Verification note: `geometric_reading` and the numpy pieces of
`semantic_reading` (LDA projection, probe prediction) are pure numpy/plain
Python and ARE runtime-verified in this pass (see
tests/test_dual_reading.py). The frozen-head decode piece
(`tuned_lens_cluster.frozen_head_decode`) needs torch + a real model and is
NOT — same limitation as core/intervention.py, for the same reason (no
torch, no network, in the sandbox this was written in).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Geometric reading
# ---------------------------------------------------------------------------

def _squared_norm_frac(vector: np.ndarray, U: Optional[np.ndarray]) -> Optional[float]:
    """
    ||U^T v||^2 / ||v||^2, or None if U is absent (projectors dict doesn't
    have this key — e.g. no "U_S" computed for this layer).
    """
    if U is None or U.shape[1] == 0:
        return None
    denom = float(np.dot(vector, vector))
    if denom < 1e-12:
        return None
    projected = U.T @ vector
    return float(np.dot(projected, projected) / denom)


def effective_rank_contribution(
    population: np.ndarray,
    point_membership_mask: Optional[np.ndarray],
) -> float:
    """
    How much `population`'s effective rank drops when the point (or every
    member of the cluster) marked by `point_membership_mask` is removed.
    See core/DESIGN_dual_reading.md, "effective_rank_contribution — the
    one new definition" for the full justification.

    NaN if point_membership_mask is None (nothing to leave out), if it
    selects nothing, or if removing it would leave zero rows (effective
    rank of an empty set isn't defined).
    """
    from core.metrics import effective_rank

    if point_membership_mask is None:
        return float("nan")
    if not point_membership_mask.any():
        return 0.0  # leaving out nothing changes nothing, exactly

    remaining = population[~point_membership_mask]
    if remaining.shape[0] == 0:
        return float("nan")

    full_rank = effective_rank(population, mode="raw")
    remaining_rank = effective_rank(remaining, mode="raw")
    return float(full_rank - remaining_rank)


def geometric_reading(
    vector: np.ndarray,
    population: np.ndarray,
    projectors: dict,
    point_membership_mask: Optional[np.ndarray] = None,
) -> dict:
    """
    The geometric half of a dual reading. See core/DESIGN_dual_reading.md
    for the full output-schema spec.

    Parameters
    ----------
    vector       : (d,) — the point's own vector, or a cluster's centroid.
    population   : (n, d) — every token at the same (checkpoint, prompt,
        layer) as `vector`, used only for effective_rank_contribution.
    projectors   : dict with any of "U_pos", "U_neg", "U_S", "U_A" — same
        shape probe_subspace.py / eigenspace_degeneracy.py already consume.
        Missing keys give None for that field, not an error.
    point_membership_mask : (n,) bool, or None — see effective_rank_contribution.

    Returns
    -------
    dict: v_attractive_frac, v_repulsive_frac, real_frac, imag_frac,
    effective_rank_contribution.
    """
    return {
        "v_attractive_frac": _squared_norm_frac(vector, projectors.get("U_pos")),
        "v_repulsive_frac":  _squared_norm_frac(vector, projectors.get("U_neg")),
        "real_frac":         _squared_norm_frac(vector, projectors.get("U_S")),
        "imag_frac":         _squared_norm_frac(vector, projectors.get("U_A")),
        "effective_rank_contribution": effective_rank_contribution(
            population, point_membership_mask
        ),
    }


# ---------------------------------------------------------------------------
# Semantic reading
# ---------------------------------------------------------------------------

def semantic_reading(
    vector: np.ndarray,
    lda_direction: Optional[np.ndarray] = None,
    probe=None,
    model=None,
    tokenizer=None,
    top_k: int = 20,
) -> dict:
    """
    The semantic half of a dual reading. See core/DESIGN_dual_reading.md
    for the full output-schema spec.

    Parameters
    ----------
    vector        : (d,) — same vector geometric_reading was given.
    lda_direction : (d,) unit vector, or None. Supplied already fit
        (eigenspace_degeneracy.py's lda_direction) — this function only
        projects, never fits.
    probe         : a fitted classifier with .predict(X) taking (1, d), or
        None. Supplied already fit (probe_subspace.py) — never fit here.
    model, tokenizer : for frozen-head decode (tuned_lens_cluster.py).
        Both required together for decode_* fields to be populated;
        either missing -> those fields are None.

    Returns
    -------
    dict: decode_entropy, decode_top1_id, decode_top1_token,
    decode_top1_prob, decode_top_k, lda_projection, probe_predicted_label.
    """
    result = {
        "decode_entropy":        None,
        "decode_top1_id":        None,
        "decode_top1_token":     None,
        "decode_top1_prob":      None,
        "decode_top_k":          None,
        "lda_projection":        None,
        "probe_predicted_label": None,
    }

    if lda_direction is not None:
        result["lda_projection"] = float(np.dot(vector, lda_direction))

    if probe is not None:
        pred = probe.predict(vector.reshape(1, -1))
        result["probe_predicted_label"] = int(pred[0])

    if model is not None and tokenizer is not None:
        from tuned_lens_cluster import frozen_head_decode

        decoded = frozen_head_decode(vector, model, tokenizer, top_k=top_k)
        result["decode_entropy"] = decoded["entropy"]
        result["decode_top_k"] = decoded["top"]
        if decoded["top"]:
            top1 = decoded["top"][0]
            result["decode_top1_id"] = top1["id"]
            result["decode_top1_token"] = top1["token"]
            result["decode_top1_prob"] = top1["prob"]

    return result


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def dual_reading(
    vector: np.ndarray,
    population: np.ndarray,
    projectors: dict,
    point_membership_mask: Optional[np.ndarray] = None,
    lda_direction: Optional[np.ndarray] = None,
    probe=None,
    model=None,
    tokenizer=None,
    top_k: int = 20,
) -> dict:
    """
    Paired geometric + semantic reading for one point of interest (a
    token's own vector, or a cluster's centroid — see
    core/DESIGN_dual_reading.md, "What a 'point of interest' is"). Every
    input beyond `vector`/`population`/`projectors` is independently
    optional; a missing input degrades the corresponding output field(s)
    to None/NaN, never an exception.

    Returns
    -------
    {"geometric": {...}, "semantic": {...}} — see
    core/DESIGN_dual_reading.md for the full field-by-field spec.
    """
    return {
        "geometric": geometric_reading(
            vector, population, projectors, point_membership_mask
        ),
        "semantic": semantic_reading(
            vector, lda_direction, probe, model, tokenizer, top_k
        ),
    }


# ---------------------------------------------------------------------------
# Particle-table projection (see DESIGN doc, "Particle-table projection")
# ---------------------------------------------------------------------------

def to_particle_row_fields(reading: dict) -> dict:
    """
    Reduce a dual_reading() result to the scalar-only subset that fits
    core.particles.ParticleTable's columnar contract (one float per row;
    no strings, no nested lists — see core/DESIGN_dual_reading.md,
    "Particle-table projection" for exactly why decode_top1_token and
    decode_top_k are excluded here).

    Returns a flat dict ready to feed into ParticleTable.add_column-style
    usage: "v_attractive_proj" / "v_repulsive_proj" match the columns
    core/particles.py already reserves for this primitive; everything
    else is prefixed "extra__" to match that module's convention for
    columns beyond its fixed schema.
    """
    g = reading["geometric"]
    s = reading["semantic"]
    return {
        "v_attractive_proj":            g["v_attractive_frac"],
        "v_repulsive_proj":             g["v_repulsive_frac"],
        "extra__real_frac":             g["real_frac"],
        "extra__imag_frac":             g["imag_frac"],
        "extra__eff_rank_contribution": g["effective_rank_contribution"],
        "extra__decode_entropy":        s["decode_entropy"],
        "extra__decode_top1_id":        s["decode_top1_id"],
        "extra__decode_top1_prob":      s["decode_top1_prob"],
        "extra__lda_projection":        s["lda_projection"],
        "extra__probe_predicted_label": s["probe_predicted_label"],
    }
