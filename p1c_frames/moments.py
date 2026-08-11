"""
p1c_frames/moments.py — sub-experiment C: the cumulant ladder, the moment
identity, and what "effective rank" actually measured.

TWO QUESTIONS, BOTH SETTLED FROM ARTIFACTS ALREADY ON DISK.

1. IS THE FOUR-BETA ENERGY SWEEP REDUNDANT? E_beta is the moment generating
   function of the pairwise-cosine distribution, so reporting it at four
   betas is a reparameterization of that distribution's first few moments.
   The non-redundant version is the cumulant ladder kappa_1 (common mode),
   kappa_2 (spread), kappa_3 (asymmetry), with 1/PR = kappa_2 + kappa_1^2.
   Verify the reconstruction against measured E_beta and report the
   residual per beta.

2. WAS THE RANK COLLAPSE A SINK COUNT? Raw effective rank is 1/<s^2> with
   norm-squared weights, and in the near-orthogonal limit it degenerates to
   (sum n_i^2)^2 / sum n_i^4 — the participation ratio of the NORM
   distribution alone, carrying zero directional content. status-1's
   "MinRank -> 2.3 by step 143000" is therefore not yet a geometric claim.
   Putting raw rank, normed rank, PR, and the norm-PR on the same axes
   settles it: if raw rank tracks norm-PR, the collapse is sinks.

COST: [R]. Reads geometry.json / energies.json, or activations.npz for the
norm-weighted quantities.

A NOTE ON WHICH INPUT TO USE. The cumulant ladder can be recovered two
ways, and they are NOT equally good:

  from the Gram        exact, needs activations.npz
  from ip_histogram    [R]-cheapest, but the histogram is OFF-DIAGONAL and
                       the energy identity is over the full n^2 Gram
                       INCLUDING the unit diagonal. The conversion
                       <G^k>_full = [1 + (n-1)<G^k>_off] / n is exact but
                       O(1/n): at n = 20 the naive off-diagonal kappa_1
                       reads +0.0030 against a true +0.0523, an order of
                       magnitude. core.metrics.cumulants_from_ip_histogram
                       applies the conversion; nothing else should read the
                       histogram moments directly.
"""

from __future__ import annotations

import numpy as np

from core.metrics import (
    cumulants_from_ip_histogram, energy_from_cumulants, gram_cumulants,
    norm_participation_ratio, effective_rank,
)


# ---------------------------------------------------------------------------
# The energy reconstruction check
# ---------------------------------------------------------------------------

def verify_moment_identity(cumulants: dict, energies: dict) -> dict:
    """
    Reconstruct E_beta from the cumulant ladder and report the residual at
    each measured beta.

    cumulants : {"kappa1", "kappa2", "kappa3"} for one layer
    energies  : {beta: measured E_beta} for the same layer

    The expected pattern, from the derivation's own range-of-validity
    check: relative error under 1% for beta <= 2 and roughly 25% at
    beta = 5, where the truncation fails because the exponential is
    dominated by the right tail of the cosine distribution — twelve moments
    are needed there, not three.

    So this function is NOT a pass/fail gate. It quantifies where the
    ladder can stand in for the energy sweep and where it cannot, which
    determines which columns the re-report can actually drop.
    """
    out = {}
    for beta, E in energies.items():
        b = float(beta)
        if not np.isfinite(E) or b <= 0:
            continue
        approx = energy_from_cumulants(cumulants["kappa1"], cumulants["kappa2"],
                                       cumulants["kappa3"], b)
        den = abs(E) if abs(E) > 1e-15 else np.nan
        out[b] = {
            "measured": float(E),
            "two_term": approx["two_term"],
            "three_term": approx["three_term"],
            "rel_err_two": float(abs(approx["two_term"] - E) / den),
            "rel_err_three": float(abs(approx["three_term"] - E) / den),
            # The ladder is a faithful stand-in only where this is small.
            "ladder_sufficient": bool(abs(approx["two_term"] - E) / den < 0.01),
        }
    return out


def ladder_from_layer(layer: dict, n_tokens: int) -> dict:
    """
    Cumulant ladder for one saved layer record, preferring the exact path.

    Order of preference:
      1. `gram_cumulants` if analysis_p1 wrote it (post-update runs)
      2. `ip_histogram` with the off-diagonal -> full conversion
    and the source is recorded, because the two are not interchangeable at
    small n and a mixed-provenance series would be silently inconsistent.
    """
    if layer.get("gram_cumulants"):
        c = dict(layer["gram_cumulants"])
        c["source"] = "gram_exact"
        return c
    hist = layer.get("ip_histogram")
    if hist:
        c = cumulants_from_ip_histogram(hist, n_tokens)
        c["source"] = "ip_histogram_converted"
        return c
    return {"kappa1": np.nan, "kappa2": np.nan, "kappa3": np.nan,
            "source": "unavailable"}


# ---------------------------------------------------------------------------
# The rank reconciliation
# ---------------------------------------------------------------------------

def rank_panel(raw_activations: np.ndarray) -> dict:
    """
    Four rank-like quantities for one layer, on the same axes.

    raw_activations : (n_tokens, d) RAW residual stream — the norms are the
                      whole point, so this must not be the unit-norm array.

        shannon_raw     entropy rank of the unnormalized cloud. Mixes
                        direction and scale. This is what status-1's
                        MinRank column reported.
        shannon_normed  entropy rank on the sphere. Direction only. The
                        frame-correct quantity.
        pr_rank         participation-ratio rank 1/<G^2> of the normed
                        Gram. Related to the energy expansion by
                        E ~ 1/(2b) + k1/2 + b/(4 PR).
        norm_pr         (sum n_i^2)^2 / sum n_i^4. ZERO directional
                        content by construction.

    `sink_fraction` is the diagnostic: shannon_raw / norm_pr. Near 1 means
    the raw rank is being set by the norm distribution and carries no
    geometric information; well above 1 means direction is doing the work.
    """
    X = np.asarray(raw_activations, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1)
    Xn = X / np.maximum(norms[:, None], 1e-12)
    G = Xn @ Xn.T

    sr = effective_rank(X, mode="raw")
    npr = norm_participation_ratio(X)
    return {
        "shannon_raw": float(sr),
        "shannon_normed": float(effective_rank(Xn, mode="normed")),
        "pr_rank": float(gram_cumulants(G)["pr_rank"]),
        "norm_pr": float(npr),
        "sink_ratio": float(sr / npr) if npr > 1e-12 else float("nan"),
        "norm_max_over_median": float(norms.max() / max(np.median(norms), 1e-12)),
        "n_tokens": int(X.shape[0]),
    }


def adjudicate_sink_hypothesis(panel_by_layer: list,
                               close_tol: float = 0.25) -> dict:
    """
    Does the reported rank collapse reduce to attention sinks?

    panel_by_layer : list of rank_panel outputs, one per layer

    The test is whether shannon_raw tracks norm_pr rather than
    shannon_normed. Reported as the fraction of layers where the two agree
    within `close_tol` relative, plus the correlation across layers of each
    pairing — a single layer can agree by coincidence, a whole depth
    profile cannot.
    """
    raw = np.array([p["shannon_raw"] for p in panel_by_layer])
    npr = np.array([p["norm_pr"] for p in panel_by_layer])
    nrm = np.array([p["shannon_normed"] for p in panel_by_layer])

    close = np.abs(raw - npr) / np.maximum(np.abs(npr), 1e-12) < close_tol
    def _corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a[m], b[m])[0, 1])

    c_norm_pr = _corr(raw, npr)
    c_normed = _corr(raw, nrm)

    if np.isfinite(c_norm_pr) and c_norm_pr > 0.9 and c_norm_pr > c_normed:
        verdict = ("SINKS — raw effective rank tracks the norm-participation "
                   "ratio, not the directional rank. status-1's rank-collapse "
                   "row is a statement about outlier token norms and must be "
                   "rewritten on the normed quantity.")
    elif np.isfinite(c_normed) and c_normed > 0.9 and c_normed > c_norm_pr:
        verdict = ("DIRECTIONAL — raw rank tracks the sphere rank, so the "
                   "collapse survives the frame correction. The original "
                   "claim stands, now on the frame-correct quantity.")
    else:
        verdict = ("MIXED — raw rank tracks neither cleanly. Both terms are "
                   "moving; report normed rank and norm-PR separately and "
                   "drop the raw column rather than interpreting it.")

    return {"verdict": verdict,
            "corr_raw_vs_norm_pr": c_norm_pr,
            "corr_raw_vs_normed": c_normed,
            "frac_layers_close_to_norm_pr": float(close.mean()),
            "min_shannon_raw": float(np.nanmin(raw)),
            "min_shannon_normed": float(np.nanmin(nrm)),
            "min_norm_pr": float(np.nanmin(npr))}
