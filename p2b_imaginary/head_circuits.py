"""
p2b_imaginary/head_circuits.py — the per-head OV circuit, kept factored.

WHY THIS REPLACES `ov_total`
----------------------------
`p2_eigenspectra/weights.py:184` builds `ov_total = sum(ov_per_head)`, and
every Phase 2b number so far is a statistic of that sum. But the operator
attention actually applies at layer l is

    o_i = sum_h sum_j alpha^h_ij  x_j  W_OV^h

with a DIFFERENT attention pattern per head. `sum_h W_OV^h` is the operator
only in the counterfactual where every head attends identically. It is not a
thing the model ever forms. So "84-97.5% of OV's spectral energy is
rotational" is a statistic of a fiction, and the fix is not a better estimator
of it — it is to work per head.

WHY PER-HEAD NEEDS THE CORE, NOT THE FULL SPACE
------------------------------------------------
`W_OV^h = W_O^h W_V^h` with `W_O^h: (d, d_head)` and `W_V^h: (d_head, d)`, so
its rank is at most `d_head` — 64 of 1024 at Pythia-410m. Embedded in `d`
dimensions it has `d - d_head` zero eigenvalues, and a Schur partition counts
every one of them as a real block.

WHICH statistics that corrupts is not obvious, and the answer is only half of
what it looks like:

  - DIMENSION fractions are destroyed. Measured on a 256/16 stand-in, the SAME
    head reads 5.5% of dimensions rotating in full space and 87.5% in its own
    core. `dim_complex_fraction` is not a per-head quantity unless it is
    computed in the core.
  - ENERGY fractions are NOT. `|0|^2 = 0`, so the zeros contribute nothing to
    the numerator and nothing to the denominator either. The published
    84-97.5% figure is an energy fraction, so it does not suffer from this and
    the two computations agree to 1e-6 (pinned in
    `TestRankArtifact.test_energy_fraction_is_rank_invariant`).
  - Frobenius ratios are not affected either, for the same reason.

The core is still the right object — it is the correct place to ask about
dimensions, and it is 4000x cheaper per head. But the rank argument alone does
not overturn the headline; the SHARED-ATTENTION argument above does.

The nonzero spectrum is exactly a `d_head x d_head` problem:

    eig(W_O W_V) \\ {0}  ==  eig(W_V W_O)                        [verified: 0.0]

`ov_head_core` in `weights.py` is already `W_V^h W_O^h`. It is computed and
then not persisted; `extract_head_cores` reconstructs it from the saved dense
per-head matrices when it is absent.

THE SYMMETRIC AND ANTISYMMETRIC PARTS STAY FACTORED TOO
--------------------------------------------------------
This is what makes the causal test affordable. With `k = d_head`:

    S_h = (W_OV + W_OV^T)/2 = B_S C   where B_S = [W_O,  W_V^T]/sqrt(2)
    A_h = (W_OV - W_OV^T)/2 = B_A C   where B_A = [W_O, -W_V^T]/sqrt(2)
                                            C   = [W_V ; W_O^T]/sqrt(2)

`B` is `(d, 2k)` and `C` is `(2k, d)`. Verified exact to 2e-17. So:

  - `rank(S_h) <= 2k` — 128 of 1024, not 1024.
  - Applying `S_h` to activations is `(Y @ B) @ C`: O(n*d*2k), never O(n*d^2),
    and no `(d, d)` array is ever materialised.
  - The nonzero spectrum of `S_h` is `eig(C B)`, a `2k x 2k` problem.

Cost, per layer, at Pythia-410m: `n_heads * d_head^3 = 16 * 64^3` against
`d_model^3 = 1024^3`. A 256x reduction, and the ratio grows with model size —
which is what makes pythia-1.4b (d=2048, d_head=128) reachable at all.

WHAT THIS MODULE DOES NOT DO
-----------------------------
It does not decide whether the per-head or the summed object is "the" answer.
Both are computed and reported side by side: the summed one for continuity
with the GPT-2 study and with the published 84-97.5% figure, the per-head one
because it is the operator that exists. Where they disagree, that disagreement
is the finding.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

#: Relative tolerance for calling a singular value zero when recovering
#: factors from a dense per-head matrix.
FACTOR_REL_TOL: float = 1e-10


# ---------------------------------------------------------------------------
# Factors
# ---------------------------------------------------------------------------

def factor_from_dense(OV_h: np.ndarray, d_head: Optional[int] = None,
                      rel_tol: float = FACTOR_REL_TOL) -> dict:
    """
    Recover `(W_O, W_V)` factors from a dense per-head `W_OV`.

    `weights.py` saves `ov_head{h}_{layer}` as a dense `(d, d)` array even
    though its rank is `d_head` — at 410m that is 16 heads x 24 layers x
    1024^2 x 4 bytes ~ 1.6 GB per checkpoint for an object that fits in
    1/16th of that. This reads the dense form back into factors so the rest of
    the module never touches `(d, d)` again; a future `weights.py` that saves
    `W_O` and `W_V` directly makes this function unnecessary.

    The factorisation is not unique — any `W_O M`, `M^{-1} W_V` works. The SVD
    choice is taken because it is the one that makes `rank` well defined and
    the factors well conditioned. Nothing downstream depends on which basis is
    chosen: `S`, `A`, and every spectrum computed here are invariant.

    Returns dict(W_O, W_V, rank, d, singular_values).
    """
    OV_h = np.asarray(OV_h, dtype=np.float64)
    d = OV_h.shape[0]
    U, s, Vt = np.linalg.svd(OV_h, full_matrices=False)

    cut = (rel_tol * s[0]) if s.size and s[0] > 0 else 0.0
    r = int(np.sum(s > cut))
    if d_head is not None:
        # Trust the architecture over the numerics: a head whose numerical
        # rank came out below d_head is a degenerate head, not a redefinition
        # of the head width, and padding keeps the factor shapes uniform.
        r = min(int(d_head), r) if r else 0
        r = int(d_head)

    W_O = U[:, :r] * s[:r]
    W_V = Vt[:r, :]
    return {
        "W_O": np.ascontiguousarray(W_O),
        "W_V": np.ascontiguousarray(W_V),
        "rank": int(np.sum(s > cut)),
        "d": int(d),
        "d_head": int(r),
        "singular_values": s,
    }


def head_core(W_O: np.ndarray, W_V: np.ndarray) -> np.ndarray:
    """
    `W_V W_O`, the `(d_head, d_head)` matrix carrying the head's entire
    nonzero spectrum.

    `eig(W_O W_V) \\ {0} == eig(W_V W_O)` exactly. This is `weights.py`'s
    `ov_head_core`, which that module computes and does not save.
    """
    return np.asarray(W_V, dtype=np.float64) @ np.asarray(W_O, dtype=np.float64)


# ---------------------------------------------------------------------------
# S / A in factored form
# ---------------------------------------------------------------------------

def sym_antisym_factors(W_O: np.ndarray, W_V: np.ndarray) -> dict:
    """
    Exact `(d, 2k)` x `(2k, d)` factors for the symmetric and antisymmetric
    parts of `W_OV = W_O W_V`.

        S = B_S C,  B_S = [W_O,  W_V^T]/sqrt(2)
        A = B_A C,  B_A = [W_O, -W_V^T]/sqrt(2)
                    C   = [W_V ; W_O^T]/sqrt(2)

    Both share `C`, which is why an intervention that swaps `S` for `V` costs
    one extra `(n, 2k)` matmul rather than a second projection.

    Returns dict(B_S, B_A, C, d, k, rank_bound).
    """
    W_O = np.asarray(W_O, dtype=np.float64)
    W_V = np.asarray(W_V, dtype=np.float64)
    r2 = np.sqrt(2.0)
    return {
        "B_S": np.concatenate([W_O, W_V.T], axis=1) / r2,
        "B_A": np.concatenate([W_O, -W_V.T], axis=1) / r2,
        "C": np.concatenate([W_V, W_O.T], axis=0) / r2,
        "d": int(W_O.shape[0]),
        "k": int(W_O.shape[1]),
        "rank_bound": int(2 * W_O.shape[1]),
    }


def apply_factored(Y: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    `Y @ (B C)` computed as `(Y @ B) @ C`.

    O(n*d*2k) rather than O(n*d^2), and the `(d, d)` product is never formed.
    At 410m that is 128 columns instead of 1024 — the difference between an
    intervention that runs and one that does not.
    """
    return (np.asarray(Y, dtype=np.float64) @ np.asarray(B, dtype=np.float64)) \
        @ np.asarray(C, dtype=np.float64)


def factored_spectrum(B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Nonzero eigenvalues of `B C` via `eig(C B)`, a `2k x 2k` problem.

    The zeros are omitted deliberately. Including them is what makes a
    per-head complex fraction a statement about the null space rather than
    about the head: measured on a 256/16 stand-in, the same head reads 5.5%
    rotating in full space and 87.5% in its core.
    """
    return np.linalg.eigvals(
        np.asarray(C, dtype=np.float64) @ np.asarray(B, dtype=np.float64))


# ---------------------------------------------------------------------------
# Per-head spectral statistics
# ---------------------------------------------------------------------------

def head_spectrum(W_O: np.ndarray, W_V: np.ndarray,
                  rel_tol: float = 0.01) -> dict:
    """
    Spectral statistics for one head, computed in its own core.

    Three quantities that were previously one number called "rotational
    fraction":

      `complex_energy_fraction_core`
          |lambda|^2 in complex pairs over the head's NONZERO spectrum. The
          per-head answer, and rank-invariant — the same value the ambient
          computation gives, since |0|^2 = 0.
      `dim_complex_fraction_core` vs `dim_complex_fraction_ambient`
          How many dimensions rotate, in the core and in `d`. These DO differ,
          by the rank ratio, and only the core one is a property of the head.
      `rotational_frobenius_fraction`
          ||A||_F^2 / ||W_OV||_F^2, from the factored parts. A norm question,
          not a spectral one, and it does not depend on the rank at all.

    `rel_tol` is the `|Im| > tol*(|Re| + eps)` criterion — the tolerance-
    sensitive definition `core/precision_policy.py` item P2 is about. It is
    used here only for `n_complex_relative`; the energy fractions use the
    exact conjugate-pair structure of the eigenvalues.
    """
    W_O = np.asarray(W_O, dtype=np.float64)
    W_V = np.asarray(W_V, dtype=np.float64)
    d, k = W_O.shape

    core = head_core(W_O, W_V)
    eigs = np.linalg.eigvals(core)
    is_cx = np.abs(eigs.imag) > 0.0
    is_cx_rel = np.abs(eigs.imag) > rel_tol * (np.abs(eigs.real) + 1e-12)

    e_all = float(np.sum(np.abs(eigs) ** 2))
    e_cx = float(np.sum(np.abs(eigs[is_cx]) ** 2))

    fac = sym_antisym_factors(W_O, W_V)
    # ||B C||_F^2 = trace(C^T B^T B C) — computed through the factors.
    def _frob_sq(B, C):
        return float(np.sum((np.asarray(B).T @ np.asarray(B)) *
                            (np.asarray(C) @ np.asarray(C).T)))
    s_frob = _frob_sq(fac["B_S"], fac["C"])
    a_frob = _frob_sq(fac["B_A"], fac["C"])
    v_frob = float(np.sum((W_O.T @ W_O) * (W_V @ W_V.T)))

    return {
        "d": int(d),
        "d_head": int(k),
        "n_eigs_core": int(eigs.size),
        "n_complex_pairs": int(is_cx.sum() // 2),
        "n_complex_relative": int(is_cx_rel.sum()),
        "complex_energy_fraction_core": float(e_cx / max(e_all, 1e-300)),
        # No `complex_energy_fraction_ambient` field: it would be the same
        # number. |0|^2 = 0, so the null space contributes nothing to the
        # numerator and nothing to the denominator, and the energy fraction is
        # rank-invariant. Only the DIMENSION fraction collapses, which is why
        # both of those are reported below and the energy one is not doubled.
        "dim_complex_fraction_core": float(is_cx.sum() / max(k, 1)),
        "dim_complex_fraction_ambient": float(is_cx.sum() / max(d, 1)),
        "eigenvalue_energy": e_all,
        "spectral_radius": float(np.max(np.abs(eigs))) if eigs.size else 0.0,
        "theta_mean": (float(np.mean(np.abs(np.angle(eigs[is_cx]))))
                       if is_cx.any() else float("nan")),
        "frac_repulsive_real_part": (float((eigs[is_cx].real < 0).mean())
                                     if is_cx.any() else float("nan")),
        "S_frob_sq": s_frob,
        "A_frob_sq": a_frob,
        "V_frob_sq": v_frob,
        "rotational_frobenius_fraction": float(a_frob / max(v_frob, 1e-300)),
    }


def layer_head_spectra(ov_per_head: Sequence[np.ndarray],
                       d_head: Optional[int] = None) -> dict:
    """
    `head_spectrum` for every head in a layer, plus the head-to-head spread.

    `spread` matters because the summed object hides it: sixteen heads with
    complex fractions from 0.1 to 0.9 sum to something whose spectrum reports
    a single middling number, and the summed object is the one Phase 2b has
    been measuring. If the spread is large, "OV is 84-97% rotational" was
    never a statement about any head.
    """
    per_head = []
    for OV_h in ov_per_head:
        f = factor_from_dense(OV_h, d_head=d_head)
        per_head.append(head_spectrum(f["W_O"], f["W_V"]))

    def col(key):
        return np.array([h[key] for h in per_head], dtype=np.float64)

    cef = col("complex_energy_fraction_core")
    return {
        "per_head": per_head,
        "n_heads": len(per_head),
        "complex_energy_fraction_mean": float(np.nanmean(cef)),
        "complex_energy_fraction_std": float(np.nanstd(cef)),
        "complex_energy_fraction_min": float(np.nanmin(cef)),
        "complex_energy_fraction_max": float(np.nanmax(cef)),
        "dim_complex_fraction_core_mean": float(np.nanmean(
            col("dim_complex_fraction_core"))),
        "rotational_frobenius_fraction_mean": float(np.nanmean(
            col("rotational_frobenius_fraction"))),
        "theta_mean": float(np.nanmean(col("theta_mean"))),
        "frac_repulsive_real_part_mean": float(np.nanmean(
            col("frac_repulsive_real_part"))),
        "spectral_radius_max": float(np.nanmax(col("spectral_radius"))),
    }


# ---------------------------------------------------------------------------
# The shared-attention counterfactual, made explicit
# ---------------------------------------------------------------------------

def summed_vs_per_head(ov_per_head: Sequence[np.ndarray],
                       ov_total: Optional[np.ndarray] = None,
                       d_head: Optional[int] = None) -> dict:
    """
    The summed statistic beside the per-head one, with the gap named.

    `sum_h W_OV^h` is the operator only under a counterfactual the model does
    not satisfy: that every head attends identically. This function does not
    adjudicate between them — it reports both and the disagreement, because
    the disagreement is what says how much of the 84-97.5% headline survives
    dropping that counterfactual.

    `head_agreement` is the fraction of heads whose core complex-energy
    fraction lies within 0.05 of the summed value. Low agreement means the
    summed number describes no head in the layer.
    """
    from p2b_imaginary.rotational_schur import (
        complex_energy_fraction, extract_schur_blocks,
    )

    per = layer_head_spectra(ov_per_head, d_head=d_head)
    if ov_total is None:
        ov_total = sum(np.asarray(M, dtype=np.float64) for M in ov_per_head)
    summed = complex_energy_fraction(extract_schur_blocks(ov_total))

    cef = np.array([h["complex_energy_fraction_core"] for h in per["per_head"]])
    s_val = summed["complex_energy_fraction"]

    return {
        "summed": {
            "complex_energy_fraction": s_val,
            "dim_complex_fraction": summed["dim_complex_fraction"],
            "n_complex": summed["n_complex"],
            "caveat": (
                "sum_h W_OV^h is the effective operator only if every head "
                "shares an attention pattern. It is retained for continuity "
                "with the GPT-2 study and with the published 84-97.5% figure."
            ),
        },
        "per_head": {
            "complex_energy_fraction_mean": per["complex_energy_fraction_mean"],
            "complex_energy_fraction_std": per["complex_energy_fraction_std"],
            "complex_energy_fraction_min": per["complex_energy_fraction_min"],
            "complex_energy_fraction_max": per["complex_energy_fraction_max"],
            "dim_complex_fraction_core_mean":
                per["dim_complex_fraction_core_mean"],
            "n_heads": per["n_heads"],
        },
        "gap": float(s_val - per["complex_energy_fraction_mean"]),
        "head_agreement": float(np.mean(np.abs(cef - s_val) < 0.05)),
        "head_spread": per["complex_energy_fraction_std"],
    }


__all__ = [
    "FACTOR_REL_TOL",
    "factor_from_dense",
    "head_core",
    "sym_antisym_factors",
    "apply_factored",
    "factored_spectrum",
    "head_spectrum",
    "layer_head_spectra",
    "summed_vs_per_head",
]
