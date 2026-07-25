"""
qk_offset_null.py — Offset-resolved QK antisymmetry and the P6-I2b nulls.

Sibling to qk_decompose.py, not a replacement. qk_decompose's
`decompose_qk_matrix` and `logit_partition` stay untouched so the frozen
GPT-2 reference remains bit-identical; this module holds everything that
only makes sense once the bilinear depends on relative offset.

What changed and why
--------------------
On Pythia the attention bilinear is M(Δ) = W_Q R(Δ) W_K^T, not W_Q W_K^T.
Three consequences kill the original P6-I2 test (see the 2026-07-22
PREDICTIONS addendum):

  1. a_frac depends on Δ analytically, and the induction / same-content pair
     sets are not offset-matched. A positive result is obtainable from an
     offset-distribution difference alone.
  2. The stated null — "same-content pairs have x_q ~ x_k so x^T A x -> 0" —
     is a statement about the CONTENT bilinear. Rotation breaks the symmetry
     it relies on, so same-content pairs are not a floor.
  3. Content and positional antisymmetry are not additive, so the rotary
     contribution cannot be subtracted off and the remainder called content.
     Nulls must be computed at the offsets in play.

Hence: offset-matched comparison, and three nulls reported together rather
than one.

  N1 rotary-only        rope_sa_fractions(delta) — closed form, no weights
  N2 offset-matched     same-content pairs at the SAME delta
  N3 offset-shuffled    induction pairs with delta permuted within the set

A pass requires clearing N1 and N2. N3 separates "content and offset jointly
required" from "either alone suffices".

Frame discipline
----------------
Every function here takes activations already in the reader's frame and a
FrameSpec describing it. Nothing in this module normalizes anything. Callers
that pass L2-normalized activations to a Pythia head are making the error
this whole work stream exists to close, and the FrameSpec is what makes that
visible in the output record.

See DESIGN_pythia_frames.md item 5 and the PREDICTIONS addendum.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from core.frames import FrameSpec, attach_frame
from core.metrics import _as_numpy
from core.rope import (
    DEFAULT_ROPE_BASE,
    apply_rope,
    causal_pair_mask,
    rope_rotation,
    rope_sa_fractions,
)


P6_I2B_MIN_DELTA = 0.05
P6_I2B_ALPHA = 0.05
P6_I2B_MIN_FIDELITY = 0.9


# ---------------------------------------------------------------------------
# Offset-resolved logit partition
# ---------------------------------------------------------------------------

def offset_logit_partition(
    X,
    WQ,
    WK,
    rotary_ndims: int,
    base: float = DEFAULT_ROPE_BASE,
    bq=None,
    bk=None,
    scale: float | None = None,
    positions=None,
) -> dict:
    """
    Split each pair's logit into symmetric and antisymmetric contributions,
    with the rotation at that pair's own offset.

    For pair (i, j) with Delta = j - i the logit is q_i^T R(Delta) k_j. The
    S/A split is taken of the effective per-pair bilinear:

        s_contrib[i, j] = ( q_i^T R(D) k_j + q_j^T R(-D) k_i ) / 2
        a_contrib[i, j] = ( q_i^T R(D) k_j - q_j^T R(-D) k_i ) / 2

    which is exactly qk_decompose's x^T S x / x^T A x when R = I, and reduces
    to it on non-rotary models. Note q_j^T R(-D) k_i is the logit of the
    REVERSED pair — the transpose is over the pair, not over the matrix, so
    this stays computable without ever forming M(Delta).

    Returns dict with s_contrib, a_contrib, a_frac_mat, logits, offsets.
    """
    Xa = _as_numpy(X).astype(np.float64, copy=False)
    n = Xa.shape[0]
    pos = np.arange(n, dtype=np.float64) if positions is None else \
        np.atleast_1d(_as_numpy(positions)).astype(np.float64, copy=False)

    Q = Xa @ _as_numpy(WQ).astype(np.float64, copy=False)
    K = Xa @ _as_numpy(WK).astype(np.float64, copy=False)
    if bq is not None:
        Q = Q + _as_numpy(bq).astype(np.float64, copy=False).reshape(1, -1)
    if bk is not None:
        K = K + _as_numpy(bk).astype(np.float64, copy=False).reshape(1, -1)

    if rotary_ndims > 0:
        Qe = apply_rope(Q, pos, rotary_ndims, base)
        Ke = apply_rope(K, pos, rotary_ndims, base)
    else:
        Qe, Ke = Q, K

    logits = Qe @ Ke.T
    if scale is not None:
        logits = logits * float(scale)

    # forward[i, j] = logit(i -> j); reverse[i, j] = logit(j -> i)
    forward = logits
    reverse = logits.T
    s_contrib = (forward + reverse) / 2.0
    a_contrib = (forward - reverse) / 2.0

    denom = np.abs(s_contrib) + np.abs(a_contrib)
    a_frac_mat = np.divide(
        np.abs(a_contrib), denom,
        out=np.zeros_like(denom), where=denom > 1e-24,
    )
    offsets = pos[None, :] - pos[:, None]        # offsets[i, j] = j - i

    return {
        "logits": logits,
        "s_contrib": s_contrib,
        "a_contrib": a_contrib,
        "a_frac_mat": a_frac_mat,
        "offsets": offsets,
    }


def head_a_frac_by_offset(
    WQ,
    WK,
    offsets,
    head_size: int,
    rotary_ndims: int,
    base: float = DEFAULT_ROPE_BASE,
) -> dict:
    """
    Weight-level a_frac of M(Delta) at each requested offset, plus the
    rotary-only null at the same offsets.

    The weight-level number is a property of the head; the pair-level
    a_frac_mat above is a property of the head AND the activations. Reporting
    both is what separates "this head's bilinear is antisymmetric" from
    "these particular tokens produced an antisymmetric logit".

    Returns dict(offsets, a_frac_weight, a_frac_rotary_null).
    """
    from core.rope import qk_sa_fractions_at_offset

    offs = np.unique(np.asarray(offsets, dtype=np.int64).ravel())
    aw, an = [], []
    for d in offs:
        R = rope_rotation(int(d), head_size, rotary_ndims, base)
        aw.append(qk_sa_fractions_at_offset(WQ, WK, R)["a_frac"])
        an.append(rope_sa_fractions(int(d), head_size, rotary_ndims, base)["a_frac"])
    return {
        "offsets": offs,
        "a_frac_weight": np.asarray(aw),
        "a_frac_rotary_null": np.asarray(an),
    }


# ---------------------------------------------------------------------------
# Offset matching
# ---------------------------------------------------------------------------

def pair_offsets(pairs) -> np.ndarray:
    """
    Delta = key - query for canonical (query, key) tuples with query > key.

    Always non-positive: causal attention only lets a later query read an
    earlier key, so the rotary offsets in play are all <= 0. Any analysis
    that sweeps positive offsets is describing pairs the softmax never sees.
    """
    if len(pairs) == 0:
        return np.zeros(0, dtype=np.int64)
    arr = np.asarray(pairs, dtype=np.int64)
    return arr[:, 1] - arr[:, 0]


def match_pairs_by_offset(target_pairs, pool_pairs, tolerance: int = 0) -> dict:
    """
    For each target pair, the pool pairs sharing its offset (within tolerance).

    This is the fix for the confound that kills the original P6-I2: comparing
    an induction pair at Delta = -7 against a pooled same-content mean drawn
    mostly from Delta = -1 measures the offset distribution, not the content.

    Returns dict(matched, unmatched_targets, coverage) where `matched` is a
    list of (target_pair, [pool_pairs]) and `coverage` is the fraction of
    targets with at least one match. Low coverage is a reason to fall back to
    the regression route, not a reason to pool.
    """
    t_off = pair_offsets(target_pairs)
    p_off = pair_offsets(pool_pairs)
    matched, unmatched = [], []
    for tp, d in zip(target_pairs, t_off):
        sel = [pp for pp, pd in zip(pool_pairs, p_off) if abs(int(pd - d)) <= tolerance]
        if sel:
            matched.append((tuple(tp), sel))
        else:
            unmatched.append(tuple(tp))
    n = max(len(list(target_pairs)), 1)
    return {
        "matched": matched,
        "unmatched_targets": unmatched,
        "coverage": len(matched) / n,
    }


def residualize_on_offset(values, offsets, degree: int = 1) -> dict:
    """
    Remove the offset trend from a_frac before comparing groups.

    The fallback when offset matching has poor coverage. Fits a low-degree
    polynomial in Delta across ALL pairs — pooling the groups deliberately,
    so the fit cannot absorb the group difference being tested — and returns
    residuals.

    Returns dict(residuals, coeffs, r2).
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    d = np.asarray(offsets, dtype=np.float64).ravel()
    if v.shape != d.shape:
        raise ValueError(f"residualize_on_offset: shape mismatch {v.shape} vs {d.shape}")
    if v.size <= degree + 1:
        raise ValueError(
            f"residualize_on_offset: {v.size} points cannot support degree "
            f"{degree}; use offset matching instead"
        )
    coeffs = np.polyfit(d, v, degree)
    fit = np.polyval(coeffs, d)
    resid = v - fit
    ss_tot = float(np.sum((v - v.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return {"residuals": resid, "coeffs": coeffs, "r2": r2}


# ---------------------------------------------------------------------------
# The three nulls
# ---------------------------------------------------------------------------

def rotary_null(offsets, head_size: int, rotary_ndims: int,
                base: float = DEFAULT_ROPE_BASE) -> np.ndarray:
    """
    N1. The a_frac geometry alone supplies at each offset. Closed form, no
    weights, no activations — so it cannot be fitted to the data.
    """
    return np.asarray(
        [rope_sa_fractions(int(d), head_size, rotary_ndims, base)["a_frac"]
         for d in np.asarray(offsets).ravel()],
        dtype=np.float64,
    )


def offset_matched_null(a_frac_mat, induction_pairs, same_content_pairs,
                        tolerance: int = 0) -> dict:
    """
    N2. Same-content pairs at the same offset as each induction pair.

    Returns dict(deltas, per_pair, coverage, n_used) where `deltas` are the
    per-induction-pair differences (induction a_frac minus the mean of its
    offset-matched same-content pairs).
    """
    A = np.asarray(a_frac_mat, dtype=np.float64)
    m = match_pairs_by_offset(induction_pairs, same_content_pairs, tolerance)
    deltas, per_pair = [], []
    for tp, pool in m["matched"]:
        q, k = tp
        ref = float(np.mean([A[pq, pk] for pq, pk in pool]))
        val = float(A[q, k])
        deltas.append(val - ref)
        per_pair.append({"pair": tp, "offset": int(k - q), "a_frac": val,
                         "matched_mean": ref, "delta": val - ref,
                         "n_matched": len(pool)})
    return {
        "deltas": np.asarray(deltas, dtype=np.float64),
        "per_pair": per_pair,
        "coverage": m["coverage"],
        "n_used": len(deltas),
        "unmatched": m["unmatched_targets"],
    }


def offset_shuffled_null(a_frac_mat, induction_pairs, n_shuffles: int = 200,
                         seed: int = 0) -> dict:
    """
    N3. Induction pairs with their offsets permuted within the set.

    Preserves the marginal distributions of both content-match and offset
    while destroying their pairing, so a statistic that survives this
    requires the two JOINTLY rather than either alone. Implemented by
    re-reading a_frac at (query, query + permuted_offset) where that index
    is in range and still causal.

    Returns dict(null_means, observed_mean, p_value, n_valid).
    """
    A = np.asarray(a_frac_mat, dtype=np.float64)
    n = A.shape[0]
    pairs = [tuple(p) for p in induction_pairs]
    if len(pairs) < 2:
        return {"null_means": np.zeros(0), "observed_mean": float("nan"),
                "p_value": None, "n_valid": 0, "degenerate": True,
                "n_distinct_offsets": 0}

    offs = pair_offsets(pairs)
    observed = float(np.mean([A[q, k] for q, k in pairs]))

    # Permuting identical offsets is a no-op. Induction pairs frequently all
    # share one offset (the repeat period), in which case N3 has no power at
    # all — say so rather than returning a p-value of 1.0 that looks like
    # evidence of absence.
    n_distinct = int(np.unique(offs).size)

    # Distinct permutations of the offset multiset. With k pairs sharing one
    # offset the permutation is a no-op; more generally the smallest p this
    # test can ever return is 1/n_perms, so a small pair set is underpowered
    # by construction and must say so rather than returning a large p that
    # reads as evidence of absence.
    counts = Counter(int(o) for o in offs)
    n_perms = math.factorial(len(pairs))
    for c in counts.values():
        n_perms //= math.factorial(c)
    min_p = max(1.0 / (int(n_shuffles) + 1), 1.0 / max(n_perms, 1))

    if n_distinct < 2:
        return {"null_means": np.zeros(0), "observed_mean": observed,
                "p_value": None, "n_valid": 0, "degenerate": True,
                "n_distinct_offsets": n_distinct, "n_perms": n_perms,
                "min_achievable_p": min_p, "underpowered": True}

    rng = np.random.default_rng(seed)

    null_means = []
    for _ in range(int(n_shuffles)):
        perm = rng.permutation(offs)
        vals = []
        for (q, _k), d in zip(pairs, perm):
            k2 = q + int(d)
            if 0 <= k2 < q:                     # in range and still causal
                vals.append(A[q, k2])
        if vals:
            null_means.append(float(np.mean(vals)))
    null_means = np.asarray(null_means, dtype=np.float64)
    if null_means.size == 0:
        return {"null_means": null_means, "observed_mean": observed,
                "p_value": None, "n_valid": 0, "degenerate": True,
                "n_distinct_offsets": n_distinct}
    p = float((np.sum(null_means >= observed) + 1) / (null_means.size + 1))
    return {"null_means": null_means, "observed_mean": observed,
            "p_value": p, "n_valid": int(null_means.size),
            "degenerate": False, "n_distinct_offsets": n_distinct,
            "n_perms": int(n_perms), "min_achievable_p": min_p,
            "underpowered": bool(min_p > P6_I2B_ALPHA)}


# ---------------------------------------------------------------------------
# P6-I2b
# ---------------------------------------------------------------------------

def evaluate_p6_i2b(
    a_frac_mat,
    induction_pairs,
    same_content_pairs,
    head_size: int,
    rotary_ndims: int,
    base: float = DEFAULT_ROPE_BASE,
    frame: FrameSpec | None = None,
    fidelity: dict | None = None,
    tolerance: int = 0,
    n_shuffles: int = 200,
    seed: int = 0,
) -> dict:
    """
    P6-I2b: after offset matching, is induction a_frac elevated above both
    the offset-matched same-content null (N2) and the rotary-only null (N1)?

    Deliberately reports a verdict of "null" rather than "fail" when the
    effect is absent: on a rotary architecture the expected result IS the
    negative, and it is a clean architectural finding rather than a
    disappointment. See the PREDICTIONS addendum.

    `fidelity` is the output of core.rope.qk_prediction_fidelity for this
    head. A head whose weight-space prediction does not track its real logits
    cannot support a claim about its bilinear, so a low value forces the
    verdict to "unverifiable" regardless of the statistics.
    """
    A = np.asarray(a_frac_mat, dtype=np.float64)
    result = {
        "n_induction": len(list(induction_pairs)),
        "n_same_content": len(list(same_content_pairs)),
    }

    if result["n_induction"] < 3 or result["n_same_content"] < 3:
        result.update(verdict="insufficient_pairs", delta_vs_n2=None,
                      delta_vs_n1=None, p_value_n3=None)
        return _stamp(result, frame)

    n2 = offset_matched_null(A, induction_pairs, same_content_pairs, tolerance)
    ind_off = pair_offsets(induction_pairs)
    ind_vals = np.asarray([A[q, k] for q, k in induction_pairs], dtype=np.float64)
    n1 = rotary_null(ind_off, head_size, rotary_ndims, base)
    n3 = offset_shuffled_null(A, induction_pairs, n_shuffles, seed)

    result.update(
        observed_mean=float(np.mean(ind_vals)),
        offset_coverage=n2["coverage"],
        n_offset_matched=n2["n_used"],
        delta_vs_n2=float(np.mean(n2["deltas"])) if n2["n_used"] else None,
        delta_vs_n1=float(np.mean(ind_vals - n1)),
        rotary_null_mean=float(np.mean(n1)),
        p_value_n3=n3["p_value"],
        n3_degenerate=bool(n3.get("degenerate", False)),
        n3_underpowered=bool(n3.get("underpowered", False)),
        n3_min_achievable_p=n3.get("min_achievable_p"),
        offsets=ind_off.tolist(),
    )

    if fidelity is not None and fidelity.get("pearson", 1.0) < P6_I2B_MIN_FIDELITY:
        result["verdict"] = "unverifiable_low_fidelity"
        result["fidelity"] = fidelity
        return _stamp(result, frame)

    if n2["coverage"] < 0.5:
        # Too few matched offsets to trust the matched comparison. Say so
        # rather than silently falling back to a pooled mean.
        result["verdict"] = "insufficient_offset_coverage"
        return _stamp(result, frame)

    clears_n2 = (result["delta_vs_n2"] or 0.0) > P6_I2B_MIN_DELTA
    clears_n1 = result["delta_vs_n1"] > 0.0
    clears_n3 = (n3["p_value"] is not None
                 and n3["p_value"] < P6_I2B_ALPHA)

    result["clears_n1"] = bool(clears_n1)
    result["clears_n2"] = bool(clears_n2)
    result["clears_n3"] = bool(clears_n3)
    result["verdict"] = "supported" if (clears_n1 and clears_n2) else "null"
    return _stamp(result, frame)


def _stamp(result: dict, frame: FrameSpec | None) -> dict:
    if frame is not None:
        attach_frame(result, frame)
    return result
