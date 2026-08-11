"""
p2d_operator_activation/gradient_flow_condition.py — sub-experiment D1:
which heads are actually inside the theorem's hypotheses?

THE CONDITION

Section 3.4. (SA) is NOT a Wasserstein gradient flow in the standard
metric — its field is a logarithmic derivative (eq. 3.7). It IS a gradient
flow in the reweighted metric

    <a, b>_X = sum_i Z_{beta,i}(X) <a_i, b_i>                    (3.12)

under a standing condition that is stated and then never revisited:

    Q^T K symmetric,   AND   V = Q^T K.

Both halves are directly measurable on a trained model, and this project
has never tested either.

WHY THIS CHANGES THE QUESTION

Phase 1 reports energy-monotonicity violations rising from 3 at step 0 to
64 by step 512, and reads that as the theorem failing. But the theorem has
hypotheses. A head satisfying both conditions above is inside the paper's
gradient-flow regime and MUST show monotone E_beta; a head far outside
carries no monotonicity guarantee at all.

So the sharp question is not "is the theorem violated" but "which heads are
outside its hypotheses, and do the violations localize there?" That is
prediction P-M1, and it converts a falsification into an attribution.

WHAT IS MEASURED, PER HEAD

    asymmetry     ||Skew(M_h)||_F / ||M_h||_F,  Skew(A) = (A - A^T)/2
                  0 = perfectly symmetric (condition 1 satisfied)
                  1/sqrt(2)-ish = a generic random matrix
                  1 = perfectly antisymmetric (pure rotation)

    ov_qk_align   <W_OV, M_h>_F / (||W_OV||_F ||M_h||_F)
                  1 = V equals Q^T K up to positive scale (condition 2)

    in_regime     both conditions within tolerance

with M_h = W_Q^{(h)T} W_K^{(h)} / sqrt(d_head), acting on the LN'd states
attention actually reads.

A NOTE ON SCALE. Condition 2 as written is V = Q^T K exactly, but the
cosine above is scale-invariant, so it tests V ∝ Q^T K with a positive
constant. That is the right relaxation: an overall positive rescaling of V
rescales time in the ODE and does not change the gradient-flow structure or
the sign of dE/dt. A NEGATIVE constant does — it is the V = -I_d repulsive
case — which is why the signed cosine is reported rather than its absolute
value, and why `ov_qk_align` near -1 is recorded as a distinct regime
rather than as "aligned".

p2b_imaginary/rotational_schur.py already performs the symmetric/
antisymmetric split this needs; `asymmetry` here is the same decomposition
read as a scalar, deliberately recomputed from M rather than imported so
that the two can be cross-checked rather than sharing a bug.
"""

from __future__ import annotations

import numpy as np


SYMMETRY_TOL = 0.10        # ||Skew||/||M|| below this counts as symmetric
ALIGN_TOL = 0.70           # cosine above this counts as V ~ Q^T K


def qk_matrix(wq: np.ndarray, wk: np.ndarray, d_head: int = None) -> np.ndarray:
    """
    M_h = W_Q^{(h)} W_K^{(h)T} / sqrt(d_head), in (d_model, d_model).

    Inputs are the canonical (d_model, d_head) orientation that
    p2_eigenspectra/weights.extract_qk_per_head produces, so that
    x_i^T M x_j is the attention logit. The 1/sqrt(d_head) is included
    because it is not a convention — the paper's footnote 2 is explicit
    that beta arises from exactly this factor together with the typical
    magnitude of Q and K, so an M without it is off by a factor that IS
    the quantity beta_eff measures.
    """
    wq = np.asarray(wq, dtype=np.float64)
    wk = np.asarray(wk, dtype=np.float64)
    if wq.shape != wk.shape:
        raise ValueError(f"wq {wq.shape} and wk {wk.shape} must match")
    d_h = int(d_head if d_head is not None else wq.shape[1])
    return (wq @ wk.T) / np.sqrt(d_h)


def symmetry_split(M: np.ndarray) -> dict:
    """
    Sym/Skew decomposition and the asymmetry ratio.

    Also returns the eigenvalue range of Sym(M), because condition 1 alone
    is not the whole story: section 3.4 wants Q^T K symmetric, but the
    first-order term of the generalized energy is a quadratic form of
    Sym(M) at the centroid, so a symmetric M with a negative-definite
    Sym is symmetric AND repulsive. The two facts travel together and
    reporting only the ratio would hide it.
    """
    M = np.asarray(M, dtype=np.float64)
    S = 0.5 * (M + M.T)
    A = 0.5 * (M - M.T)
    nm = float(np.linalg.norm(M, "fro"))
    if nm < 1e-15:
        return {"asymmetry": float("nan"), "fro_M": nm}
    ev = np.linalg.eigvalsh(S)
    return {
        "asymmetry": float(np.linalg.norm(A, "fro") / nm),
        "sym_frac": float(np.linalg.norm(S, "fro") / nm),
        "fro_M": nm,
        "sym_eig_min": float(ev[0]),
        "sym_eig_max": float(ev[-1]),
        "sym_frac_negative": float((ev < 0).mean()),
        "sym_trace": float(ev.sum()),
    }


def ov_qk_alignment(W_OV: np.ndarray, M: np.ndarray) -> dict:
    """
    Signed Frobenius cosine between the OV circuit and the QK matrix.

    Also returns the alignment against Sym(M) alone. The condition is
    V = Q^T K, and if M is far from symmetric then no symmetric V can
    match it; splitting the two tells us whether a head fails the
    condition because V is wrong or because M is not symmetric in the
    first place. Those are different failures and the plain cosine
    conflates them.
    """
    V = np.asarray(W_OV, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    if V.shape != M.shape:
        raise ValueError(f"W_OV {V.shape} and M {M.shape} must match")
    nv, nm = float(np.linalg.norm(V, "fro")), float(np.linalg.norm(M, "fro"))
    if nv < 1e-15 or nm < 1e-15:
        return {"align": float("nan"), "align_sym": float("nan")}
    S = 0.5 * (M + M.T)
    ns = float(np.linalg.norm(S, "fro"))
    return {
        "align": float((V * M).sum() / (nv * nm)),
        "align_sym": float((V * S).sum() / (nv * ns)) if ns > 1e-15 else float("nan"),
        "fro_OV": nv,
        "scale_ratio": float(nv / nm),
    }


def head_regime(wq, wk, W_OV, d_head: int = None,
                sym_tol: float = SYMMETRY_TOL,
                align_tol: float = ALIGN_TOL) -> dict:
    """
    Full gradient-flow classification for one head.

    `regime` is one of:
      gradient_flow     symmetric M and V aligned with it. The theorem's
                        hypotheses hold; this head MUST show monotone
                        E_beta, and a violation here is a real anomaly.
      symmetric_only    M symmetric but V not aligned. Section 3.4 does not
                        apply; no monotonicity guarantee.
      repulsive_aligned V anti-aligned with M (cosine < -align_tol). This is
                        the V = -I_d case in disguise, where the paper
                        itself predicts DECREASING energy. A violation here
                        is the predicted behaviour.
      outside           neither condition. No guarantee in either direction.
    """
    M = qk_matrix(wq, wk, d_head=d_head)
    sym = symmetry_split(M)
    ali = ov_qk_alignment(W_OV, M)

    is_sym = np.isfinite(sym["asymmetry"]) and sym["asymmetry"] < sym_tol
    a = ali["align"]
    if not np.isfinite(a):
        regime = "outside"
    elif a < -align_tol:
        regime = "repulsive_aligned"
    elif is_sym and a > align_tol:
        regime = "gradient_flow"
    elif is_sym:
        regime = "symmetric_only"
    else:
        regime = "outside"

    return {
        **sym, **ali,
        "regime": regime,
        "in_gradient_flow_regime": regime == "gradient_flow",
        "sym_tol": sym_tol, "align_tol": align_tol,
        # Distance from the regime, so heads can be ORDERED rather than
        # only binned. P-M1 is a correlation claim, and a two-bin split
        # throws away most of the power a continuous score has.
        "regime_distance": float(np.hypot(max(sym["asymmetry"], 0.0),
                                          max(1.0 - a, 0.0)))
        if np.isfinite(a) and np.isfinite(sym["asymmetry"]) else float("nan"),
    }


def layer_regimes(wq_per_head, wk_per_head, ov_per_head,
                  d_head: int = None, **kw) -> list:
    """Per-head classification for one layer."""
    n = min(len(wq_per_head), len(wk_per_head), len(ov_per_head))
    return [dict(head=h, **head_regime(wq_per_head[h], wk_per_head[h],
                                       ov_per_head[h], d_head=d_head, **kw))
            for h in range(n)]


# ---------------------------------------------------------------------------
# P-M1
# ---------------------------------------------------------------------------

def adjudicate_p_m1(regimes: list, violations: list) -> dict:
    """
    P-M1: energy-monotonicity violations concentrate in heads far from
    Q^T K symmetric and V = Q^T K.
    Falsifier: no correlation.

    regimes    : per-head dicts from head_regime, with a `layer` key
    violations : per-LAYER violation counts, aligned by layer index

    THE MEASUREMENT PROBLEM, STATED RATHER THAN PAPERED OVER. Phase 1's
    E_beta is computed on the residual stream, which is a per-LAYER
    quantity, while the regime score is per-HEAD. There is no per-head
    energy, so the correlation below is between a layer's violation count
    and an AGGREGATE of its heads' regime distances. That aggregate is a
    choice, and the answer depends on it:

      mean   treats the layer as failing if its heads fail on average
      min    treats one in-regime head as enough to protect the layer
      max    treats one out-of-regime head as enough to break it

    All three are reported. If they disagree in sign, P-M1 is not
    adjudicable from per-layer energies and needs per-head ablation
    instead — which is a real result about the experiment's resolution and
    should be recorded as one rather than resolved by picking whichever
    aggregate confirms.
    """
    by_layer: dict = {}
    for r in regimes:
        by_layer.setdefault(int(r.get("layer", -1)), []).append(r)

    layers = sorted(k for k in by_layer if k >= 0)
    if not layers:
        return {"verdict": "no layer-tagged regimes supplied"}

    v = np.asarray(violations, dtype=np.float64)
    out = {"n_layers": len(layers), "aggregates": {}}

    for name, fn in (("mean", np.nanmean), ("min", np.nanmin), ("max", np.nanmax)):
        score = np.array([fn([r["regime_distance"] for r in by_layer[l]])
                          for l in layers], dtype=np.float64)
        k = min(len(score), len(v))
        s, vv = score[:k], v[:k]
        m = np.isfinite(s) & np.isfinite(vv)
        if m.sum() < 3 or np.std(s[m]) < 1e-12 or np.std(vv[m]) < 1e-12:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(s[m], vv[m])[0, 1])
        out["aggregates"][name] = {"corr": corr, "n": int(m.sum())}

    corrs = [a["corr"] for a in out["aggregates"].values() if np.isfinite(a["corr"])]
    if not corrs:
        out["verdict"] = "UNDETERMINED — too few usable layers."
    elif len({np.sign(c) for c in corrs}) > 1:
        out["verdict"] = ("NOT ADJUDICABLE — the three head-to-layer aggregates "
                          "disagree in sign. Per-layer energies do not resolve "
                          "a per-head claim; this needs head ablation.")
    elif min(corrs) > 0.5:
        out["verdict"] = ("CONFIRMED — violations concentrate in layers whose "
                          "heads are far from the gradient-flow condition. The "
                          "monotonicity break is a hypothesis failure, not a "
                          "theorem failure.")
    elif max(corrs) < 0.2:
        out["verdict"] = ("FALSIFIED — no correlation. Violations are not "
                          "explained by leaving the gradient-flow regime, and "
                          "the break needs a different attribution.")
    else:
        out["verdict"] = "WEAK — correlation present but under 0.5 somewhere."

    # The cleanest evidence, if it exists: a head inside the regime whose
    # layer still violates is a genuine anomaly, not a hypothesis failure.
    in_regime_layers = [l for l in layers
                        if all(r["in_gradient_flow_regime"] for r in by_layer[l])]
    out["fully_in_regime_layers"] = in_regime_layers
    out["violations_in_regime_layers"] = [
        float(v[l]) for l in in_regime_layers if l < len(v)]
    return out
