"""
p2d_operator_activation/operator_pairing.py — sub-experiments D2 and D4:
the quantities neither Phase 1 nor Phase 2 can compute alone.

THE GAP THIS FILLS

The paper's E_beta assumes Q^T K = I. The model's own coupling is
M_h = W_Q^{(h)T} W_K^{(h)} / sqrt(d_head), acting on the LN'd states y that
attention actually reads:

    E_beta^{(h)} = (1 / 2 beta) * < exp(beta * y_i^T M_h y_j) >

Expanding:

  FIRST ORDER   <y^T M y'> = ybar^T M ybar, a quadratic form of the QK
                operator at the centroid. Negative when the centroid
                overlaps Sym(M)'s negative eigenspace. This is
                `V_repulsive_local` as a scalar on the same axis as
                E_beta, which is what makes the two comparable at all.

  SECOND ORDER  <(y^T M y')^2> = tr(M^T C M C) = sum_ab lam_a lam_b
                |<u_a, M u_b>|^2, with C the token covariance.

That second-order term is the bilinear pairing of the OPERATOR spectrum
against the ACTIVATION covariance spectrum. Phase 2 has the left factor.
Phase 1 has the right factor. Nobody has computed the product.

It defines an operator-conditioned rank:

    PR_M = (tr M C)^2 / tr(M^T C M C)
         = how many of the cloud's directions this head actually couples

Heads with large ||M|| and small PR_M are strong operators pointed where
the tokens are not. That is a candidate explanation for a specific
unexplained Phase 1 observation: the beta-independence of energy violations
after step 512. If M concentrates on few directions, the higher moments
collapse and only <G> survives, which would make the violation count
insensitive to beta exactly as observed.

SANITY ANCHOR: at M = I, PR_M reduces to (tr C)^2 / tr(C^2), the ordinary
participation-ratio rank of the activation cloud. Asserted in the tests
rather than assumed.

COST: [R + W]. Weights plus saved activations, no forward pass.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# D2 — operator-conditioned rank
# ---------------------------------------------------------------------------

def token_covariance(Y: np.ndarray, center: bool = True) -> np.ndarray:
    """
    C = (1/n) sum_i y_i y_i^T from LN'd states.

    center : subtract the mean first. BOTH are meaningful and they answer
             different questions — the uncentred C carries the common mode,
             which for a strongly anisotropic transformer cloud dominates
             its top eigenvalue, so PR_M computed on it can read ~1 purely
             because every token shares a direction. The centred version
             asks which of the SPREAD directions the head couples. Default
             centred; the caller should run both when the common mode is
             large (which core/metrics kappa_1 measures directly).
    """
    Y = np.asarray(Y, dtype=np.float64)
    if center:
        Y = Y - Y.mean(axis=0, keepdims=True)
    return (Y.T @ Y) / Y.shape[0]


def operator_conditioned_rank(M: np.ndarray, C: np.ndarray) -> dict:
    """
    PR_M = (tr M C)^2 / tr(M^T C M C), plus the pieces it is built from.

    Computed via the trace forms directly rather than by forming the
    d x d products where avoidable:

        tr(M C)       = sum(M * C^T)
        tr(M^T C M C) = sum((C M) * (M C))

    both O(d^2) in memory instead of a d x d x d chain, which matters at
    d = 1024 across 24 layers x 16 heads.

    THE SECOND CONTRACTION IS EASY TO GET WRONG AND THE M = I ANCHOR DOES
    NOT CATCH IT. sum((C M) * (C M^T)) contracts to tr(C M M C), not to
    tr(M^T C M C); the two coincide at M = I and at symmetric M, so an
    identity-anchored test passes while the quantity is wrong for every
    real head. The error is visible only in the sign: tr(M^T C M C) =
    ||C^{1/2} M C^{1/2}||_F^2 and is therefore NON-NEGATIVE, while the
    wrong contraction goes negative on a generic M (measured: -72.08
    against a true +167.00). `coupled_mass` below is non-negative by
    construction, which makes it the running check on this — a negative
    value means the contraction has been broken again.

    `coupling_efficiency` is PR_M / PR_C: what fraction of the cloud's own
    effective dimensionality this head actually touches. PR_M alone
    conflates "the head is selective" with "the cloud is low-rank", and
    the whole point of the pairing is to separate those.

    THE CANCELLATION TRAP. The numerator (tr M C)^2 uses the SIGNED trace,
    so a head that couples the cloud strongly but with mixed signs — some
    directions amplified, others suppressed — has tr(MC) near zero and
    reads PR_M near zero, which looks identical to a head that couples
    nothing at all. Measured on a random M against a near-isotropic C,
    PR_M comes out at 0.00 for exactly this reason, and the derived
    `misdirection` then diverges.

    These are opposite situations and must not share a number, so
    `coupled_mass` is reported alongside:

        coupled_mass = tr(M^T C M C) / (||M||_F^2 * tr(C^2))

    which is sign-blind and measures how much of the cloud's variance the
    operator touches at all. Read the pair: low PR_M with low coupled_mass
    is a head pointed away from the tokens (the beta-independence
    hypothesis); low PR_M with HIGH coupled_mass is a head that couples
    strongly and cancels, which is a rotation, not an absence.
    """
    M = np.asarray(M, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    if M.shape != C.shape:
        raise ValueError(f"M {M.shape} and C {C.shape} must match")

    tr_MC = float((M * C.T).sum())
    # tr(M^T C M C) = sum((C M) * (M C)). See the docstring: the plausible
    # (C M) * (C M^T) is tr(C M M C) and differs on any non-symmetric M.
    tr_MtCMC = float(((C @ M) * (M @ C)).sum())
    if tr_MtCMC < -1e-9 * max(abs(tr_MC), 1.0):
        raise AssertionError(
            f"tr(M^T C M C) = {tr_MtCMC:.6g} < 0, which is impossible for "
            f"symmetric PSD C (it equals ||C^(1/2) M C^(1/2)||_F^2). Either "
            f"C is not a covariance or the contraction is wrong."
        )

    tr_C = float(np.trace(C))
    tr_C2 = float((C * C.T).sum())
    pr_C = (tr_C ** 2 / tr_C2) if tr_C2 > 1e-30 else float("nan")
    pr_M = (tr_MC ** 2 / tr_MtCMC) if tr_MtCMC > 1e-30 else float("nan")

    fro_M2 = float((M * M).sum())
    coupled = (tr_MtCMC / (fro_M2 * tr_C2)) if (fro_M2 > 1e-30 and tr_C2 > 1e-30) else float("nan")

    return {
        "pr_M": float(pr_M),
        "pr_C": float(pr_C),
        "coupled_mass": float(coupled),
        # Guard against reading a cancellation as an absence — see the
        # docstring. True when the signed trace has collapsed but the
        # operator is still touching the cloud.
        "cancellation_suspected": bool(
            np.isfinite(pr_M) and pr_M < 1.0
            and np.isfinite(coupled) and coupled > 0.5 / max(pr_C, 1.0)
        ),
        "coupling_efficiency": float(pr_M / pr_C) if np.isfinite(pr_C) and pr_C > 1e-12 else float("nan"),
        "tr_MC": tr_MC,
        "tr_MtCMC": tr_MtCMC,
        "fro_M": float(np.linalg.norm(M, "fro")),
        # A strong operator pointed where the tokens are not: large norm,
        # small coupled rank. This is the composite the beta-independence
        # hypothesis predicts should be large.
        # nan rather than a huge number when pr_M has collapsed: a
        # divide-by-almost-zero would sort straight to the top of exactly
        # the ranking this quantity exists to produce. Consult
        # `cancellation_suspected` before reading a large value — a
        # rotation-like head has pr_M ~ 0 for a reason that is not
        # misdirection.
        "misdirection": (float(np.linalg.norm(M, "fro") / pr_M)
                         if (np.isfinite(pr_M) and pr_M > 1e-6)
                         else float("nan")),
    }


def spectral_pairing(M: np.ndarray, C: np.ndarray, top_k: int = 32) -> dict:
    """
    The explicit sum_ab lam_a lam_b |<u_a, M u_b>|^2 over the top-k
    activation eigendirections.

    tr(M^T C M C) is that sum in closed form, so this adds nothing to the
    scalar — it decomposes it. The reportable object is the (a, b) matrix
    of contributions: whether a head couples the cloud's top direction to
    itself (diagonal-dominant, an amplifier), to a different direction
    (off-diagonal, a rotator), or to nothing in the top-k (mass leaking to
    the tail, meaning the head reads directions the cloud barely occupies).

    `top_k_mass` is the check on that last case: if it is small, the scalar
    PR_M is being set by directions outside the truncation and the
    decomposition below is not representative.
    """
    C = np.asarray(C, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    lam, U = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1][:top_k]
    lam_k, U_k = lam[order], U[:, order]

    MU = M @ U_k
    proj = U_k.T @ MU                       # <u_a, M u_b>
    contrib = np.outer(lam_k, lam_k) * (proj ** 2)

    tr_full = float(((C @ M) * (M @ C)).sum())
    return {
        "eigenvalues": lam_k,
        "contrib": contrib,
        "top_k_mass": float(contrib.sum() / tr_full) if abs(tr_full) > 1e-30 else float("nan"),
        "diag_fraction": float(np.trace(contrib) / contrib.sum())
        if contrib.sum() > 1e-30 else float("nan"),
        "cloud_top_k_mass": float(lam_k.sum() / max(lam.sum(), 1e-30)),
        "top_k": int(len(lam_k)),
    }


# ---------------------------------------------------------------------------
# D4 — the model's own energy
# ---------------------------------------------------------------------------

def generalized_energy(Y: np.ndarray, M: np.ndarray, betas=(0.1, 1.0, 2.0, 5.0),
                       normalize_rows: bool = True) -> dict:
    """
    E_beta^{(h)} = (1/2 beta) < exp(beta y^T M y') > on LN'd states.

    normalize_rows : project y onto the sphere first. The paper's E_beta is
                     defined for unit-norm particles, and leaving the norms
                     in makes the exponent scale with ||y||^2 — which for a
                     transformer's growing residual stream would make the
                     energy a norm measurement wearing a geometry costume.
                     The norms are not discarded silently: `norm_cv` is
                     reported so the size of what was removed is visible.

    The exponent is shifted by its max before exponentiating. Without that,
    beta = 5 on a d = 1024 cloud with an untamed M overflows float64, and
    the overflow shows up as inf in one layer of one head rather than as an
    error — which is how a NaN gets into an aggregate.
    """
    Y = np.asarray(Y, dtype=np.float64)
    norms = np.linalg.norm(Y, axis=1)
    if normalize_rows:
        Y = Y / np.maximum(norms[:, None], 1e-12)

    M = np.asarray(M, dtype=np.float64)
    Z = Y @ M @ Y.T                       # (n, n) logits y_i^T M y_j
    n = Z.shape[0]

    out = {
        "norm_cv": float(norms.std() / max(norms.mean(), 1e-12)),
        "logit_mean": float(Z.mean()),
        "logit_std": float(Z.std()),
        "energies": {},
        "overflow_guarded": {},
    }
    for b in betas:
        b = float(b)
        x = b * Z
        mx = float(x.max())
        # exp(x) = exp(mx) * exp(x - mx); carry the shift analytically.
        val = np.exp(x - mx).mean()
        log_e = mx + np.log(max(val, 1e-300))
        out["energies"][b] = float(np.exp(log_e) / (2.0 * b)) if log_e < 700 else float("inf")
        out["overflow_guarded"][b] = bool(log_e >= 700)

    ybar = Y.mean(axis=0)
    first = float(ybar @ M @ ybar)
    C = (Y.T @ Y) / n
    second = float(((C @ M) * (M @ C)).sum())

    out["first_order"] = first          # ybar^T M ybar — the repulsion scalar
    out["second_order"] = second        # tr(M^T C M C)
    # Sign of the first-order term IS the attractive/repulsive call for this
    # head, on the same axis as E_beta and computed from the model's own
    # operator rather than from the identity-weight proxy.
    out["regime"] = ("attractive" if first > 0 else
                     "repulsive" if first < 0 else "neutral")
    return out


def energy_attribution(E_identity: dict, E_head: dict) -> dict:
    """
    Put the paper's proxy and the model's own energy side by side.

    E_identity : {beta: E} from core.metrics.interaction_energies_batched
                 (the Q^T K = I assumption)
    E_head     : the `energies` dict from generalized_energy

    The two together give an attribution of the monotonicity break: if the
    identity-weight energy is non-monotone but the head's own energy is
    monotone, the break is an artifact of the Q^T K = I substitution, not a
    property of the model. That possibility has never been checked and it
    would invalidate the reading of every violation count in Phase 1.
    """
    betas = sorted(set(E_identity) & set(E_head))
    return {
        "betas": betas,
        "identity": {b: float(E_identity[b]) for b in betas},
        "head": {b: float(E_head[b]) for b in betas},
        "ratio": {b: (float(E_head[b] / E_identity[b])
                      if abs(E_identity[b]) > 1e-15 else float("nan"))
                  for b in betas},
    }


def monotonicity_compare(E_identity_series: list, E_head_series: list,
                         rel_tol: float = 1e-3) -> dict:
    """
    Violation counts under both energies, on the same relative rule Phase 1
    uses, so the counts are directly comparable.

    `explained_by_proxy` is the case that matters: violations under the
    identity-weight energy that DISAPPEAR under the head's own energy. Each
    one is a Phase 1 violation that was an artifact of assuming Q^T K = I.
    """
    def _viol(series):
        a = np.asarray(series, dtype=np.float64)
        idx = []
        for i in range(1, len(a)):
            if not (np.isfinite(a[i]) and np.isfinite(a[i - 1])):
                continue
            if a[i - 1] - a[i] > rel_tol * abs(a[i - 1]):
                idx.append(i)
        return idx

    vi, vh = _viol(E_identity_series), _viol(E_head_series)
    si, sh = set(vi), set(vh)
    return {
        "violations_identity": vi,
        "violations_head": vh,
        "n_identity": len(vi),
        "n_head": len(vh),
        "explained_by_proxy": sorted(si - sh),
        "new_under_head": sorted(sh - si),
        "shared": sorted(si & sh),
        "rel_tol": rel_tol,
    }
