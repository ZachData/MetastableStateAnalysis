"""
p1c_frames/hemisphere_feasibility.py — sub-experiment E: does the token
cloud satisfy Lemma 6.4's cone condition, and by how much?

THE CONDITION

Lemma 6.4: if all x_i(0) lie in an open hemisphere — i.e. there exists w
with <x_i, w> > 0 for every i — then the dynamics collapse to a single
point EXPONENTIALLY. The proof sets r(t) = min_i <x_i(t), w>, shows
r' >= 0 so the hemisphere is forward-invariant, and then gets the rate from
alpha' >= (1 - alpha) / (2 n e^{2 beta}).

The crucial detail, which the paper states explicitly: **only positivity of
the attention weights a_ij(t) is used.** That is why the lemma extends to
arbitrary Q, K with V = I. Softmax weights are always positive. So the
hypothesis of Lemma 6.4 is ENTIRELY a condition on the configuration —
nothing about the weights can rescue or break it.

Theorem 6.7 (Wendel) gives the probability for a random configuration:

    P(all in one hemisphere) = 2^{-(n-1)} * sum_{k < d} C(n-1, k)

which equals 1 whenever d > n. Our prompts have n in [20, 512] and
Pythia-410M has d = 1024, so d > n for every prompt: at random
initialization the tokens are almost surely in an open hemisphere, and
Lemma 6.4 then predicts exponential collapse.

WHY THE INTERESTING OUTCOME IS THE FAILURE

P-H1 is registered as "feasible at every checkpoint," which Wendel makes
near-certain for a random cloud. That is deliberate. The prediction is
stated in the direction that is boring if true, so that the informative
outcome — infeasibility, or feasibility with a margin near zero — is what
we are actually looking for. Either would mean the embedding layer is doing
something specific to escape a regime that otherwise forces exponential
collapse. Report the MARGIN, not the boolean.

Note this is a DIFFERENT question from p1b_hemisphere/. That module finds
bipartitions — how the cloud splits into two groups. This one tests whether
the whole cloud fits in one open halfspace through the origin. A cloud with
a clean bipartition can still satisfy the cone condition, and a cloud with
none can fail it.

COST: [R]. Everything below runs on the Gram matrix alone.

THE EXACT SOLUTION

The margin

    m = max_{||w|| <= 1} min_i <x_i, w>

is, by minimax duality, exactly the distance from the origin to the convex
hull of the points:

    m = dist(0, conv{x_i}) = sqrt( min_{lambda in simplex} lambda^T G lambda )

and the cone condition holds iff m > 0, i.e. iff 0 is not in the hull.

Two consequences worth having:

  * It needs only G. No d-dimensional optimization, no LP solver, no
    dependency. The problem is n-dimensional regardless of model width,
    and Phase 1 already computes G for every layer of every run.
  * It is a convex QP with an exact optimum, not a feasibility heuristic.
    A boolean from an LP with a tolerance would give "feasible" for a
    cloud whose margin is 1e-9, which is the case we most want to catch.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Wendel's theorem
# ---------------------------------------------------------------------------

def wendel_probability(n: int, d: int) -> float:
    """
    P(n random points on S^{d-1} all lie in some open hemisphere)
        = 2^{-(n-1)} * sum_{k=0}^{d-1} C(n-1, k).

    Computed in log space: at n = 512 the binomials overflow float64 well
    before the sum is taken, and the naive form silently returns inf/nan
    exactly in the range our prompts occupy.

    Returns 1.0 when d > n - 1 (the sum covers every term).
    """
    n, d = int(n), int(d)
    if n < 1:
        return 1.0
    if d >= n:
        return 1.0
    ks = np.arange(0, min(d, n))          # k < d, and C(n-1,k)=0 for k>n-1
    log_binom = (gammaln(n) - gammaln(ks + 1) - gammaln(n - ks))
    m = log_binom.max()
    total = m + np.log(np.exp(log_binom - m).sum())
    return float(np.exp(total - (n - 1) * np.log(2.0)))


# ---------------------------------------------------------------------------
# Minimum-norm point in the convex hull
# ---------------------------------------------------------------------------

def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {lambda >= 0, sum lambda = 1}. O(n log n)."""
    n = v.size
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def hull_min_norm(G: np.ndarray, max_iter: int = 5000,
                  tol: float = 1e-12) -> dict:
    """
    min_{lambda in simplex} lambda^T G lambda, by projected gradient with
    Nesterov acceleration and a Lipschitz step.

    G is the Gram of UNIT-NORM rows, so the objective's gradient is
    2 G lambda and the Lipschitz constant is 2 * lambda_max(G) <= 2n.
    Using the true top eigenvalue rather than the bound matters: on a
    strongly anisotropic cloud lambda_max is close to n and the bound is
    fine, but on a near-orthogonal one lambda_max ~ 1 and the bound would
    give a step n times too small.

    Returns margin (= sqrt of the optimum), the optimal lambda, and
    convergence diagnostics. `converged` False means the margin is an
    upper bound on the true one, which matters: an unconverged run can
    only make the cloud look MORE feasible than it is.
    """
    G = np.asarray(G, dtype=np.float64)
    n = G.shape[0]
    if n == 1:
        return {"margin": float(np.sqrt(max(G[0, 0], 0.0))), "lambda": np.array([1.0]),
                "n_iter": 0, "converged": True, "obj": float(G[0, 0])}

    L = 2.0 * float(np.linalg.eigvalsh(G)[-1])
    step = 1.0 / max(L, 1e-12)

    lam = np.full(n, 1.0 / n)
    y = lam.copy()
    t_k = 1.0
    obj_prev = float(lam @ G @ lam)
    n_iter = max_iter
    converged = False

    for it in range(max_iter):
        grad = 2.0 * (G @ y)
        lam_new = _project_simplex(y - step * grad)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k))
        y = lam_new + ((t_k - 1.0) / t_next) * (lam_new - lam)
        lam, t_k = lam_new, t_next

        obj = float(lam @ G @ lam)
        if abs(obj_prev - obj) < tol * max(abs(obj_prev), 1e-16):
            n_iter, converged = it + 1, True
            break
        obj_prev = obj

    obj = float(lam @ G @ lam)
    return {"margin": float(np.sqrt(max(obj, 0.0))), "lambda": lam,
            "n_iter": n_iter, "converged": converged, "obj": obj}


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

def hemisphere_test(G: np.ndarray, d_model: int = None,
                    zero_tol: float = 1e-8) -> dict:
    """
    Full cone-condition report for one layer.

    G          : (n, n) Gram of unit-norm activations for this layer
    d_model    : model width, for the Wendel reference probability
    zero_tol   : margin below this counts as "on the boundary" rather than
                 strictly feasible. A margin of 1e-9 is not a hemisphere in
                 any meaningful sense, and calling it one would hide
                 exactly the outcome this sub-experiment exists to detect.

    Returns feasible / boundary / margin / min_pairwise_ip / wendel_p /
    n_active_support and convergence info.
    """
    G = np.asarray(G, dtype=np.float64)
    n = G.shape[0]
    res = hull_min_norm(G)
    margin = res["margin"]

    lam = res["lambda"]
    # How many points carry the certificate. When 0 IS in the hull, the
    # support is the subset of tokens that spans it — i.e. the tokens
    # responsible for breaking the cone condition, which is the thing worth
    # reporting rather than just the failure.
    support = int(np.sum(lam > 1e-6))

    iu = np.triu_indices(n, k=1)
    min_ip = float(G[iu].min()) if n > 1 else 1.0

    out = {
        "n_tokens": int(n),
        "margin": float(margin),
        "feasible": bool(margin > zero_tol),
        "boundary": bool(margin <= zero_tol),
        "support_size": support,
        "min_pairwise_ip": min_ip,
        "converged": res["converged"],
        "n_iter": res["n_iter"],
        "zero_tol": float(zero_tol),
        # An unconverged optimizer can only overstate the margin, so a
        # "feasible" verdict from an unconverged run is not trustworthy
        # while an "infeasible" one still is.
        "verdict_trustworthy": bool(res["converged"] or margin <= zero_tol),
    }
    if d_model is not None:
        out["d_model"] = int(d_model)
        out["wendel_p"] = wendel_probability(n, d_model)
        out["wendel_says_certain"] = bool(d_model > n)
    return out


def hemisphere_profile(activations_or_grams, d_model: int = None) -> dict:
    """
    Per-layer cone-condition profile for one run.

    activations_or_grams : (n_layers, n_tokens, d) unit-norm activations,
                           or a list of (n, n) Gram matrices.

    The layer at which the margin first crosses zero — if it does — is the
    depth at which the network leaves the regime where Lemma 6.4 forces
    exponential collapse. That is the number this sub-experiment exists to
    produce; a per-run boolean would not have been worth the module.
    """
    arr = np.asarray(activations_or_grams)
    if arr.ndim == 3 and arr.shape[1] != arr.shape[2]:
        grams = [arr[l] @ arr[l].T for l in range(arr.shape[0])]
    elif arr.ndim == 3:
        # Square middle/last dims: ambiguous between (L, n, d) with n == d
        # and a stack of Grams. Resolve by symmetry rather than guessing.
        grams = ([arr[l] for l in range(arr.shape[0])]
                 if np.allclose(arr[0], arr[0].T, atol=1e-6)
                 else [arr[l] @ arr[l].T for l in range(arr.shape[0])])
    else:
        raise ValueError(f"expected 3-D input, got shape {arr.shape}")

    per_layer = [hemisphere_test(g, d_model=d_model) for g in grams]
    margins = np.array([p["margin"] for p in per_layer])
    feas = np.array([p["feasible"] for p in per_layer])

    first_infeasible = int(np.argmin(feas)) if not feas.all() else -1
    return {
        "per_layer": per_layer,
        "margins": margins,
        "all_feasible": bool(feas.all()),
        "first_infeasible_layer": first_infeasible,
        "n_infeasible_layers": int((~feas).sum()),
        "min_margin": float(margins.min()),
        "min_margin_layer": int(np.argmin(margins)),
        "layer0_margin": float(margins[0]),
        "final_margin": float(margins[-1]),
    }
