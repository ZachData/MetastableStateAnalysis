"""
core/dissipation.py — the energy-dissipation identity, and transport
distances between layer measures.

WHY THIS EXISTS
---------------
E_beta is this project's central object. Every "energy violation" is the
sign of Delta E_beta across a layer boundary. The gradient that produces
that sign has never been evaluated anywhere in the repository.

It has a closed form. `core.metrics.interaction_energy` defines

    E_beta = (1 / (2*beta*n^2)) * sum_ij exp(beta * <x_i, x_j>)

and differentiating it — the factor 2 from the symmetric double sum and
the beta from the chain rule both cancel —

    dE_beta/dx_i = (1/n^2) * sum_j exp(beta * <x_i, x_j>) * x_j

which is an unnormalised attention-weighted average of the other tokens:
literally what an attention head computes when Q^T K = I. The paper's
gradient-flow condition (Geshkovski et al. §3.4, which Phase 2d's D1
tests algebraically from the weights) is visible here as an identity on
activations rather than a hypothesis tested through a proxy.

WHAT THE IDENTITY BUYS
----------------------
On the sphere, with P_perp(u) = I - u u^T the tangential projector,

    G_i = P_perp(x_i) dE_beta/dx_i          tangential energy gradient
    v_i = P_perp(x_i) dx_i / ||x_i||        tangential velocity

    Delta E_beta  =  sum_i <G_i, v_i>  +  O(||dx||^2)

Four things follow, and they are the reason this module is worth having
rather than another scalar:

1. **Per-particle attribution.** A violation is `sum_i <G_i, v_i> > 0` —
   the network pushing uphill. The sum is over particles, so every
   violation decomposes onto specific tokens instead of being a
   layer-level event with no internal structure. This is the missing
   link between Phase 5c's "particles first" framing and Phase 1's
   energy series.

2. **An EXACT attention/FFN split.** On Pythia's parallel residual
   (`core.sublayer_streams`, use_parallel_residual=True) the identity
   dx = dx_attn + dx_ffn holds exactly, and P_perp is linear, so

       sum_i <G_i, v_i> = sum_i <G_i, v_i^attn> + sum_i <G_i, v_i^ffn>

   exactly. This is what status-2.md asks for ("re-enables the
   attn-vs-FFN energy panels") and is strictly stronger than GPT-2's
   `p2_eigenspectra/decompose.py`, which carried the sequential-ordering
   confound design-2.md documents at length.

3. **The spectral subspaces join the same identity.** Splitting dx
   through Phase 2's Schur projectors
   (`p2_eigenspectra.weights.build_subspace_projectors`) factors the
   dissipation a second way, giving a channel x subspace x particle
   decomposition. Phase 2's displacement test and Phase 1's energy
   series stop being two correlated measurements and become two
   marginals of one exact quantity.

4. **The linearisation residual is a result, not an error term.**
   `Delta E_beta - sum_i <G_i, v_i>` is the second-order piece. It
   measures whether the continuum limit the whole project assumes —
   a residual block as a forward-Euler step of an ODE — is actually
   valid at that layer. A large residual is a finding about the
   project's framing. It is returned, never swallowed.

Note also that ||v_i|| is EXACTLY Phase 1c's existing step size
h_l = ||P_perp(dx)|| / ||x||. This module is the rigorous form of a
heuristic already in use, not a competing one.

TRANSPORT
---------
The state of the paper's dynamics is a measure on the sphere, so the
natural distance between layers is Wasserstein, not a scalar summary.
Two couplings are computed and BOTH are reported:

  - identity coupling (token i to token i): an upper bound on W_2, and
    equal to the project's current step-size convention;
  - optimal coupling (exact linear assignment at these n): true W_2.

The GAP between them is a new observable: it measures how much of a
layer's displacement is tokens *swapping places* — motion that leaves
the distribution unchanged — versus genuine motion of the measure. No
existing metric in the project separates those, and they mean very
different things for a claim about clustering.

CONVENTIONS AND SCOPE
---------------------
- Everything is computed on the sphere (`core.metrics.l2_normalize`),
  matching `interaction_energy` and the paper.
- Inputs are RAW residual-stream states and RAW deltas. Normalising is
  this module's job: the exact additivity in (2) holds for raw deltas,
  and is destroyed by normalising each stream separately first.
- This module is PURE — numpy in, dicts out, no file I/O, no model
  loading — for the same reason `core/metrics.py` is. Nothing is
  registered in `core/artifacts.py` because nothing here writes a file;
  the phase driver that consumes this is what declares an artifact.
- Callers wanting the per-particle values in the canonical table should
  attach the `per_particle` arrays returned here as `extra` columns on
  a `core.particles.ParticleTable`. This module deliberately does not
  import it, to stay dependency-free.
- **Frame.** These functions do not resolve a reading frame. If you are
  asking about the operator *attention* applies, pass LN'd states via
  `core.ln_frame.frame_for_hidden_state` — Phase 2d's open item 1 is
  this exact trap, and re-deriving the off-by-one at the call site is
  how it gets got wrong.

NUMERICS
--------
On unit vectors <x_i, x_j> is in [-1, 1], so exp(beta*G) is
representable in float64 for beta up to ~700. Above that the shifted
form is used, absolute energies are refused (None with an explicit
status rather than a manufactured number — status-2b.md known-issue 5),
and `overflow_guarded` is set. Dissipation sign and attribution survive
the shift; absolute E_beta does not.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from core.metrics import l2_normalize, gram_matrix, interaction_energy

#: Above this beta, exp(beta * <x,y>) on unit vectors is not representable
#: in float64 and the shifted form is used instead.
BETA_OVERFLOW_THRESHOLD: float = 700.0

#: Rows whose tangential gradient or velocity has norm below this are
#: treated as having no defined alignment angle, rather than being given
#: an arbitrary one.
ALIGNMENT_NORM_FLOOR: float = 1e-12


# ---------------------------------------------------------------------------
# Tangential geometry
# ---------------------------------------------------------------------------

def tangential(X_hat: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Row-wise projection of W onto the tangent space of the sphere at
    X_hat: P_perp(x) w = w - <x, w> x, for unit-norm rows of X_hat.

    Linear in W — which is what makes every decomposition in this module
    exactly additive rather than approximately so.
    """
    X_hat = np.asarray(X_hat, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    radial = np.sum(X_hat * W, axis=-1, keepdims=True)
    return W - radial * X_hat


def tangential_velocity(X: np.ndarray, dX: np.ndarray) -> np.ndarray:
    """
    v_i = P_perp(x_i) dx_i / ||x_i||, the sphere-projected displacement.

    ||v_i|| is exactly Phase 1c's per-layer step size
    h_l = ||P_perp(dx)|| / ||x||, so `np.linalg.norm(v, axis=1)` is that
    quantity per particle rather than a new convention.
    """
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    X_hat = X / norms
    return tangential(X_hat, np.asarray(dX, dtype=np.float64)) / norms


# ---------------------------------------------------------------------------
# The energy gradient
# ---------------------------------------------------------------------------

def energy_gradient(X, beta: float, tangential_only: bool = True) -> dict:
    """
    Gradient of E_beta with respect to each particle position.

        dE_beta/dx_i = (1/n^2) * sum_j exp(beta * <x_i, x_j>) * x_j

    matching `core.metrics.interaction_energy`'s normalisation exactly.
    Rows of X are projected onto the sphere first.

    Parameters
    ----------
    tangential_only : project the gradient into the tangent space at each
                      particle (the default, and the correct object for
                      dynamics constrained to the sphere). The radial
                      component is not a force on the sphere; it is what
                      the normalisation removes.

    Returns
    -------
    dict with:
      grad            : (n, d) the gradient (tangential if requested)
      grad_norm       : (n,) row norms
      log_scale       : float — for the shifted branch, the natural log of
                        the factor that was divided out of `grad`.
                        0.0 on the normal branch, so `grad` is the true
                        gradient there and callers need not special-case.
      overflow_guarded: bool
      n, beta         : echoed for provenance
    """
    beta = float(beta)
    X_hat = l2_normalize(X)
    n = X_hat.shape[0]
    G = X_hat @ X_hat.T

    overflow_guarded = beta > BETA_OVERFLOW_THRESHOLD
    if overflow_guarded:
        # Shift by the max exponent and carry it analytically. Weights
        # keep their ratios, so direction and relative attribution are
        # exact; the absolute scale lives in log_scale.
        shift = float(beta * np.max(G))
        Wt = np.exp(beta * G - shift)
        log_scale = shift
    else:
        Wt = np.exp(beta * G)
        log_scale = 0.0

    grad = (Wt @ X_hat) / (n * n)
    if tangential_only:
        grad = tangential(X_hat, grad)

    return {
        "grad": grad,
        "grad_norm": np.linalg.norm(grad, axis=-1),
        "log_scale": log_scale,
        "overflow_guarded": bool(overflow_guarded),
        "n": int(n),
        "beta": beta,
    }


def _energy_or_none(X, beta: float, overflow_guarded: bool):
    """E_beta, or None when the shifted branch makes the absolute value
    unrepresentable. Never a manufactured 0.0 (status-2b.md issue 5)."""
    if overflow_guarded:
        return None
    value = interaction_energy(X, beta)
    return float(value) if np.isfinite(value) else None


# ---------------------------------------------------------------------------
# Dissipation
# ---------------------------------------------------------------------------

def dissipation(X, dX, beta: float) -> dict:
    """
    First-order energy change across one layer boundary, with the
    second-order residual reported rather than absorbed.

        first_order = sum_i <G_i, v_i>
        actual      = E_beta(x + dx) - E_beta(x)      (both on the sphere)
        residual    = actual - first_order

    A positive `first_order` is an energy violation attributed to the
    particles that produced it: `per_particle` carries the per-token
    contribution, which sums to `first_order` exactly.

    Signs are NOT clipped — status-2b.md's V2 lesson. A negative
    attribution is information.
    """
    X = np.asarray(X, dtype=np.float64)
    dX = np.asarray(dX, dtype=np.float64)
    if X.shape != dX.shape:
        raise ValueError(f"dissipation: X {X.shape} and dX {dX.shape} must match")

    g = energy_gradient(X, beta)
    v = tangential_velocity(X, dX)
    per_particle = np.sum(g["grad"] * v, axis=-1)
    first_order = float(per_particle.sum())

    e_before = _energy_or_none(X, beta, g["overflow_guarded"])
    e_after = _energy_or_none(X + dX, beta, g["overflow_guarded"])
    if e_before is None or e_after is None:
        actual = None
        residual = None
        rel_residual = None
        status = "energy_unrepresentable_at_this_beta"
    else:
        actual = e_after - e_before
        residual = actual - first_order
        denom = max(abs(actual), abs(first_order))
        rel_residual = float(residual / denom) if denom > 0 else None
        status = "ok"

    return {
        "first_order": first_order,
        "per_particle": per_particle,
        "actual_delta_E": actual,
        "residual": residual,
        "relative_residual": rel_residual,
        "step_size": float(np.linalg.norm(v, axis=-1).sum()),
        "step_size_per_particle": np.linalg.norm(v, axis=-1),
        "overflow_guarded": g["overflow_guarded"],
        "log_scale": g["log_scale"],
        "status": status,
        "beta": float(beta),
    }


def dissipation_by_channel(X, dX_attn, dX_ffn, beta: float) -> dict:
    """
    Exact additive attribution of the first-order energy change to the
    attention and FFN channels.

    Valid as an EXACT identity only where dx = dx_attn + dx_ffn holds
    exactly — Pythia / GPT-NeoX with use_parallel_residual=True. On a
    sequential architecture (GPT-2, BERT, ALBERT) the two streams are
    not co-domain-additive in this way and `sum_check` will show it, so
    the check is the guard rather than a model-name branch.

    `sum_check` is returned, not asserted away: it is the evidence that
    the split is exact on this architecture.
    """
    X = np.asarray(X, dtype=np.float64)
    dX_attn = np.asarray(dX_attn, dtype=np.float64)
    dX_ffn = np.asarray(dX_ffn, dtype=np.float64)

    g = energy_gradient(X, beta)
    grad = g["grad"]

    v_attn = tangential_velocity(X, dX_attn)
    v_ffn = tangential_velocity(X, dX_ffn)
    v_total = tangential_velocity(X, dX_attn + dX_ffn)

    per_attn = np.sum(grad * v_attn, axis=-1)
    per_ffn = np.sum(grad * v_ffn, axis=-1)
    per_total = np.sum(grad * v_total, axis=-1)

    d_attn = float(per_attn.sum())
    d_ffn = float(per_ffn.sum())
    d_total = float(per_total.sum())

    scale = max(abs(d_attn), abs(d_ffn), abs(d_total), 1e-300)
    sum_check = float(abs(d_attn + d_ffn - d_total) / scale)

    total_mag = abs(d_attn) + abs(d_ffn)
    return {
        "attn": d_attn,
        "ffn": d_ffn,
        "total": d_total,
        "per_particle_attn": per_attn,
        "per_particle_ffn": per_ffn,
        # Share of the *magnitude*, so a cancelling pair is visible as two
        # large shares rather than as one small total. None when both are
        # zero, rather than a 0.5 that reads as "evenly split".
        "attn_share": float(abs(d_attn) / total_mag) if total_mag > 0 else None,
        "ffn_share": float(abs(d_ffn) / total_mag) if total_mag > 0 else None,
        "sum_check": sum_check,
        "exact": bool(sum_check < 1e-10),
        "overflow_guarded": g["overflow_guarded"],
        "beta": float(beta),
    }


def dissipation_by_subspace(X, dX, beta: float,
                            P_attract: np.ndarray,
                            P_repulse: np.ndarray) -> dict:
    """
    Split the first-order energy change through Phase 2's attractive and
    repulsive Schur subspaces.

    P_attract / P_repulse come from
    `p2_eigenspectra.weights.build_subspace_projectors` — keys
    "schur_attract" and "schur_repulse". They are complementary
    orthogonal projectors (Z_+ Z_+^T + Z_- Z_-^T = I for orthogonal Z),
    so the split is exact, and `sum_check` reports it.

    Row-vector convention throughout this project: the value pathway is
    x @ OV, so a displacement is projected as dX @ P.
    """
    X = np.asarray(X, dtype=np.float64)
    dX = np.asarray(dX, dtype=np.float64)
    P_attract = np.asarray(P_attract, dtype=np.float64)
    P_repulse = np.asarray(P_repulse, dtype=np.float64)

    g = energy_gradient(X, beta)
    grad = g["grad"]

    v_a = tangential_velocity(X, dX @ P_attract)
    v_r = tangential_velocity(X, dX @ P_repulse)
    v_t = tangential_velocity(X, dX)

    per_a = np.sum(grad * v_a, axis=-1)
    per_r = np.sum(grad * v_r, axis=-1)
    d_a, d_r = float(per_a.sum()), float(per_r.sum())
    d_t = float(np.sum(grad * v_t, axis=-1).sum())

    scale = max(abs(d_a), abs(d_r), abs(d_t), 1e-300)
    return {
        "attractive": d_a,
        "repulsive": d_r,
        "total": d_t,
        "per_particle_attractive": per_a,
        "per_particle_repulsive": per_r,
        "sum_check": float(abs(d_a + d_r - d_t) / scale),
        "overflow_guarded": g["overflow_guarded"],
        "beta": float(beta),
    }


def gradient_flow_alignment(X, dX, beta: float) -> dict:
    """
    How close is this layer to being a Wasserstein gradient flow of
    E_beta?

        cos_i = <-G_i, v_i> / (||G_i|| ||v_i||)

    +1 everywhere is an exact gradient descent on E_beta; -1 is exact
    ascent; 0 is motion orthogonal to the energy landscape, which is the
    interesting middle case — a layer doing something E_beta cannot see.

    This is the measured counterpart of Phase 2d's D1, which tests the
    same question algebraically from the weights (Q^T K symmetric and
    V = Q^T K). D1 asks whether the head *could* be a gradient flow;
    this asks whether the layer *is* one, on the actual activations.

    The distribution is returned, not just the mean: a mean near zero is
    produced both by uniformly orthogonal motion and by half the
    particles descending while half ascend, and those are different
    findings.

    Particles whose gradient or velocity is numerically zero have no
    defined angle and are counted in `n_undefined` rather than assigned
    one. If none are defined, every statistic is None with an explicit
    status (status-2b.md known-issue 5).
    """
    X = np.asarray(X, dtype=np.float64)
    g = energy_gradient(X, beta)
    grad, v = g["grad"], tangential_velocity(X, dX)

    gn = np.linalg.norm(grad, axis=-1)
    vn = np.linalg.norm(v, axis=-1)
    defined = (gn > ALIGNMENT_NORM_FLOOR) & (vn > ALIGNMENT_NORM_FLOOR)

    cos = np.full(gn.shape, np.nan, dtype=np.float64)
    if np.any(defined):
        cos[defined] = (-np.sum(grad[defined] * v[defined], axis=-1)
                        / (gn[defined] * vn[defined]))
        # Guard against |cos| > 1 from rounding before anything downstream
        # takes an arccos of it.
        cos[defined] = np.clip(cos[defined], -1.0, 1.0)

    n_def = int(defined.sum())
    if n_def == 0:
        return {
            "per_particle": cos, "mean": None, "median": None, "std": None,
            "q10": None, "q90": None, "frac_descending": None,
            "n_defined": 0, "n_undefined": int(gn.size),
            "status": "no_particle_has_a_defined_alignment",
            "beta": float(beta),
        }

    c = cos[defined]
    return {
        "per_particle": cos,
        "mean": float(np.mean(c)),
        "median": float(np.median(c)),
        "std": float(np.std(c)),
        "q10": float(np.quantile(c, 0.10)),
        "q90": float(np.quantile(c, 0.90)),
        "frac_descending": float(np.mean(c > 0)),
        "n_defined": n_def,
        "n_undefined": int(gn.size - n_def),
        "status": "ok",
        "beta": float(beta),
    }


# ---------------------------------------------------------------------------
# Transport distances between layer measures
# ---------------------------------------------------------------------------

def _sphere(X) -> np.ndarray:
    return l2_normalize(np.asarray(X, dtype=np.float64))


def w2_identity(X, Y) -> float:
    """
    W_2 under the identity coupling (token i to token i), on the sphere.

    An upper bound on true W_2, since the identity coupling is one
    admissible transport plan among many. This is the project's existing
    convention made explicit as a coupling choice.
    """
    A, B = _sphere(X), _sphere(Y)
    if A.shape != B.shape:
        raise ValueError(f"w2_identity: shapes {A.shape} and {B.shape} differ")
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=-1))))


def w2_optimal(X, Y) -> dict:
    """
    True W_2 between two equal-weight n-point empirical measures on the
    sphere, via exact linear assignment.

    Equal atom weights make the optimal plan a permutation (Birkhoff),
    so `scipy.optimize.linear_sum_assignment` on the squared-distance
    cost is exact rather than a relaxation. O(n^3); fine at the n ~ 500
    of this project's prompts, and `sliced_w2` is the fallback above
    that.

    `swap_fraction` and the identity/optimal gap are the point of this
    function. Displacement that a permutation absorbs is tokens changing
    places, which leaves the measure untouched; only the residual after
    optimal matching is motion of the distribution.
    """
    A, B = _sphere(X), _sphere(Y)
    if A.shape != B.shape:
        raise ValueError(f"w2_optimal: shapes {A.shape} and {B.shape} differ")
    n = A.shape[0]

    # Cost for the ASSIGNMENT only. On unit rows ||a-b||^2 = 2 - 2<a,b>,
    # which avoids materialising an (n, n, d) difference. This form loses
    # precision for near-coincident points (2 - 2c cancels as c -> 1), but
    # the assignment is a comparison between entries and is insensitive to
    # an absolute error far below the gaps that decide it.
    cost = 2.0 - 2.0 * (A @ B.T)
    np.maximum(cost, 0.0, out=cost)

    rows, cols = linear_sum_assignment(cost)

    # Evaluate the two distances from DIRECT differences, not from `cost`.
    # W_2 is a square root, so it doubles the relative error of a squared
    # cost: the cancellation in `2 - 2c` puts a ~1e-8 floor under a
    # distance that should read exactly 0 for two identical measures.
    # Differencing the coordinates first has no such cancellation, and
    # costs one O(n d) pass now that the permutation is known.
    w2_opt = float(np.sqrt(np.mean(np.sum((A[rows] - B[cols]) ** 2, axis=-1))))
    w2_id = float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=-1))))

    assert w2_opt >= -1e-12, "W2 cannot be negative"
    assert w2_opt <= w2_id + 1e-9, "optimal coupling cannot beat itself"

    swap_fraction = float(np.mean(cols != rows))
    assert 0.0 <= swap_fraction <= 1.0

    return {
        "w2": w2_opt,
        "w2_identity": w2_id,
        # 1 means the whole displacement was tokens swapping places (the
        # measure did not move); 0 means none of it was. None when there
        # was no displacement to attribute.
        "swap_absorbed_fraction": (float(1.0 - w2_opt / w2_id)
                                   if w2_id > 1e-12 else None),
        "swap_fraction": swap_fraction,
        "assignment": cols,
        "n": int(n),
    }


def sliced_w2(X, Y, n_proj: int = 128, rng: Optional[np.random.Generator] = None) -> dict:
    """
    Sliced W_2: average 1-D W_2 over random directions, where 1-D optimal
    transport is just sorting. O(n_proj * n * (d + log n)) — the fallback
    when `w2_optimal`'s O(n^3) is too slow, and the practical choice for
    cross-model work.

    `seed`-free by default but accepts an explicit Generator, since a
    sliced distance is a random estimate and reporting one without its
    seed is not reproducible (`core.seeds` holds this project's
    convention).
    """
    rng = np.random.default_rng() if rng is None else rng
    A, B = _sphere(X), _sphere(Y)
    d = A.shape[1]

    dirs = rng.standard_normal((d, int(n_proj)))
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)

    pa = np.sort(A @ dirs, axis=0)
    pb = np.sort(B @ dirs, axis=0)
    per_slice = np.mean((pa - pb) ** 2, axis=0)

    return {
        "sliced_w2": float(np.sqrt(per_slice.mean())),
        "std_over_slices": float(np.sqrt(per_slice).std()),
        "n_proj": int(n_proj),
    }


def wasserstein_arc_length(trajectory, optimal: bool = True) -> dict:
    """
    Length of a layer trajectory in Wasserstein space:
    sum_l W_2(mu_l, mu_{l+1}).

    This is the coordinate-free form of Phase 1c's
    T_eff = sum_l h_l. Both couplings are returned so the two can be
    compared directly against the existing number rather than replacing
    it silently.

    `straightness` = W_2(mu_0, mu_L) / arc_length, in [0, 1]. Near 1 is a
    trajectory moving steadily in one direction; near 0 is a long path
    with little net displacement, which is what dwelling in a metastable
    state looks like measured on the measure rather than inferred from a
    clustering algorithm.

    Parameters
    ----------
    trajectory : (n_layers, n_tokens, d)
    optimal    : also compute the optimal-coupling arc length. O(L n^3);
                 set False for a quick identity-coupling pass.
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 3:
        raise ValueError(f"wasserstein_arc_length: expected (L, n, d), got {traj.shape}")
    L = traj.shape[0]
    if L < 2:
        return {
            "arc_length_identity": None, "arc_length_optimal": None,
            "endpoint_w2": None, "straightness": None, "per_step": [],
            "n_layers": int(L),
            "status": "trajectory_too_short_for_an_arc_length",
        }

    per_step = []
    for l in range(L - 1):
        step = {"layer": l, "w2_identity": w2_identity(traj[l], traj[l + 1])}
        if optimal:
            o = w2_optimal(traj[l], traj[l + 1])
            step["w2_optimal"] = o["w2"]
            step["swap_absorbed_fraction"] = o["swap_absorbed_fraction"]
        per_step.append(step)

    arc_id = float(sum(s["w2_identity"] for s in per_step))
    arc_opt = float(sum(s["w2_optimal"] for s in per_step)) if optimal else None

    endpoint = (w2_optimal(traj[0], traj[-1])["w2"] if optimal
                else w2_identity(traj[0], traj[-1]))
    reference = arc_opt if optimal else arc_id
    straightness = float(endpoint / reference) if reference > 1e-12 else None

    return {
        "arc_length_identity": arc_id,
        "arc_length_optimal": arc_opt,
        "endpoint_w2": float(endpoint),
        "straightness": straightness,
        "per_step": per_step,
        "n_layers": int(L),
        "status": "ok",
    }
