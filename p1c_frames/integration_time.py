"""
p1c_frames/integration_time.py — sub-experiment A: how far does the network
actually integrate the paper's dynamics?

THE ARGUMENT

The paper's model is a continuous-time ODE on the sphere,

    dx_i/dt = X[mu](x_i) = P^perp_{x_i} ( sum_j a_ij x_j ).

A transformer residual block is a forward-Euler step of that ODE. The step
size is NOT 1 — it is however far the block actually moves the state:

    x_{l+1} = x_l + h_l * X(x_l)      =>      h_l = ||dx_l^tangential|| / ||X(x_l)||

and the network's EFFECTIVE INTEGRATION TIME on a prompt is T_eff = sum_l h_l.

This is the quantity that turns "the trained model resists collapse" from a
comparison against t = infinity into a comparison against a specific finite
number. Integrating eq. (6.9) at n = 467 puts gamma = 0.9 at t* ~ 4.2,
near-invariant in beta. So:

    T_eff >~ t*  and no collapse   =>  the network integrates far enough,
                                       and something is actively resisting.
    T_eff <<  t*                   =>  the network never runs the dynamics
                                       long enough to collapse, and the
                                       "resistance" is partly depth.

Prediction P-γ2 is registered on the second outcome.

THREE STEP-SIZE DEFINITIONS, AND WHY THE CHOICE DECIDES THE ANSWER

MATH.md sec. 8 defines h_l = ||P^perp(dx_l)|| / ||x_l||. That is the
*displacement on the sphere*, which is the numerator above — it is the step
size only if ||X|| = 1. The paper's own bound is ||X|| <= 1 (softmax rows
sum to 1 and ||x_j|| = 1), with equality only for a fully collapsed cloud;
for a spread-out cloud ||X|| is far below 1.

So h_displacement systematically UNDERSTATES T_eff, by exactly the factor
that the field is weaker than its bound. That bias points toward "the
network never integrates far enough," which is the direction that would
make us wrongly conclude our headline result is an artifact of depth. It is
the one error here that is expensive, so all three are computed:

    h_displacement  ||P^perp dx|| / ||x||           (MATH.md sec. 8 as written)
    h_calibrated    h_displacement / ||X(x_l)||     (the actual Euler step)
    h_attn_only     as h_calibrated, but with dx restricted to the attention
                    branch. The paper's dynamics have NO FFN — sec. 2 writes
                    the feed-forward layer down and then excludes it, and
                    every theorem in Parts 1-2 is single-head, no-FFN. Using
                    the full block delta credits the ODE with motion produced
                    by a term that is not in it. Pythia's parallel residual
                    (out = x + attn(ln1 x) + mlp(ln2 x)) makes the split
                    exact, with no ordering confound.

Report all three. If they disagree by less than the margin against t*, the
conclusion is robust; if they straddle it, the conclusion is a definition.

COST: [R]. dx_l = x_{l+1} - x_l is recoverable from activations.npz alone
(`raw = norms[..., None] * activations`), for parallel and sequential
residual alike, since HF hidden_states are the residual stream at block
boundaries. The attention-only split additionally needs sublayer streams.
No forward passes.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# The paper's vector field, evaluated on a measured configuration
# ---------------------------------------------------------------------------

def sa_field(U: np.ndarray, beta: float, causal: bool = True) -> np.ndarray:
    """
    X[mu](x_i) for the (SA) dynamics at a measured configuration.

    U      : (n, d) UNIT-NORM rows — the configuration on S^{d-1}
    beta   : inverse temperature. Use the measured beta_eff, not a swept
             value; the paper's footnote 2 is explicit that beta is a
             derived quantity of a trained head (it comes from d_h^{-1/2}
             and the typical magnitude of Q, K), not a free knob.
    causal : restrict the sum to j <= i. The paper's model is
             non-causal — every particle interacts with every other. A
             decoder-only transformer is causal, so the field the network
             actually applies is the masked one. Both are available;
             `causal=True` is the honest comparison for Pythia and is the
             default, but note it is a departure from the theory and the
             masked field is systematically weaker (fewer terms, and the
             early tokens see almost no one).

    Returns (n, d).
    """
    U = np.asarray(U, dtype=np.float64)
    n = U.shape[0]
    G = U @ U.T
    logits = beta * G
    if causal:
        mask = np.tril(np.ones((n, n), dtype=bool))
        logits = np.where(mask, logits, -np.inf)
    logits = logits - logits.max(axis=1, keepdims=True)
    A = np.exp(logits)
    A = A / A.sum(axis=1, keepdims=True)

    target = A @ U                                  # sum_j a_ij x_j
    radial = np.sum(target * U, axis=1, keepdims=True)
    return target - radial * U                      # P^perp_{x_i}


def field_magnitude(U: np.ndarray, beta: float, causal: bool = True) -> np.ndarray:
    """Per-token ||X[mu](x_i)||. Bounded above by 1 (paper, sec. 2)."""
    return np.linalg.norm(sa_field(U, beta, causal=causal), axis=1)


# ---------------------------------------------------------------------------
# Step sizes
# ---------------------------------------------------------------------------

def _unit(X: np.ndarray):
    X = np.asarray(X, dtype=np.float64)
    r = np.linalg.norm(X, axis=-1, keepdims=True)
    r = np.where(r < 1e-12, 1.0, r)
    return X / r, r[..., 0]


def tangential_displacement(x_l: np.ndarray, x_next: np.ndarray) -> np.ndarray:
    """
    Per-token || P^perp_{x_l} (x_{l+1} - x_l) || / ||x_l||.

    Equivalently, to first order, the displacement of the UNIT vector:
    with u = x/r, du = P^perp_x(dx)/r. Computing it in the projected form
    rather than as ||u_{l+1} - u_l|| is deliberate — the two agree only to
    first order, and the projected form is the one that isolates motion
    ALONG the sphere from residual-stream norm growth, which is not motion
    on the sphere at all and which trained transformers do a great deal of.
    """
    x_l = np.asarray(x_l, dtype=np.float64)
    dx = np.asarray(x_next, dtype=np.float64) - x_l
    r = np.linalg.norm(x_l, axis=-1)
    r = np.where(r < 1e-12, 1.0, r)
    u = x_l / r[:, None]
    radial = np.sum(dx * u, axis=1, keepdims=True)
    tang = dx - radial * u
    return np.linalg.norm(tang, axis=1) / r


def step_sizes(raw_states: np.ndarray, beta: float,
               attn_delta: np.ndarray = None,
               causal: bool = True) -> dict:
    """
    Per-layer step sizes for one run.

    raw_states : (n_layers, n_tokens, d) RAW residual stream (i.e.
                 norms[..., None] * activations from activations.npz, NOT
                 the unit-norm array — the norm carries the denominator).
    beta       : measured beta_eff for this run.
    attn_delta : (n_layers-1, n_tokens, d) attention-branch contribution to
                 each block's delta, if sublayer streams are available.
                 Optional; h_attn_only is nan without it.

    Returns dict of (n_layers-1,) arrays plus scalars:
        h_displacement, h_calibrated, h_attn_only  — per layer boundary
        field_mag                                   — mean ||X|| per layer
        T_eff_displacement, T_eff_calibrated, T_eff_attn_only
    """
    raw_states = np.asarray(raw_states, dtype=np.float64)
    L = raw_states.shape[0]
    if L < 2:
        raise ValueError("need at least two layers to form a step")

    h_disp = np.empty(L - 1)
    h_cal = np.empty(L - 1)
    h_attn = np.full(L - 1, np.nan)
    fmag = np.empty(L - 1)

    for l in range(L - 1):
        x_l = raw_states[l]
        U, _ = _unit(x_l)

        d_tan = tangential_displacement(x_l, raw_states[l + 1])
        mag = field_magnitude(U, beta, causal=causal)

        h_disp[l] = float(np.mean(d_tan))
        fmag[l] = float(np.mean(mag))
        # Per-token ratio then averaged, not ratio of averages: a few
        # sink tokens with tiny field magnitude would otherwise dominate
        # the denominator of a ratio-of-means and inflate the step.
        safe = np.where(mag < 1e-9, np.nan, mag)
        h_cal[l] = float(np.nanmean(d_tan / safe))

        if attn_delta is not None:
            x_attn_next = x_l + np.asarray(attn_delta[l], dtype=np.float64)
            d_tan_a = tangential_displacement(x_l, x_attn_next)
            h_attn[l] = float(np.nanmean(d_tan_a / safe))

    return {
        "h_displacement": h_disp,
        "h_calibrated": h_cal,
        "h_attn_only": h_attn,
        "field_mag": fmag,
        "T_eff_displacement": float(np.nansum(h_disp)),
        "T_eff_calibrated": float(np.nansum(h_cal)),
        "T_eff_attn_only": float(np.nansum(h_attn)),
        "n_layers": int(L),
        "beta": float(beta),
        "causal": bool(causal),
    }


def unit_steps(U: np.ndarray, U_next: np.ndarray, X: np.ndarray, lift: str = "tangent") -> dict:
    """
    Per-token step sizes for a move between two UNIT configurations, in
    whatever frame the caller built them (Track A: unit LN1 rows of the
    reading block at both ends, `trackA_force/design-A.md` "A1a").

    U, U_next : (n, d) unit rows, one token per row, same order
    X         : (n, d) the field evaluated on U (`sa_field`)

    With Δ_tan the move's part tangent to the sphere at U:
        disp = |Δ_tan|                 (1c's displacement)
        cal  = |Δ_tan| / |X|           (1c's calibrated step)
        proj = <Δ_tan, X> / |X|^2      (the move ALONG the field)
    For Δ = hX + ξ with ξ ⊥ X, proj = h and cal = sqrt(h² + |ξ|²/|X|²):
    the calibrated step counts motion off the field as time
    (`tools/math_checks/a1_collapse_time.py`, item 4). A token whose field
    is below 1e-9 gets NaN for cal and proj.

    ``lift``: ``tangent`` takes Δ_tan of the chord, which returns V/sqrt(1+|V|²)
    for a tangent move V followed by retraction (A1a's gate, G3: up to 6 % low
    on its largest steps); ``log`` rescales Δ_tan to the arc length
    arccos<U, U_next> (the sphere's log map, exact for motion along a great
    circle; 4.3 % at most on the same cells, bias ≈ |V|²/3); ``gnomonic`` takes
    U_next/<U_next, U> − U, which inverts one add-and-renormalise step exactly
    (a tangent V back to V; a radial part a gives V/(1+a)); NaN where
    <U_next, U> ≤ 0.
    """
    D = lifted_move(U, U_next, lift)
    X = np.asarray(X, dtype=np.float64)
    xx = np.sum(X * X, axis=1)
    safe = np.where(xx < 1e-18, np.nan, xx)
    disp = np.linalg.norm(D, axis=1)
    return {"disp": disp, "cal": disp / np.sqrt(safe),
            "proj": np.sum(D * X, axis=1) / safe}


def lifted_move(U: np.ndarray, U_next: np.ndarray, lift: str = "tangent") -> np.ndarray:
    """(n, d) per-token move tangent at U, lifted as in `unit_steps` (NaN rows where gnomonic is undefined)."""
    U = np.asarray(U, dtype=np.float64)
    U1 = np.asarray(U_next, dtype=np.float64)
    if lift == "gnomonic":
        c = np.sum(U * U1, axis=1, keepdims=True)
        D = U1 / np.where(c > 0, c, np.nan) - U
    else:
        D = U1 - U
    D = D - np.sum(D * U, axis=1, keepdims=True) * U
    if lift == "log":
        nt = np.linalg.norm(D, axis=1, keepdims=True)
        th = np.arccos(np.clip(np.sum(U * U1, axis=1, keepdims=True), -1.0, 1.0))
        D = D / np.maximum(nt, 1e-300) * th
    elif lift not in ("tangent", "gnomonic"):
        raise ValueError(f"unknown lift {lift!r}")
    return D


def drift_free_slope(U: np.ndarray, U_next: np.ndarray, X: np.ndarray, lift: str = "tangent") -> float:
    """
    The move along the field with one shared drift removed: the least-squares h in
    ``Δ_i = h·X_i + c + ξ_i`` (c one vector for all tokens), i.e.
    ``Σ⟨Δ_i − Δ̄, X_i − X̄⟩ / Σ|X_i − X̄|²`` over the rows where Δ and the field are defined (Track A,
    Blocked 47). Exact for ``Δ = hX`` (or ``hX + c``); a constant drift reads 0. A drift that
    is only nearly constant (a tangent projection, per-token norms) leaks through; A1a's gate
    G2s measures how much.
    """
    D = lifted_move(U, U_next, lift)
    X = np.asarray(X, dtype=np.float64)
    # rows `unit_steps` leaves NaN: an undefined lift, or no field (the first causal token, whose
    # log-lift move is arccos round-off ~1e-8, not a move)
    ok = np.all(np.isfinite(D), axis=1) & (np.sum(X * X, axis=1) >= 1e-18)
    if ok.sum() < 2:
        return float("nan")
    Dc = D[ok] - D[ok].mean(axis=0)
    Xc = X[ok] - X[ok].mean(axis=0)
    den = float(np.sum(Xc * Xc))
    return float(np.sum(Dc * Xc) / den) if den > 1e-18 else float("nan")


def cumulative_time(h: np.ndarray) -> np.ndarray:
    """
    T_eff(l) — the time coordinate of each layer boundary, starting at 0.

    Length n_layers, so it aligns with the per-layer `ip_mean` series:
    entry l is the integration time ELAPSED BEFORE layer l, which is what
    gamma_beta must be evaluated at to compare against layer l's measured
    inner product.
    """
    h = np.asarray(h, dtype=np.float64)
    return np.concatenate([[0.0], np.nancumsum(h)])


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def verdict(result: dict, t_star: float) -> dict:
    """
    Adjudicate P-γ2 for one run, across all three step definitions.

    The point of reporting all three is that the prediction is only
    meaningful if they agree. `robust` is False when they straddle t*,
    which means the answer is a definitional choice rather than a
    measurement and must be reported as such.
    """
    keys = ["T_eff_displacement", "T_eff_calibrated", "T_eff_attn_only"]
    vals = {k: result[k] for k in keys}
    finite = {k: v for k, v in vals.items() if np.isfinite(v)}
    if not finite:
        return {"t_star": t_star, "robust": False,
                "reading": "no finite T_eff", **vals}

    below = {k: v < t_star for k, v in finite.items()}
    robust = len(set(below.values())) == 1
    if robust and all(below.values()):
        reading = ("T_eff << t*: the network does not integrate far enough "
                   "to collapse. Compare against gamma_beta(T_eff), not "
                   "against t = infinity.")
    elif robust:
        reading = ("T_eff >= t*: the network integrates past the collapse "
                   "time and has not collapsed. Resistance is genuine and "
                   "now quantitative.")
    else:
        reading = ("Step definitions STRADDLE t*. The answer is a choice of "
                   "definition, not a measurement — report the spread, not "
                   "a verdict.")
    return {"t_star": float(t_star), "robust": bool(robust),
            "reading": reading,
            "ratio_calibrated": float(finite.get("T_eff_calibrated", np.nan) / t_star),
            **vals}
