"""
core/precision_policy.py — P2 and item 13, which are the same failure on
opposite sides of the forward pass.

P2 (weights).  Pythia ships fp16. A genuinely real eigenvalue pair,
perturbed at fp16 epsilon, splits into a complex pair with a tiny imaginary
part — exactly the regime where `rotational_fraction`'s relative tolerance
counts it as rotation. The "84-97% complex" headline may be a threshold
artifact.

Item 13 (activations).  `models.extract_activations` runs the forward pass
under bf16 autocast, then casts to float32. bf16 carries an 8-bit mantissa,
roughly 3 significant decimal digits — a worse noise floor than the fp16
weight storage P2 already flags. Everything downstream inherits it.

Why this is a surface and not a re-run
--------------------------------------
The knob is not precision alone. `p2b_imaginary.layernorm_jacobian.
rotational_fraction` counts a dimension complex when

    |Im lambda| > tol * (|Re lambda| + eps),      tol = 0.01

which is a RELATIVE criterion, and a relative criterion is precisely what an
fp16-epsilon split of a real pair defeats: the split is small in absolute
terms and unbounded in ratio when |Re lambda| is also small. So the object
to report is the value over the joint (tol, perturbation) grid. A single
float64 re-run answers half the question and looks like it answered all of
it.

Costs nothing to run now
-------------------------
Weights only. No activations, no forward passes, no checkpoints to
download beyond what Phase 2 already caches. P2 is the one policy item that
can be closed today, independent of the frame card.

What this does NOT reopen
--------------------------
Phase 2b's causal conclusion. `elim_signed = 1.0` and `elim_rotation = 0.0`
in all 35/35 runs are unaffected by how the complex fraction is counted —
the rescaled-frame test does not consult the tolerance. State that next to
any re-run so the result is not read as reopening `rotation_neutral`.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from core.metrics import _as_numpy
from core.nulls import sigma_from_null

#: Relative tolerances to sweep around the shipped 0.01.
DEFAULT_TOLS: tuple = (0.001, 0.003, 0.01, 0.03, 0.1)

#: Fraction of the shipped value the observed number may move across the
#: grid before the headline is called sensitive rather than stable.
STABILITY_BAND: float = 0.05

SHIPPED_TOL: float = 0.01


# ---------------------------------------------------------------------------
# Precision round-trips — the real storage effect, not a synthetic jitter
# ---------------------------------------------------------------------------

def fp16_roundtrip(x) -> np.ndarray:
    """float64 -> float16 -> float64. What the checkpoint actually did."""
    a = _as_numpy(x).astype(np.float64, copy=False)
    return a.astype(np.float16).astype(np.float64)


def bf16_roundtrip(x) -> np.ndarray:
    """
    float64 -> bfloat16 -> float64, round-to-nearest-even on the mantissa.

    numpy has no bfloat16, so this truncates a float32 to its top 16 bits
    with the standard rounding bias. Non-finite inputs pass through the same
    path as any other bit pattern; screen for them upstream if that matters.
    """
    a = np.ascontiguousarray(_as_numpy(x).astype(np.float32))
    u = a.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    trunc = ((u + bias) & np.uint32(0xFFFF0000)).astype(np.uint32)
    return trunc.view(np.float32).astype(np.float64)


def relative_jitter(x, eps: float, rng=None) -> np.ndarray:
    """
    Multiplicative uniform jitter at relative magnitude `eps`. Use when the
    question is "how much perturbation would it take", as opposed to
    "what did this dtype actually do" — the round-trips answer the latter
    and are the ones the policy needs.
    """
    rng = rng if rng is not None else np.random.default_rng()
    a = _as_numpy(x).astype(np.float64, copy=False)
    return a * (1.0 + eps * rng.uniform(-1.0, 1.0, size=a.shape))


PERTURBATIONS = {
    "none": lambda M, rng: _as_numpy(M).astype(np.float64, copy=False),
    "fp16_roundtrip": lambda M, rng: fp16_roundtrip(M),
    "bf16_roundtrip": lambda M, rng: bf16_roundtrip(M),
}


# ---------------------------------------------------------------------------
# P2 — the (tol, perturbation) surface
# ---------------------------------------------------------------------------

def _default_frac_fn() -> Callable:
    """
    Lazy import so `core` does not acquire a dependency on a phase package.
    Pass `frac_fn` explicitly from the caller in phase code.
    """
    from p2b_imaginary.layernorm_jacobian import rotational_fraction
    return rotational_fraction


def complex_fraction_surface(M,
                             tols: Sequence[float] = DEFAULT_TOLS,
                             perturbation: str = "fp16_roundtrip",
                             n_draws: int = 32,
                             jitter_eps: float | None = None,
                             frac_fn: Callable | None = None,
                             rng=None) -> dict:
    """
    Complex-energy fraction of one matrix over the tolerance sweep, at
    float64 baseline and under a repeated perturbation.

    A deterministic round-trip gives an identical result on every draw, so
    `n_draws` only bites when `jitter_eps` is supplied — in that case the
    round-trip is replaced by random relative jitter at that magnitude and
    the spread is a real distribution. Both are reported through
    `core.nulls.sigma_from_null`, so a precision claim lands in a status
    table in the same shape as every other claim in this project.

    Returns dict(tols, baseline, perturbed_mean, sigma, moved_fraction).
    """
    frac = frac_fn or _default_frac_fn()
    rng = rng if rng is not None else np.random.default_rng(0)
    A = _as_numpy(M).astype(np.float64, copy=False)

    if jitter_eps is not None:
        draw = lambda: relative_jitter(A, jitter_eps, rng)
        n = int(n_draws)
    else:
        if perturbation not in PERTURBATIONS:
            raise ValueError(
                f"complex_fraction_surface: perturbation must be one of "
                f"{tuple(PERTURBATIONS)}, got {perturbation!r}"
            )
        draw = lambda: PERTURBATIONS[perturbation](A, rng)
        n = 1  # deterministic; more draws would only repeat the same number

    tols = [float(t) for t in tols]
    baseline, perturbed, summaries = [], [], []
    for t in tols:
        b = float(frac(A, tol=t))
        vals = np.array([float(frac(draw(), tol=t)) for _ in range(n)])
        baseline.append(b)
        perturbed.append(float(vals.mean()))
        summaries.append(sigma_from_null(b, vals) if n > 1 else
                         {"observed": b, "null_mean": float(vals.mean()),
                          "null_std": 0.0, "z_score": float("nan"),
                          "percentile": float("nan"), "n_null": int(n)})

    baseline = np.array(baseline)
    perturbed = np.array(perturbed)
    ref = baseline[tols.index(SHIPPED_TOL)] if SHIPPED_TOL in tols else baseline[0]
    denom = max(abs(float(ref)), 1e-12)

    return {
        "tols": tols,
        "shipped_tol": SHIPPED_TOL,
        "baseline": baseline.tolist(),
        "perturbed_mean": perturbed.tolist(),
        "perturbation": "jitter" if jitter_eps is not None else perturbation,
        "jitter_eps": jitter_eps,
        "sigma": summaries,
        "tol_span": float((baseline.max() - baseline.min()) / denom),
        "precision_span": float(np.max(np.abs(perturbed - baseline)) / denom),
        "reference_value": float(ref),
    }


def precision_verdict(surface: dict, band: float = STABILITY_BAND) -> dict:
    """
    Turn the surface into the sentence that goes in status-2b.md.

    stable               neither the tolerance sweep nor the precision
                         round-trip moves the value by more than `band`.
                         The 84-97% headline stands as written.
    threshold_sensitive  the tolerance sweep moves it. The number is a
                         property of the counting rule, and must be quoted
                         with its tol.
    precision_sensitive  the round-trip moves it. The number is a property
                         of fp16 storage; re-derive from float64.
    both                 quote neither without both qualifiers.
    """
    t = surface["tol_span"] > band
    p = surface["precision_span"] > band
    name = ("both" if (t and p) else
            "threshold_sensitive" if t else
            "precision_sensitive" if p else "stable")
    return {
        "verdict": name,
        "tol_span": surface["tol_span"],
        "precision_span": surface["precision_span"],
        "band": band,
        "causal_conclusion_affected": False,
        "note": ("Phase 2b's rotation_neutral result does not consult this "
                 "tolerance; elim_signed = 1.0 stands regardless of the "
                 "verdict above."),
    }


def analyze_ov_precision(ov_list: Sequence, layer_names: Sequence[str] | None = None,
                         **kw) -> dict:
    """Run the surface over every layer's OV and aggregate the verdict."""
    names = list(layer_names) if layer_names is not None else \
        [f"layer_{i}" for i in range(len(ov_list))]
    per_layer, verdicts = {}, []
    for name, M in zip(names, ov_list):
        s = complex_fraction_surface(M, **kw)
        v = precision_verdict(s)
        per_layer[name] = {"surface": s, "verdict": v}
        verdicts.append(v["verdict"])
    worst = ("both" if "both" in verdicts else
             "threshold_sensitive" if "threshold_sensitive" in verdicts else
             "precision_sensitive" if "precision_sensitive" in verdicts else
             "stable")
    return {"per_layer": per_layer, "overall_verdict": worst,
            "n_layers": len(names)}


# ---------------------------------------------------------------------------
# Item 13 — the activation-side noise floor
# ---------------------------------------------------------------------------

def metric_under_precision(X, metric_fn: Callable,
                           precisions: Sequence[str] = ("float64", "fp16_roundtrip",
                                                        "bf16_roundtrip")) -> dict:
    """
    One metric evaluated on the same activations at several precisions.

    `relative_spread` is the number that belongs next to any claim resting
    on a small difference between two activation-derived quantities: if the
    difference being claimed is inside the spread, the claim is inside the
    noise floor and the run should be repeated with autocast disabled.

    The mitigation is cheap and worth defaulting to. One prompt battery is
    not a training run; compute is not the constraint here.
    """
    A = _as_numpy(X).astype(np.float64, copy=False)
    table = {}
    for p in precisions:
        if p == "float64":
            table[p] = float(metric_fn(A))
        elif p in PERTURBATIONS:
            table[p] = float(metric_fn(PERTURBATIONS[p](A, None)))
        else:
            raise ValueError(f"metric_under_precision: unknown precision {p!r}")
    vals = np.array(list(table.values()), dtype=np.float64)
    ref = max(abs(table.get("float64", float(vals[0]))), 1e-12)
    return {
        "values": table,
        "absolute_spread": float(vals.max() - vals.min()),
        "relative_spread": float((vals.max() - vals.min()) / ref),
    }


def precision_summary_lines(result: dict) -> list:
    return [
        "Precision policy (P2 / item 13):",
        f"  layers analysed : {result.get('n_layers', 0)}",
        f"  overall verdict : {result.get('overall_verdict', 'unknown')}",
        "  causal conclusion (rotation_neutral) is unaffected either way.",
    ]
