"""
p1c_frames/design_test.py — sub-experiment F: is the trained configuration
a sharp configuration?

WHERE THIS COMES FROM

Section 9.1. With V = -I_d the interaction energy DECREASES along
trajectories, and rewriting

    E_beta[mu] = (e^b / 2b) * int int exp(-(b/2) ||x - x'||^2) dmu dmu

makes minimizing over n-atom empirical measures the optimal-point-
configuration problem. Theorem 9.2 (from Cohn-Kumar) then says any global
minimum is a SHARP CONFIGURATION in the sense of Definition 9.1 —

    m distinct pairwise inner products, and a spherical (2m-1)-design

— or the vertices of the 600-cell.

WHY THIS IS THE PHASE'S POINT

Blog 1's headline is "trained weights resist collapse." If resistance means
the trained model sits in the repulsive regime, then the paper predicts a
SPECIFIC limit geometry — not a diffuse spread, but a sharp configuration.
That is the first target geometry the project's central empirical claim has
ever had. P-S1 is registered on it: trained centroids should be closer to a
spherical t-design than step-0 centroids.

If they are, "resisting collapse" has a name. If they are not, the
repulsive-regime reading of the result needs revision.

THE TEST

On S^{d-1}, a set is a spherical t-design iff its normalized Gegenbauer
moments vanish for every degree up to t:

    Q_k = (1/n^2) * sum_ij P_k^{(d)}( <x_i, x_j> ) = 0,   1 <= k <= t

where P_k^{(d)} is the Gegenbauer/ultraspherical polynomial normalized to
P_k(1) = 1. This is the addition-theorem form: Q_k is a positive multiple of
|| sum_i Y_k(x_i) ||^2 summed over the degree-k spherical harmonics, so
Q_k >= 0 ALWAYS, and Q_k = 0 exactly when the configuration integrates every
degree-k harmonic correctly. Non-negativity is a free correctness check on
the implementation and is asserted below.

Cheap, exact, needs only the Gram matrix, and is a strong structural
signature: Q_k is not a fitted quantity with a threshold, it is zero or it
is not.

NUMERICS

scipy.special.gegenbauer(k, alpha) is unusable here: alpha = (d-2)/2 is 511
for Pythia-410M, and the coefficient representation overflows. The
normalized polynomials satisfy a stable three-term recurrence instead
(Mueller's "Legendre polynomials in d dimensions"):

    P_0 = 1,  P_1 = t,
    (k + d - 2) P_{k+1}(t) = (2k + d - 2) t P_k(t) - k P_{k-1}(t)

evaluated pointwise on the Gram entries. P_k(1) = 1 follows by induction,
which is the second correctness check asserted below.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Normalized Gegenbauer polynomials
# ---------------------------------------------------------------------------

def gegenbauer_normalized(t, k: int, d: int) -> np.ndarray:
    """
    P_k^{(d)}(t), normalized so P_k(1) = 1, evaluated elementwise.

    Stable for any d — no coefficient expansion, no gamma functions. The
    recurrence is run in float64 on the array directly.
    """
    t = np.asarray(t, dtype=np.float64)
    if k == 0:
        return np.ones_like(t)
    if k == 1:
        return t.copy()
    p_prev, p_cur = np.ones_like(t), t.copy()
    for j in range(1, k):
        p_next = ((2 * j + d - 2) * t * p_cur - j * p_prev) / (j + d - 2)
        p_prev, p_cur = p_cur, p_next
    return p_cur


def gegenbauer_moments(G: np.ndarray, t_max: int = 6, d: int = None,
                       check: bool = True) -> dict:
    """
    Q_k for k = 1..t_max from a Gram matrix.

    G : (n, n) Gram of unit-norm points
    d : ambient dimension. Required — the polynomials depend on it, and a
        design on S^1 is not a design on S^1023.

    Returns Q (array of length t_max), plus the design order t_design =
    the largest t such that Q_k < tol for all k <= t.

    Q_k >= 0 by the addition theorem. A negative value beyond float noise
    means the recurrence has been mis-indexed or d is wrong, so it is
    checked rather than assumed.
    """
    G = np.asarray(G, dtype=np.float64)
    if d is None:
        raise ValueError("d (ambient dimension) is required")
    n = G.shape[0]
    Gc = np.clip(G, -1.0, 1.0)      # float error can push |cos| past 1

    Q = np.empty(t_max, dtype=np.float64)
    for k in range(1, t_max + 1):
        Q[k - 1] = float(gegenbauer_normalized(Gc, k, d).mean())
        if check and Q[k - 1] < -1e-8:
            raise AssertionError(
                f"Q_{k} = {Q[k-1]:.3e} < 0, which the addition theorem "
                f"forbids. Check d ({d}) and the recurrence."
            )
    if check:
        one = gegenbauer_normalized(np.array([1.0]), min(t_max, 5), d)[0]
        assert abs(one - 1.0) < 1e-8, f"P_k(1) = {one}, expected 1"

    return {"Q": Q, "degrees": np.arange(1, t_max + 1), "n": int(n), "d": int(d)}


def design_order(Q: np.ndarray, tol: float) -> int:
    """
    Largest t with Q_k < tol for every k <= t. 0 if Q_1 already fails.

    The tolerance is not a free parameter — use `random_baseline_Q` to set
    it from the sampling floor at this (n, d). A fixed absolute tolerance
    would make every large-n configuration look like a design, since Q_k
    for i.i.d. uniform points is O(1/n).
    """
    for i, q in enumerate(Q):
        if q >= tol:
            return i
    return len(Q)


# ---------------------------------------------------------------------------
# The sampling floor
# ---------------------------------------------------------------------------

def random_baseline_Q(n: int, d: int, t_max: int = 6, n_trials: int = 64,
                      seed: int = 0) -> dict:
    """
    Q_k for n i.i.d. uniform points on S^{d-1}, over n_trials draws.

    THIS IS THE ONLY MEANINGFUL REFERENCE. For i.i.d. points E[Q_k] = 1/n
    exactly (the i = j diagonal terms contribute n * P_k(1) = n, and the
    off-diagonal terms have mean zero), so Q_k -> 0 as n grows for ANY
    configuration, design or not. Comparing a raw Q_k against zero, or
    against a fixed tolerance, would find "designs" everywhere at n = 512.

    The quantity to report is therefore the RATIO Q_k / Q_k^random: below 1
    means better-than-chance equidistribution at degree k, and a genuine
    t-design has ratio ~ 0.

    Returns mean, std, and the 95th percentile per degree, plus the
    theoretical 1/n for comparison.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n_trials, t_max))
    for i in range(n_trials):
        X = rng.normal(size=(n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        out[i] = gegenbauer_moments(X @ X.T, t_max=t_max, d=d, check=False)["Q"]
    return {
        "mean": out.mean(axis=0), "std": out.std(axis=0),
        "p95": np.percentile(out, 95, axis=0),
        "theoretical": np.full(t_max, 1.0 / n),
        "n_trials": int(n_trials), "n": int(n), "d": int(d),
    }


# ---------------------------------------------------------------------------
# Definition 9.1's other half: m distinct inner products
# ---------------------------------------------------------------------------

def inner_product_modes(G: np.ndarray, bins: int = 100,
                        prominence_frac: float = 0.10,
                        window: int = 2) -> dict:
    """
    The "m distinct pairwise inner products" half of Definition 9.1.

    A sharp configuration has an off-diagonal cosine distribution that is
    multimodal with m NARROW peaks, not unimodal and broad. Reported as:
    peak count, the peak locations, and the mass concentration — the
    fraction of pairs within one bin-width of some peak, which is what
    "distinct inner products" actually means and which a peak count alone
    does not capture.

    Deliberately does not import scipy.signal.find_peaks: the prominence
    threshold there is in absolute counts and would need re-tuning per n.
    The local-max rule below is relative to the histogram's own maximum.

    TWO THINGS THIS GETS RIGHT THAT THE OBVIOUS IMPLEMENTATION DOES NOT.

    1. BOUNDARY PEAKS COUNT. An interior-only local-max scan (range(1,
       bins-1)) silently drops peaks in the first and last bin — which is
       exactly where a sharp configuration puts its mass, since inner
       products of +-1 are the extreme case of "few distinct values". The
       octahedron, which has exactly two distinct inner products (0 and
       -1) and is the sharpest configuration available in R^3, was scored
       unimodal by that version because its -1 peak sits in bin 0.

    2. A PEAK MUST BEAT A WINDOW, NOT JUST ITS NEIGHBOURS. Comparing
       against the two adjacent bins alone counts histogram noise as
       structure: 200 i.i.d. uniform points in R^20 scored five modes that
       way. The rule here requires a strict maximum over +-`window` bins,
       which is the difference between "the distribution has m distinct
       values" and "the sample is finite".
    """
    G = np.asarray(G, dtype=np.float64)
    n = G.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = G[iu]
    counts, edges = np.histogram(vals, bins=bins, range=(-1.0, 1.0))
    centres = 0.5 * (edges[:-1] + edges[1:])

    thresh = prominence_frac * counts.max() if counts.max() > 0 else 0
    peaks = []
    for i in range(bins):
        if counts[i] <= thresh:
            continue
        lo, hi = max(0, i - window), min(bins, i + window + 1)
        nbhd = counts[lo:hi]
        # Strict max over the window, with ties broken toward the left so a
        # flat plateau registers once rather than `window` times.
        if counts[i] == nbhd.max() and counts[i] > counts[lo:i].max(initial=-1):
            peaks.append(i)

    binw = edges[1] - edges[0]
    near = np.zeros(len(vals), dtype=bool)
    for i in peaks:
        near |= np.abs(vals - centres[i]) <= binw
    return {
        "n_modes": len(peaks),
        "mode_locations": [float(centres[i]) for i in peaks],
        "mass_at_modes": float(near.mean()),
        "ip_mean": float(vals.mean()),
        "ip_std": float(vals.std()),
        "unimodal": bool(len(peaks) <= 1),
    }


# ---------------------------------------------------------------------------
# The P-S1 comparison
# ---------------------------------------------------------------------------

def design_report(centroids: np.ndarray, d: int = None, t_max: int = 6,
                  n_trials: int = 64) -> dict:
    """
    Full sharp-configuration report for one set of cluster centroids.

    centroids : (m, d) — NOT required to be unit-norm; normalized here,
                because centroids of clusters on the sphere are inside it
                and the design question is about directions.

    The headline number is `Q_ratio` — Q_k against the i.i.d. floor at the
    same (m, d). `t_design_vs_random` uses the random p95 as the tolerance,
    so a configuration only counts as a t-design if it beats 95% of random
    draws at every degree up to t.
    """
    X = np.asarray(centroids, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected (m, d), got {X.shape}")
    m, dim = X.shape
    d = int(d if d is not None else dim)
    if m < 2:
        return {"n_centroids": int(m), "note": "fewer than 2 centroids"}

    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    G = Xn @ Xn.T

    mom = gegenbauer_moments(G, t_max=t_max, d=d)
    base = random_baseline_Q(m, d, t_max=t_max, n_trials=n_trials)
    ratio = mom["Q"] / np.maximum(base["mean"], 1e-30)

    return {
        "n_centroids": int(m),
        "d": d,
        "Q": mom["Q"],
        "Q_random_mean": base["mean"],
        "Q_random_p95": base["p95"],
        "Q_ratio": ratio,
        "t_design_vs_random": design_order(mom["Q"], float(base["p95"][0])),
        "t_design_strict": design_order(mom["Q"], 1e-6),
        "modes": inner_product_modes(G),
        # A sharp configuration wants BOTH: few distinct inner products
        # (multimodal, concentrated) and near-vanishing low-order moments.
        # Reporting either alone would let a merely-equidistributed cloud
        # or a merely-clumped one pass.
        "sharp_score": float(np.mean(ratio)),
    }


def adjudicate_p_s1(trained: dict, step0: dict) -> dict:
    """
    P-S1: trained centroids are closer to a spherical t-design than step-0
    centroids — low-order Gegenbauer moments smaller.
    Falsifier: no difference.

    Compares on Q_ratio rather than raw Q, because the two checkpoints can
    have different centroid counts and Q_k is O(1/m); a raw comparison
    would be reading the cluster count.
    """
    if "Q_ratio" not in trained or "Q_ratio" not in step0:
        return {"verdict": "insufficient data"}
    rt, r0 = np.asarray(trained["Q_ratio"]), np.asarray(step0["Q_ratio"])
    k = min(len(rt), len(r0))
    rt, r0 = rt[:k], r0[:k]
    better = rt < r0
    margin = float(np.mean(r0 - rt))

    if better.all():
        verdict = ("CONFIRMED — trained centroids beat step-0 at every "
                   "degree. The repulsive-limit reading has a target "
                   "geometry.")
    elif better.mean() > 0.5:
        verdict = (f"PARTIAL — trained better at {int(better.sum())}/{k} "
                   f"degrees. Report which; a low-degree-only improvement "
                   f"is equidistribution, not a design.")
    else:
        verdict = ("FALSIFIED — trained centroids are no closer to a design "
                   "than step-0. The repulsive-regime story needs revision.")
    return {"verdict": verdict, "degrees_better": better.tolist(),
            "mean_ratio_trained": float(rt.mean()),
            "mean_ratio_step0": float(r0.mean()),
            "margin": margin,
            "n_centroids_trained": trained.get("n_centroids"),
            "n_centroids_step0": step0.get("n_centroids")}
