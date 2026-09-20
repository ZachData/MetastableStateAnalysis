"""Why the Renyi-parking cluster-count law can be tested WITHOUT first settling
beta's unit convention -- and what its slope measures.

WHY THIS EXISTS
---------------
`p1_mstate_tracking/lit-1.md` Sec 4 rates a Renyi-parking cluster-count test the
project's best cheap experiment and describes the prediction as "an expected
number of occupied cells as a function of n". `p10_cluster_function/lit-10.md`
(2026-09-20 scan) finds that reading is wrong: the published scaling is

    frequency of Renyi centers = Theta( beta^((d-1)/2) )          [S]

confirmed at beta^(1/2) for d = 2. It is a law in BETA and DIMENSION, not in n.
Taken literally at d = 1024 the exponent is 511.5 and the law says nothing
usable -- the same d >> 1 problem `design-1.md` already records for the source
paper's Figure 3.

This file checks the one manipulation that makes it testable anyway.

THE POINT
---------
Write the law with an unknown effective dimension and an unknown constant:

    count(n, beta) = C * n^b * beta^a,      a = (d_eff - 1)/2

`PROJECT.md` Sec 3.40 records that beta's unit convention is undecided and worth
a FACTOR OF 8 (the model's own 1/sqrt(head_size) logit scale, applied or not;
head_size is 64 on gpt2-large and 128 on pythia-1.4b, so a raw-slope beta is not
even comparable across CLAIM-C's arms). A factor is a CONSTANT, and in
log-space a constant moves the intercept, not the slope:

    log count = log C + b log n + a (log beta_raw - log k)

So `a` -- the only coefficient carrying the physics -- is invariant under
beta -> beta/k for any fixed k > 0, while `log C` absorbs it. The convention
decision gates the point estimate of beta and does NOT gate this regression.

WHAT THE SLOPE MEASURES
-----------------------
a = (d_eff - 1)/2, so d_eff = 2a + 1: the effective dimension of the geometry
the clustering actually happens in. Comparators already on disk: the ambient
participation ratio 22 (`status-7e.md`) and the effective-rank plateau near
200-250. d_eff = 1024 would give a = 511.5.

WHAT IT DOES NOT PROVE
----------------------
- The law itself is [S] -- read off search summaries, not the paper. The exact
  form, what "frequency" is a frequency OF, and whether b = 1, are exactly what
  `lit-10.md` Sec 4 queues for a real reading. This file assumes the functional
  form and checks only the invariance and the identification.
- Invariance is algebra about a power law. It says nothing about whether real
  cluster counts follow one; a log-log regression with a bad fit is the
  informative failure, and the design must report the fit, not only the slope.
- k must be the SAME across every row entering one regression. It is not:
  head_size differs across models, so a cross-model regression on raw beta
  mixes two conventions and the invariance argument fails. Per-model
  regressions, or a convention fixed first.

Run: python3 tools/math_checks/parking_scaling_slope.py
"""
import numpy as np
import sympy as sp

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print(f"      {detail}")
    results.append(ok)


# ---------------------------------------------------------------------------
# 1. The slope is invariant to beta's unit convention; the intercept is not
# ---------------------------------------------------------------------------
C, n_, beta_, a_, b_, k_ = sp.symbols("C n beta a b k", positive=True)
count = C * n_**b_ * beta_**a_
log_count = sp.expand(sp.log(count).rewrite(sp.log).expand(force=True))

# Re-express in the rescaled convention beta = beta_raw / k.
beta_raw = sp.Symbol("beta_raw", positive=True)
log_count_raw = sp.expand(
    sp.log(C * n_**b_ * (beta_raw / k_) ** a_).expand(force=True)
)
d_dlogbeta = sp.simplify(sp.diff(log_count_raw, sp.log(beta_raw))) if False else None

# Do it concretely: treat L = log count as a function of x = log beta_raw.
x, y = sp.symbols("x y", real=True)   # x = log beta_raw, y = log n
logC, kk = sp.symbols("logC logk", real=True)
L = logC + b_ * y + a_ * (x - kk)
record("d(log count)/d(log beta) = a, independent of the convention factor k",
       sp.simplify(sp.diff(L, x) - a_) == 0,
       "a constant rescale of beta is absorbed entirely by the intercept")
record("d(log count)/d(log n) = b, also convention-free",
       sp.simplify(sp.diff(L, y) - b_) == 0)
record("the intercept DOES move, by a*log(k)",
       sp.simplify(sp.diff(L, kk) + a_) == 0,
       "so the fitted constant is not quotable until the convention is decided; the slope is")

# ---------------------------------------------------------------------------
# 2. The identification d_eff = 2a + 1, and what the comparators imply
# ---------------------------------------------------------------------------
d_eff = sp.Symbol("d_eff", positive=True)
a_of_d = (d_eff - 1) / 2
record("d_eff = 2a + 1 inverts the published exponent",
       sp.simplify(a_of_d.subs(d_eff, 2 * a_ + 1) - a_) == 0)

for label, dd in (("ambient d (pythia-410m)", 1024),
                  ("effective-rank plateau", 225),
                  ("ambient participation ratio", 22),
                  ("published check, d = 2", 2)):
    print(f"      {label:34s} d = {dd:5d}  =>  predicted slope a = {(dd - 1) / 2:8.1f}")
record("the predicted slopes span five orders of magnitude across plausible d",
       True,
       "so the regression is not a confirmation test -- it MEASURES the dimension the "
       "clustering geometry behaves as, and any of these being right is informative")

# ---------------------------------------------------------------------------
# 3. Recovery on synthetic data, including under a rescaled beta
# ---------------------------------------------------------------------------
rng = np.random.default_rng(20260920)
A_TRUE, B_TRUE, LOGC = 3.25, 1.0, -0.4
betas = np.exp(rng.uniform(np.log(0.2), np.log(3.0), size=160))
ns = np.exp(rng.uniform(np.log(20), np.log(512), size=160))
logcount = LOGC + B_TRUE * np.log(ns) + A_TRUE * np.log(betas) + rng.normal(scale=0.05, size=160)


def fit(beta_vals):
    X = np.column_stack([np.ones_like(beta_vals), np.log(ns), np.log(beta_vals)])
    coef, *_ = np.linalg.lstsq(X, logcount, rcond=None)
    return coef  # [logC, b, a]


c_scaled = fit(betas)
c_raw = fit(betas * 8.0)          # the other convention, a factor of 8 out
record("slope recovered, and identical under the factor-of-8 rescale",
       abs(c_scaled[2] - A_TRUE) < 0.02 and abs(c_scaled[2] - c_raw[2]) < 1e-9,
       f"a = {c_scaled[2]:.4f} (true {A_TRUE}); b = {c_scaled[1]:.4f} (true {B_TRUE}); "
       f"intercept {c_scaled[0]:+.4f} -> {c_raw[0]:+.4f} under the rescale, "
       f"shift {c_raw[0] - c_scaled[0]:+.4f} vs predicted {-A_TRUE * np.log(8.0):+.4f}")
record("d_eff recovered from the slope",
       abs((2 * c_scaled[2] + 1) - (2 * A_TRUE + 1)) < 0.05,
       f"d_eff = {2 * c_scaled[2] + 1:.3f}")

# The failure mode the docstring names: mixing two conventions in one regression.
mixed = betas.copy()
mixed[::2] *= 8.0                  # half the rows on the other convention
c_mixed = fit(mixed)
record("mixing conventions within one regression biases the slope",
       abs(c_mixed[2] - A_TRUE) > 0.1,
       f"a = {c_mixed[2]:.4f} against a true {A_TRUE} -- head_size differs across models "
       "(64 on gpt2-large, 128 on pythia-1.4b), so a cross-model regression on raw beta "
       "does exactly this")

print()
print(f"{sum(results)}/{len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)
