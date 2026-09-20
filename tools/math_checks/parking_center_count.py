"""Lemma C.1 of 2411.04990 -- the strong-Renyi-centre count -- and what it says
about this project's carrying-capacity finding.

WHY THIS EXISTS
---------------
`docs/readings/2411.04990.md` Sec 4.3 records the paper's Lemma C.1: in an
infinitely long sequence the average number of strong Renyi centres is the
inverse spherical cap surface area

    N(delta, d) = 1 / sigma_{d-1}(B_delta)

which "grows as 1/delta^{d-1}", with closed forms pi/delta at d = 2 and
(3 sin^2(delta/2))^{-1} at d = 3, proved for ANY spherically symmetric measure
in ANY dimension.

Two properties make it worth a check rather than a citation.

1. It SATURATES IN n. The count is a limit over an infinite sequence, so more
   tokens do not buy more centres. Phase 1's unexplained finding -- max
   simultaneously-alive clusters invariant at 50-55 across all 27 checkpoints,
   while lifespan falls 7.0 -> 4.5 and births rise 113 -> 164 -- is that shape.
2. With delta = c beta^{-1/2} it reproduces the paper's Theta(beta^{(d-1)/2})
   frequency, so the two statements in the paper are one statement.

WHAT IT DOES NOT PROVE
----------------------
- Lemma C.1 counts STRONG centres (separated from every earlier particle), not
  ordinary Renyi centres. The paper is explicit that strong centres "do not
  explain all clusters" while ordinary ones "better capture the meta-stable
  clustering effect" but move. HDBSCAN clusters are neither by definition, and
  which to compare against is a design decision, not a fact.
- The i.i.d. spherically-symmetric hypothesis is the paper's, and real token
  embeddings are neither i.i.d. nor isotropic. The inversion below is therefore
  a CONSISTENCY CALCULATION, not a measurement; it says which (d_eff, c) pairs
  could produce an observed count, not which one does.
- The count is asymptotic in sequence length. At n in [20, 512] the finite-n
  correction is unquantified here.
- Nothing here touches Lemma 5.1's stationarity bound or Theorem 5.2.

Run: python3 tools/math_checks/parking_center_count.py
"""
import numpy as np
import sympy as sp

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print(f"      {detail}")
    results.append(ok)


delta, d, beta, c = sp.symbols("delta d beta c", positive=True)

# ---------------------------------------------------------------------------
# 1. The cap area, and the paper's two closed forms
# ---------------------------------------------------------------------------
# Normalised surface measure of a geodesic cap of angular radius delta on
# S^{d-1}:  sigma(B_delta) = I_{sin^2(delta/2)}((d-1)/2, 1/2) / 2  in general.
# The paper only states d = 2 and d = 3, so those are what is checked.
#
# d = 2: S^1 is the circle of circumference 2*pi; a geodesic ball of radius
# delta is an arc of length 2*delta, so sigma = 2*delta/(2*pi) = delta/pi and
# N = pi/delta.
N_d2 = sp.pi / delta
record("d = 2: N = pi/delta  [arc of length 2*delta on a circle of circumference 2*pi]",
       sp.simplify(1 / (2 * delta / (2 * sp.pi)) - N_d2) == 0)

# d = 3: S^2 unit sphere, area 4*pi; a spherical cap of angular radius delta has
# area 2*pi*(1 - cos delta), so sigma = (1 - cos delta)/2 = sin^2(delta/2) and
# N = 1/sin^2(delta/2).  THE PAPER PRINTS (3 sin^2(delta/2))^{-1}.
N_d3_derived = 1 / sp.sin(delta / 2) ** 2
N_d3_paper = 1 / (3 * sp.sin(delta / 2) ** 2)
cap_area_frac = (2 * sp.pi * (1 - sp.cos(delta))) / (4 * sp.pi)
record("d = 3: the cap fraction is sin^2(delta/2), giving N = 1/sin^2(delta/2)",
       sp.simplify(cap_area_frac - sp.sin(delta / 2) ** 2) == 0)
record("the paper's printed d = 3 form differs from that by a factor of 3",
       sp.simplify(N_d3_paper * 3 - N_d3_derived) == 0,
       "RECORDED, NOT RESOLVED: 2411.04990 App. C.4 prints (3 sin^2(delta/2))^{-1}. "
       "Either a different normalisation of sigma is in use or it is a typo; the "
       "factor is constant in delta, so it shifts the intercept of any fit and not "
       "the exponent. Do not quote an absolute d = 3 count from either form without "
       "settling it")

# ---------------------------------------------------------------------------
# 2. The small-delta scaling, and consistency with Theta(beta^{(d-1)/2})
# ---------------------------------------------------------------------------
record("d = 3 count ~ 4/delta^2 as delta -> 0, i.e. the 1/delta^{d-1} growth",
       sp.limit(N_d3_derived * delta**2, delta, 0) == 4,
       f"limit delta^2 * N = {sp.limit(N_d3_derived * delta**2, delta, 0)}")

# delta = c * beta^{-1/2}  =>  1/delta^{d-1} = c^{-(d-1)} beta^{(d-1)/2}
lhs = (c * beta ** sp.Rational(-1, 2)) ** (-(d - 1))
rhs = c ** (-(d - 1)) * beta ** ((d - 1) / 2)
record("delta = c*beta^{-1/2} turns 1/delta^{d-1} into Theta(beta^{(d-1)/2})",
       sp.simplify(sp.powsimp(lhs / rhs, force=True) - 1) == 0,
       "so Lemma C.1 and the Sec 5 frequency are one statement, and c enters only "
       "as a constant factor -- the same reason the log-log SLOPE is convention-free "
       "(tools/math_checks/parking_scaling_slope.py)")

# ---------------------------------------------------------------------------
# 3. The count does not grow with n
# ---------------------------------------------------------------------------
record("N is independent of sequence length",
       sp.diff(N_d2, sp.Symbol("n", positive=True)) == 0,
       "Lemma C.1 is a limit over an infinite sequence: the count SATURATES. "
       "Phase 1 measures max-alive clusters invariant at 50-55 across 27 checkpoints "
       "while births rise 113 -> 164 -- a fixed capacity with rising turnover, which "
       "is the shape a saturating parking count predicts")

# ---------------------------------------------------------------------------
# 4. Consistency: which (d_eff, c) could give ~50 centres at a measured beta?
# ---------------------------------------------------------------------------
# General normalised cap fraction via the regularised incomplete beta function.
def cap_fraction(delta_val, d_val):
    """Normalised surface measure of a geodesic cap of POLAR ANGLE delta on
    S^{d-1}:  (1/2) I_{sin^2 delta}((d-1)/2, 1/2)  for delta <= pi/2, and the
    complement beyond it. The argument is sin^2(delta), NOT sin^2(delta/2) --
    getting that wrong is a factor-of-2 error at d = 2 and a factor of ~4 at
    d = 3, which is how the first draft of this file failed its own d = 2 check."""
    from scipy.special import betainc
    delta_val = np.asarray(delta_val, dtype=float)
    small = np.minimum(delta_val, np.pi - delta_val)
    half = 0.5 * betainc((d_val - 1) / 2.0, 0.5, np.sin(small) ** 2)
    return np.where(delta_val <= np.pi / 2, half, 1.0 - half)


def count(delta_val, d_val):
    return 1.0 / cap_fraction(delta_val, d_val)


# sanity against the two closed forms
record("the general cap formula reproduces d = 2 and d = 3",
       abs(count(0.3, 2) - np.pi / 0.3) < 1e-9
       and abs(count(0.3, 3) - 1.0 / np.sin(0.15) ** 2) < 1e-9,
       f"d=2: {count(0.3, 2):.6f} vs {np.pi/0.3:.6f};  "
       f"d=3: {count(0.3, 3):.6f} vs {1/np.sin(0.15)**2:.6f}")

# PROJECT.md Sec 3.40: measured beta on pythia-410m step143000, scaled
# convention, median 0.50 (IQR [0.26, 0.75]); unscaled is 8x, so ~4.0.
TARGET = 52.5                    # midpoint of Phase 1's 50-55
print("      --- consistency table: c that would give ~52.5 strong centres ---")
rows = []
for beta_val, conv in ((0.50, "scaled"), (4.0, "unscaled, x8")):
    for d_val in (2, 3, 5, 8, 12, 22):
        lo, hi = 1e-4, np.pi - 1e-4
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if count(mid, d_val) > TARGET:
                lo = mid
            else:
                hi = mid
        delta_val = 0.5 * (lo + hi)
        c_val = delta_val * np.sqrt(beta_val)
        rows.append((conv, beta_val, d_val, delta_val, c_val))
        flag = "  <-- c > 1, the regime Lemma 5.1 needs" if c_val > 1 else ""
        print(f"      beta={beta_val:4.2f} ({conv:12s})  d_eff={d_val:3d}  "
              f"delta={delta_val:.4f} rad  c={c_val:.3f}{flag}")

any_valid = any(r[4] > 1 for r in rows)
record("some (convention, d_eff) pairs land in the c > 1 regime and some do not",
       any_valid and not all(r[4] > 1 for r in rows),
       "so an observed count of ~50 is NOT consistent with every (beta convention, d_eff) "
       "pair -- the carrying-capacity number constrains the pair jointly. This is a "
       "consistency calculation under the paper's i.i.d. isotropic hypothesis, not a "
       "measurement; see the docstring")

print()
print(f"{sum(results)}/{len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)
