"""The resolution of unit 3's scale grid (`p1d_cluster_ensemble/scale_spectrum.py`,
`PRESENT_SPAN`; `design-1d.md` "The fresh-seed run", row "presence").

WHAT IT CHECKS
--------------
The grid is N = 40 points log-spaced from r = 0.01 to 1.5, so consecutive points
are a factor h = 150^(1/39) apart. A robust plateau needs MIN_RUN = 3 consecutive
points.
1. Any interval [a, a * s] holds at least floor(log s / log h) grid points (when
   it lies inside the grid's range). So s >= h^3 = 150^(1/13) ~ 1.4703 is sure to
   hold 3 points, and no smaller s is: an interval of span h^3 / (1 + eps)
   starting just after a grid point holds only 2.
2. Three consecutive points span h^2 = 150^(2/39) ~ 1.2930 (the "1.293" of
   `/challenge-pr` on #136, finding 2): a window of span between 1.293 and 1.470
   may or may not hold 3 points, depending on where it sits.

WHAT IT DOES NOT PROVE
----------------------
That a window measured on the 400-point grid (`planted_window`) is a window on the
40-point grid. The 400-point span is a lower bound on the true interval only if the
cut's ARI to the planted labels stays >= 0.8 between the 400-grid points that pass;
cuts are piecewise constant in r, so a merge could fall between two of them and be
undone by the next (not seen, not excluded). Nor does it show that a present window
gives a plateau: stability, (b) and continuity are separate conditions.

Run: python3 tools/math_checks/grid_resolution_span.py
"""
import sympy as sp

N, lo, hi, MIN_RUN = 40, sp.Rational(1, 100), sp.Rational(3, 2), 3
h = (hi / lo) ** sp.Rational(1, N - 1)
assert sp.simplify(h ** MIN_RUN - 150 ** sp.Rational(1, 13)) == 0
span = float(h ** MIN_RUN)
assert abs(span - 1.4703) < 1e-4, span
assert abs(float(h ** 2) - 1.2930) < 1e-4

# 1, numerically: in log units the grid is the integers (step 1). An interval
# [u, u + L] holds #{k : u <= k <= u + L} = floor(u + L) - ceil(u) + 1 points.
import math

def held(u, L):
    return math.floor(u + L) - math.ceil(u) + 1

for i in range(10_000):
    u = i / 10_000 * 7.3
    assert held(u, MIN_RUN) >= MIN_RUN            # L = 3 steps always holds 3
assert held(1e-10, MIN_RUN - 1e-9) == MIN_RUN - 1   # and less than 3 steps may hold 2
print(f"h = {float(h):.5f}; 3 points span {float(h**2):.4f}; "
      f"a window is sure to hold {MIN_RUN} points from span {span:.4f}")
