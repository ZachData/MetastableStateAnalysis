"""B2's member statistic: the floor of the two-sided |p − ½| (`trackB_circuit/design-B2.md` "The
rule" 2; the re-gate of Blocked 62 (a), `trackB_circuit/status-B.md` "B2 unit 1b").

WHY THIS EXISTS
---------------
The rule's T is the mean over members and steps of |p − ½|, p a cell's midrank percentile among its
step's 384 heads, ranked against layer-matched random sets. `/challenge-pr` on #219 asked for G3's
dose curve to start at the 70th percentile. This check shows where the statistic's floor sits, so
that dose can be read correctly.

WHAT IT SHOWS
-------------
1. **The null mean is exactly ¼.** For n even, the midrank percentiles of n distinct values are
   (k + ½)/n, k = 0 … n − 1, and their mean |p − ½| is exactly 1/4 (shown for symbolic even n, and
   at n = 384). For continuous p ~ U(0, 1) it is also 1/4.
2. **So a set held at the q-th percentile at every step scores |q − ½|.** That is above the null
   mean only when q > ¾ or q < ¼. A set at the 70th scores 0.20 < ¼, so it ranks *below* a typical
   random set, and two-sided T cannot see it however many steps are averaged. The 75th is the
   floor exactly (in expectation).
3. **Averaging steps shrinks the spread, not the floor.** For independent p ~ U(0, 1),
   Var|p − ½| = 1/48. With m cells averaged (6 members × 9 steps = 54) the sd is (1/48 / m)^½, so
   a set above ¾ is seen once its excess clears a few sd; at m = 54 the 99th percentile of the
   null mean is about ¼ + 2.33 · 0.0196 ≈ 0.296, under the 80th's 0.30.

WHAT IT DOES NOT SHOW
---------------------
- The real null draws one head per layer from the 15 non-members, without replacement across
  steps, and percentiles are correlated across steps (#219 finding 1 quotes −0.11 to 0.18 at 512
  vs 143000). The 0.296 in 3 is the independent-cells approximation, not the gate's null.
  G3m's 9-step run measures the real one.
- Nothing here says a real member sits at any percentile.
"""

import sympy as sp


def main() -> None:
    m, k = sp.symbols("m k", integer=True, positive=True)
    n = 2 * m
    # 1. Σ_k |2k + 1 − n| over k = 0 … n − 1, split at k = m: both halves are 1 + 3 + … + (2m − 1) = m².
    lower = sp.summation(n - 1 - 2 * k, (k, 0, m - 1))
    upper = sp.summation(2 * k + 1 - n, (k, m, n - 1))
    assert sp.simplify(lower - m**2) == 0 and sp.simplify(upper - m**2) == 0
    mean_abs = sp.simplify((lower + upper) / (2 * n) / n)        # |(k + ½)/n − ½| = |2k + 1 − n| / 2n
    assert mean_abs == sp.Rational(1, 4), mean_abs
    exact384 = sum(sp.Abs(sp.Rational(2 * j + 1 - 384, 2 * 384)) for j in range(384)) / 384
    assert exact384 == sp.Rational(1, 4)

    p = sp.symbols("p", real=True)
    mu = sp.integrate(sp.Abs(p - sp.Rational(1, 2)), (p, 0, 1))
    var = sp.integrate((sp.Abs(p - sp.Rational(1, 2)) - mu) ** 2, (p, 0, 1))
    assert mu == sp.Rational(1, 4) and var == sp.Rational(1, 48)

    # 2. A set at q scores |q − ½|: above ¼ only for q > ¾ (or q < ¼).
    for q, above in ((sp.Rational(7, 10), False), (sp.Rational(3, 4), False), (sp.Rational(4, 5), True)):
        assert (sp.Abs(q - sp.Rational(1, 2)) > mu) == above

    # 3. The independent-cells 99th percentile at m = 54 cells.
    sd = sp.sqrt(var / 54)
    q99 = float(mu + 2.326 * sd)
    assert 0.29 < q99 < 0.30, q99
    print(f"null mean 1/4 (n even, n = 384, U(0,1)); var 1/48; 54-cell sd {float(sd):.4f}, "
          f"99th ≈ {q99:.3f}; 70th → 0.20, 75th → 0.25, 80th → 0.30")


if __name__ == "__main__":
    main()
