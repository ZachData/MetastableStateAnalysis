"""The design preflight's two closed forms (`tools/design_preflight.py`; `LESSONS.md` 6).

WHY THIS EXISTS
---------------
The preflight refuses a design whose best p cannot calibrate to E ≥ 1/α, and states how many
null draws a rank test needs. It also refuses a planted dose whose mean is not past the null's
mean, and B2's re-gate is the case on record: |p − ½| under a uniform p. Both numbers are
closed forms, checked here.

WHAT IT SHOWS
-------------
1. **The floor.** With e = κ p^(κ−1) and a rank p of 1/(n + 1), E_max = κ (n + 1)^(1−κ), so
   E_max ≥ 1/α exactly when n ≥ (1/(ακ))^(1/(1−κ)) − 1. At α = 1/20, κ = 1/2 that is n ≥ 1599
   (E 20 exactly). 500 draws (`P-S1`'s default) give E = √501 / 2 ≈ 11.2.
2. **The two-sided dose.** For p ~ U(0, 1), E|p − ½| = ¼ exactly, so a set held at the 70th
   percentile scores 0.2 < ¼ and sits below the null's mean (B2's re-gate,
   `tools/math_checks/b2_two_sided_floor.py`).

WHAT IT DOES NOT PROVE
----------------------
- That a design's null is valid; the floor assumes a valid rank p.
- That 1599 draws suffice for a test other than a single rank p (a bootstrap, a combined p, a
  multiple-comparison correction each have their own floor; the tool takes `p_floor` for them).
- Anything about the 0.9, 5 % and 80 % thresholds, which are placed, not derived.
"""
import sympy as sp

from tools.design_preflight import min_null_draws


def main():
    n, k, a = sp.symbols("n kappa alpha", positive=True)
    e_max = k * (n + 1) ** (1 - k)
    alpha, kappa = sp.Rational(1, 20), sp.Rational(1, 2)
    need = sp.solve(sp.Eq(e_max.subs({k: kappa}), 1 / alpha), n)
    assert need == [1599], need
    assert e_max.subs({k: kappa, n: 1599}) == 20
    assert e_max.subs({k: kappa, n: 1598}) < 20
    assert min_null_draws(0.05, 0.5) == 1599
    e500 = e_max.subs({k: kappa, n: 500})
    assert e500 == sp.sqrt(501) / 2 and float(e500) < 20
    print(f"floor: n >= {need[0]} null draws for E >= 20; 500 draws give E = {float(e500):.3g}")

    p = sp.symbols("p")
    mean = sp.integrate(sp.Abs(p - sp.Rational(1, 2)), (p, 0, 1))
    assert mean == sp.Rational(1, 4)
    assert sp.Abs(sp.Rational(7, 10) - sp.Rational(1, 2)) < mean
    print("two-sided: E|p - 1/2| = 1/4; the 70th percentile scores 1/5, below it")
    print("PASS (2 checks)")


if __name__ == "__main__":
    main()
