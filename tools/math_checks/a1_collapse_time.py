"""Track A, A1: the collapse time t* of eq. (6.9) and the step estimators A1 reads against it.

Checks (sympy, exact unless marked numeric):

1. At beta = 0, (6.9) reduces to dg/dt = 2 (1 - g)((n - 1) g + 1) / n, whose solution from
   g(0) = 0 gives t*(g) = (1/2) log(((n - 1) g + 1) / (1 - g)) in closed form. Numeric beside:
   t*(0.9) = 4.17 at n = 467 and 4.91 at n = 2041, the values `p1c_frames/gamma_ode.py` returns.
2. Fixed points of (6.9) on [-1/(n-1), 1]: g = 1 (collapse) and g = -1/(n-1) (the simplex),
   at every beta.
3. Large beta: for 0 <= g < 1, e^{beta (1 - g)} * rhs(g) -> 2 (1 - g)((n - 1) g + 1) as
   beta -> oo. The speed away from g = 0 falls like e^{-beta}, so t* grows exponentially in
   beta, and "t* is near-invariant in beta" (`integration_time.py`'s docstring) holds only on a
   bounded range (numeric, `design-A.md` "t* before any output": up to beta ~ 5.6).
4. The projected step. For a block move dx = h X + xi with xi orthogonal to X:
   h_proj = <dx, X> / |X|^2 = h exactly, while h_calibrated = |dx| / |X| = sqrt(h^2 + |xi|^2/|X|^2)
   >= h, with equality iff xi = 0. So h_calibrated counts motion off the field as time.

What this does NOT prove: that a real cloud starts pairwise orthogonal (Theorem 6.8's premise),
that d >= d*(n, beta) holds (Theorem 6.9; it cannot for the long passages, n > d = 1024), that
one beta stands for a layer of 16 heads with different beta, or that an MLP's move belongs in the
sum. Those are design choices in `trackA_force/design-A.md`, not consequences of this algebra.
"""

import sympy as sp


def main() -> int:
    g, t, n, b, h = sp.symbols("g t n beta h", real=True)
    ok = True

    rhs = 2 * sp.exp(b * g) * (1 - g) * ((n - 1) * g + 1) / (sp.exp(b) + (n - 1) * sp.exp(b * g))

    # 1. beta = 0 closed form
    rhs0 = sp.simplify(rhs.subs(b, 0))
    target0 = 2 * (1 - g) * ((n - 1) * g + 1) / n
    c1 = sp.simplify(rhs0 - target0) == 0
    tstar = sp.log(((n - 1) * g + 1) / (1 - g)) / 2
    # t*(g) inverts g(t): dt/dg must equal 1 / rhs0, and t*(0) = 0
    c1b = sp.simplify(sp.diff(tstar, g) - 1 / rhs0) == 0 and sp.simplify(tstar.subs(g, 0)) == 0
    v467 = float(tstar.subs({n: 467, g: sp.Rational(9, 10)}))
    v2041 = float(tstar.subs({n: 2041, g: sp.Rational(9, 10)}))
    c1c = abs(v467 - 4.17) < 0.01 and abs(v2041 - 4.91) < 0.01
    print(f"1. beta=0 reduction {c1}, t*(g) inverts it {c1b}; t*(0.9) n=467 {v467:.3f}, n=2041 {v2041:.3f} {c1c}")
    ok &= c1 and c1b and c1c

    # 2. fixed points
    c2 = all(sp.simplify(rhs.subs(g, gv)) == 0 for gv in (1, -1 / (n - 1)))
    print(f"2. rhs = 0 at g = 1 and g = -1/(n-1), every beta: {c2}")
    ok &= c2

    # 3. large-beta speed: rhs * e^{beta (1 - g)} has a finite nonzero limit for 0 <= g < 1
    gg = sp.Rational(1, 3)
    lim = sp.limit((rhs * sp.exp(b * (1 - g))).subs({g: gg, n: 467}), b, sp.oo)
    expect = 2 * (1 - gg) * (466 * gg + 1)
    c3 = sp.simplify(lim - expect) == 0
    lim0 = sp.limit((rhs * sp.exp(b)).subs({g: 0, n: 467}), b, sp.oo)
    c3b = sp.simplify(lim0 - 2) == 0
    print(f"3. rhs ~ e^(-beta(1-g)) * 2(1-g)((n-1)g+1) as beta -> oo (g = 1/3: {c3}; g = 0: rhs ~ 2 e^-beta {c3b})")
    ok &= c3 and c3b

    # 4. projected against calibrated step, in an orthonormal pair (X along e1, xi along e2)
    Xn, xi = sp.symbols("Xnorm xi", positive=True)
    dx = sp.Matrix([h * Xn, xi])
    X = sp.Matrix([Xn, 0])
    h_proj = sp.simplify(dx.dot(X) / X.dot(X))
    h_cal = sp.sqrt(dx.dot(dx)) / sp.sqrt(X.dot(X))
    c4 = sp.simplify(h_proj - h) == 0
    c4b = sp.simplify(h_cal**2 - (h**2 + xi**2 / Xn**2)) == 0
    print(f"4. h_proj = h {c4}; h_calibrated^2 = h^2 + |xi|^2/|X|^2 {c4b}")
    ok &= c4 and c4b

    print("ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
