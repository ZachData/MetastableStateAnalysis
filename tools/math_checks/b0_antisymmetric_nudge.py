"""B0's ± nudge: what the antisymmetric and symmetric parts of a response carry, and to what order
the + and − displacements agree (`docs/ROADMAP.md` §7, B0; `trackB_circuit/design-B.md` "The math").

WHY THIS EXISTS
---------------
B0 nudges a state or a weight by +εd and −εd and reads a response f (a later token's state, an
energy, a loss). ROADMAP §7 says "for small ε the ± distances agree to second order, so the
antisymmetric response is the sign effect with no matching". This checks the expansion for a
smooth f with a vector-valued output, to the order the harness relies on.

WHAT IT SHOWS (f smooth at x; J = Df(x), H = D²f(x), T = D³f(x))
-------------------------------------------------------------
1. A(ε) = (f(x+εd) − f(x−εd)) / 2 = ε J d + ε³/6 T[d,d,d] + O(ε⁵): odd in ε, first order is the
   linear response, no even term at all.
2. S(ε) = (f(x+εd) + f(x−εd)) / 2 − f(x) = ε²/2 H[d,d] + O(ε⁴): even in ε, no first-order term.
3. The two displacement norms ‖f(x±εd) − f(x)‖² = ε²‖Jd‖² ± ε³⟨Jd, H[d,d]⟩ + O(ε⁴): they agree
   at order ε² and first differ at ε³, i.e. at relative order ε. So "the ± distances agree to
   second order" holds for the squared norms' leading term, and their difference is a
   curvature term, not a sign effect.
4. Swapping d → −d maps A → −A and leaves S unchanged: A is the part that knows the direction's
   sign, S the part that does not.

WHAT IT DOES NOT PROVE
----------------------
- Nothing about the size of ε at which the expansion is good. Real activations sit on
  "plateaus" (Heimersheim & Mendel 2024; `trackB_circuit/lit-B.md` §2), where the response is
  near zero below a threshold and then turns on: there the leading term is not ε J d over the
  ε range read, so B0 sweeps ε and reads the curve rather than assuming the first order.
- Nothing about LayerNorm's or softmax's smoothness in float32 near the threshold; the harness's
  NUMERICS section bounds that by reading A at ε against the run-to-run floor.
- Nothing about whether a random direction is the right null; that is the calibration gate's.
"""

import sympy as sp

eps = sp.symbols("epsilon", real=True)
# a 2-output function of a 2-vector, generic up to third order around 0
x1, x2 = sp.symbols("x1 x2", real=True)
d1, d2 = sp.symbols("d1 d2", real=True)
coeffs = sp.symbols("a0:20", real=True)


def poly(c):
    # full cubic in (x1, x2) with symbolic coefficients
    mons = [1, x1, x2, x1**2, x1 * x2, x2**2, x1**3, x1**2 * x2, x1 * x2**2, x2**3]
    return sum(ci * m for ci, m in zip(c, mons))


f = sp.Matrix([poly(coeffs[:10]), poly(coeffs[10:])])
at = lambda s: f.subs({x1: s * d1, x2: s * d2}, simultaneous=True)
f0 = at(0)
fp, fm = at(eps), at(-eps)

A = sp.expand((fp - fm) / 2)
S = sp.expand((fp + fm) / 2 - f0)
J = f.jacobian([x1, x2]).subs({x1: 0, x2: 0})
Jd = J * sp.Matrix([d1, d2])

# 1. A is odd, its first-order term is J d
for i in range(2):
    assert sp.simplify(A[i].subs(eps, -eps) + A[i]) == 0
    assert sp.simplify(sp.Poly(A[i], eps).coeff_monomial(eps) - Jd[i]) == 0
    assert sp.Poly(A[i], eps).coeff_monomial(eps**2) == 0

# 2. S is even, no first-order term; its ε² term is H[d,d]/2
Hdd = sp.Matrix([sp.expand(sum(sp.diff(f[i], a, b).subs({x1: 0, x2: 0}) * da * db
                               for a, da in ((x1, d1), (x2, d2))
                               for b, db in ((x1, d1), (x2, d2)))) for i in range(2)])
for i in range(2):
    assert sp.simplify(S[i].subs(eps, -eps) - S[i]) == 0
    assert sp.Poly(S[i], eps).coeff_monomial(eps) == 0
    assert sp.simplify(sp.Poly(S[i], eps).coeff_monomial(eps**2) - Hdd[i] / 2) == 0

# 3. squared displacement norms agree at ε², differ first at ε³ by 2⟨Jd, H[d,d]⟩ / ... sign ±
Np = sp.expand(((fp - f0).T * (fp - f0))[0])
Nm = sp.expand(((fm - f0).T * (fm - f0))[0])
diff = sp.Poly(sp.expand(Np - Nm), eps)
assert diff.coeff_monomial(eps**2) == 0
c3 = sp.simplify(diff.coeff_monomial(eps**3) - 2 * (Jd.T * Hdd)[0])
assert c3 == 0, c3
assert sp.simplify(sp.Poly(Np, eps).coeff_monomial(eps**2) - (Jd.T * Jd)[0]) == 0

# 4. d -> -d flips A, keeps S
flip = {d1: -d1, d2: -d2}
for i in range(2):
    assert sp.simplify(A[i].subs(flip, simultaneous=True) + A[i]) == 0
    assert sp.simplify(S[i].subs(flip, simultaneous=True) - S[i]) == 0

print("b0_antisymmetric_nudge: 4 checks pass (A odd, first order Jd; S even, H[d,d]/2; "
      "± squared norms agree at eps^2, differ at eps^3 by 2<Jd,H[d,d]>; d -> -d flips A only)")
