"""Checks MATH_SPECTRAL_OT.md sec 2.1 and sec 2.4-2.4.4 (the S/A split algebra).

Verified exactly, symbolically, at n=4 (a generic symbolic square matrix) --
every identity below is linear/quadratic in M and holds for any n by the same
one-line argument (x^T A x = 0 for antisymmetric A), so n=4 is a faithful
instance check, not a special case the algebra depends on. What this does
NOT check: anything about a specific trained OV matrix (that is measured
data, not algebra).

Run: python3 tools/math_checks/sa_split_algebra.py
"""
import sympy as sp

n = 4
M = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"m{i}{j}"))
x = sp.Matrix(n, 1, lambda i, j: sp.Symbol(f"x{i}"))
d = sp.Matrix(n, 1, lambda i, j: sp.Symbol(f"d{i}"))  # "delta"

S = (M + M.T) / 2
A = (M - M.T) / 2


def scalar(expr):
    return sp.simplify(expr[0, 0] if hasattr(expr, "shape") else expr)


def check(name, lhs, rhs):
    diff = sp.simplify(lhs - rhs)
    if hasattr(diff, "is_zero_matrix"):
        ok = bool(diff.is_zero_matrix) or all(e == 0 for e in diff)
    else:
        ok = diff == 0
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"      residual: {diff}")
    return ok


results = []

# sec 2.1 / 2.4 core fact used everywhere: x^T A x = 0 for antisymmetric A.
results.append(check("x^T A x = 0 (antisymmetric quadratic form vanishes)",
                      scalar(x.T * A * x), 0))

# sec 2.4: discrete update x <- x + Mx.
# ||x+Mx||^2 = ||x||^2 + 2 x^T S x + ||Mx||^2
lhs = scalar((x + M * x).T * (x + M * x))
rhs = scalar(x.T * x) + 2 * scalar(x.T * S * x) + scalar((M * x).T * (M * x))
results.append(check("sec 2.4 discrete growth: ||x+Mx||^2 = ||x||^2 + 2x^TSx + ||Mx||^2",
                      lhs, rhs))

# sec 2.1: continuous flow x' = -Vx, first-order term of d/dt ||e^{-tV}x||^2 at t=0.
# e^{-tV}x = x - tVx + O(t^2), so ||.||^2 = ||x||^2 - 2t x^T V x + O(t^2).
# d/dt at t=0 = -2 x^T V x = -2 x^T S x  (V's antisymmetric part drops out).
t = sp.Symbol("t")
V = M  # reuse M as V for this convention
xt = x - t * V * x
norm2 = scalar(xt.T * xt)
first_order = sp.diff(norm2, t).subs(t, 0)
S_V = (V + V.T) / 2
results.append(check("sec 2.1 continuous flow: d/dt||e^-tV x||^2|_t=0 = -2 x^T S x",
                      first_order, -2 * scalar(x.T * S_V * x)))

# sec 2.4.1: identical form on differences delta = x - y (linearity of the update).
lhs_d = scalar((d + M * d).T * (d + M * d))
rhs_d = scalar(d.T * d) + 2 * scalar(d.T * S * d) + scalar((M * d).T * (M * d))
results.append(check("sec 2.4.1: same identity holds on delta = x - y",
                      lhs_d, rhs_d))

# sec 2.4.2: the Klein four-group table, M = S+A, M^T = S-A, -M = -S-A, -M^T = -S+A.
results.append(check("sec 2.4.2: M = S + A", M, S + A))
results.append(check("sec 2.4.2: M^T = S - A", M.T, S - A))
results.append(check("sec 2.4.2: -M = -S - A", -M, -S - A))
results.append(check("sec 2.4.2: -M^T = -S + A", -M.T, -S + A))

# sec 2.4.2: all four arms share every singular value (isometry claim) --
# check at the level of M^T M (whose eigenvalues ARE the squared singular
# values) for {M, -M} vs {M^T, -M^T}: (-M)^T(-M) = M^T M is immediate;
# (M^T)^T M^T = M M^T shares eigenvalues with M^T M (AB and BA have the same
# nonzero spectrum) -- that general fact, not a symbolic equality of the
# matrices themselves, so checked via characteristic-polynomial equality.
charpoly_MTM = (M.T * M).charpoly().as_expr()
charpoly_MMT = (M * M.T).charpoly().as_expr()
results.append(check("sec 2.4.2: char.poly(M^T M) == char.poly(M M^T) (shared singular values)",
                      sp.expand(charpoly_MTM), sp.expand(charpoly_MMT)))

# sec 2.4.3: if M = U Sigma V^T then M^T = V Sigma U^T (Sigma diagonal, so
# Sigma^T = Sigma). Checked directly via the transpose identity, generic
# rectangular sizes k < n so it is not hiding behind squareness.
k = 2
U = sp.Matrix(n, k, lambda i, j: sp.Symbol(f"u{i}{j}"))
Vv = sp.Matrix(n, k, lambda i, j: sp.Symbol(f"v{i}{j}"))
Sigma = sp.diag(*[sp.Symbol(f"s{i}") for i in range(k)])
Mfac = U * Sigma * Vv.T
results.append(check("sec 2.4.3: (U Sigma V^T)^T = V Sigma U^T",
                      Mfac.T, Vv * Sigma * U.T))

# sec 2.4.4: apply delta to M and to -M^T; verify the two stated per-arm
# formulas, then their sum and difference.
comm = M.T * M - M * M.T  # [M^T, M]
lhs_M = scalar((d + M * d).T * (d + M * d)) - scalar(d.T * d)
rhs_M = 2 * scalar(d.T * S * d) + scalar(d.T * (M.T * M) * d)
results.append(check("sec 2.4.4: Delta||.||^2 under M = 2 d^TSd + d^T(M^TM)d",
                      lhs_M, rhs_M))

NmT = -M.T
lhs_NmT = scalar((d + NmT * d).T * (d + NmT * d)) - scalar(d.T * d)
rhs_NmT = -2 * scalar(d.T * S * d) + scalar(d.T * (M * M.T) * d)
results.append(check("sec 2.4.4: Delta||.||^2 under -M^T = -2 d^TSd + d^T(MM^T)d",
                      lhs_NmT, rhs_NmT))

results.append(check("sec 2.4.4 sum: (M arm) + (-M^T arm) = d^T(M^TM + MM^T)d",
                      lhs_M + lhs_NmT, scalar(d.T * (M.T * M + M * M.T) * d)))

results.append(check("sec 2.4.4 difference: (M arm) - (-M^T arm) = 4 d^TSd + d^T[M^T,M]d",
                      lhs_M - lhs_NmT, 4 * scalar(d.T * S * d) + scalar(d.T * comm * d)))

print()
n_pass = sum(results)
print(f"{n_pass}/{len(results)} checks passed")
if n_pass != len(results):
    raise SystemExit(1)
